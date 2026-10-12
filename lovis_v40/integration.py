"""Streamlit-to-local-locator-to-Dify orchestration for Lovis V4.0.

The locator is the only scoring engine. Dify receives its structured evidence,
chooses an expert route, and summarizes a specialist's structured response.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from line_fault_locator import locate_faults

EXPERT_ENV = {
    "robot_expert": "LOVIS_V40_ROBOT_EXPERT_API_KEY",
    "motion_control_expert": "LOVIS_V40_MOTION_EXPERT_API_KEY",
    "press_expert": "LOVIS_V40_PRESS_EXPERT_API_KEY",
    "vision_expert": "LOVIS_V40_VISION_EXPERT_API_KEY",
}
ALLOWED_ROUTES = set(EXPERT_ENV) | {"human_review", "none"}
MANAGER_DECISION_FIELDS = {"next_expert", "reason", "candidate_station_id", "verified_fault"}
MANAGER_SUMMARY_FIELDS = {"summary", "verified_fault"}
EXPERT_FIELDS = {"status", "subsystem", "observations", "recommended_checks", "evidence_event_ids", "verified_fault"}
ROUTE_COMPONENTS = {
    "robot_expert": {"S02-RB01", "S02-VG01"},
    "motion_control_expert": {"S02-SV01"},
    "press_expert": {"S03-SV01", "S03-PE01", "S03-LC01"},
    "vision_expert": {"S04-CAM01"},
}


class IntegrationError(RuntimeError):
    """Safe, user-displayable integration failure (never includes credentials)."""


class DifyClient:
    def __init__(self, base_url: str, manager_key: str, expert_keys: dict[str, str], timeout: float = 25):
        self.base_url = base_url.rstrip("/")
        self.manager_key = manager_key
        self.expert_keys = expert_keys
        self.timeout = timeout

    def _post(self, path: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url.startswith("https://") and not self.base_url.startswith("http://127.0.0.1") and not self.base_url.startswith("http://localhost"):
            raise IntegrationError("DIFY_BASE_URL must use HTTPS except for localhost experiments")
        request = Request(self.base_url + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise IntegrationError(f"Dify returned HTTP {exc.code} for {path}") from None
        except (URLError, TimeoutError, OSError, json.JSONDecodeError):
            raise IntegrationError(f"Dify request failed for {path}; check endpoint/network and retry") from None
        if not isinstance(parsed, dict):
            raise IntegrationError("Dify returned a non-object response")
        return parsed

    def manager(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._post("/v1/chat-messages", self.manager_key, {
            "query": json.dumps({"stage": stage, "incident": payload}, ensure_ascii=False),
            "inputs": {}, "response_mode": "blocking", "user": "lovis-v40-streamlit"})
        answer = result.get("answer")
        if not isinstance(answer, str):
            raise IntegrationError("Management Agent returned no answer text")
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.I)
        try:
            obj = json.loads(answer)
        except json.JSONDecodeError:
            raise IntegrationError("Management Agent response was not valid JSON") from None
        if not isinstance(obj, dict):
            raise IntegrationError("Management Agent JSON must be an object")
        if stage == "dispatch":
            _validate_decision(obj)
        elif stage == "summarize":
            _validate_summary(obj)
        return obj

    def expert(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        key = self.expert_keys.get(route)
        if not key:
            raise IntegrationError(f"{route} is selected but its Dify expert API key is not configured")
        result = self._post("/v1/workflows/run", key, {
            "inputs": {"incident_payload": json.dumps(payload, ensure_ascii=False)},
            "response_mode": "blocking", "user": "lovis-v40-streamlit"})
        data = result.get("data", {})
        if not isinstance(data, dict) or data.get("status") not in {"succeeded", "completed"}:
            raise IntegrationError("Expert workflow did not return a successful run status")
        outputs = data.get("outputs", {}) if isinstance(data, dict) else {}
        raw = outputs.get("result_json") or outputs.get("text")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raise IntegrationError("Expert workflow response was not valid JSON") from None
        if not isinstance(raw, dict):
            raise IntegrationError("Expert workflow response must be a JSON object")
        _validate_expert_result(raw)
        return raw


def _validate_decision(obj: dict[str, Any]) -> None:
    if set(obj) != MANAGER_DECISION_FIELDS:
        raise IntegrationError("Management Agent dispatch JSON has missing or unsupported fields")
    route, reason, station = obj["next_expert"], obj["reason"], obj["candidate_station_id"]
    if route not in ALLOWED_ROUTES or not isinstance(reason, str) or not reason.strip():
        raise IntegrationError("Management Agent dispatch JSON has invalid route or reason")
    if route in {"human_review", "none"}:
        if station is not None:
            raise IntegrationError("Human-review dispatch must not select a candidate station")
    elif not isinstance(station, str) or not station:
        raise IntegrationError("Expert dispatch must name candidate_station_id")
    if obj["verified_fault"] is not False:
        raise IntegrationError("Management Agent cannot set verified_fault")


def _validate_summary(obj: dict[str, Any]) -> None:
    if set(obj) != MANAGER_SUMMARY_FIELDS or not isinstance(obj.get("summary"), str) or not obj["summary"].strip():
        raise IntegrationError("Management Agent summary JSON has missing or unsupported fields")
    if obj.get("verified_fault") is not False:
        raise IntegrationError("Management Agent summary cannot set verified_fault")


def _validate_expert_result(obj: dict[str, Any]) -> None:
    if set(obj) != EXPERT_FIELDS:
        raise IntegrationError("Expert workflow JSON has missing or unsupported fields")
    if obj.get("status") != "candidate_review" or not isinstance(obj.get("subsystem"), str) or not obj["subsystem"].strip():
        raise IntegrationError("Expert workflow JSON has invalid status or subsystem")
    for key in ("observations", "recommended_checks", "evidence_event_ids"):
        if not isinstance(obj.get(key), list) or any(not isinstance(item, str) or not item.strip() for item in obj[key]):
            raise IntegrationError(f"Expert workflow field {key} must be a list of non-empty strings")
    if obj.get("verified_fault") is not False:
        raise IntegrationError("Expert workflow cannot set verified_fault")


def _route_compatible(route: str, station_id: str, payload: dict[str, Any]) -> bool:
    if route not in ROUTE_COMPONENTS:
        return False
    for candidate in payload.get("station_candidates", []):
        if candidate.get("station_id") != station_id:
            continue
        associated = set(candidate.get("associated_component_ids", []))
        component = candidate.get("component_id")
        if component:
            associated.add(component)
        if station_id == "PLC01":
            return False
        return bool(associated & ROUTE_COMPONENTS[route])
    return False


def _evidence_payload(locator_result: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(e.get("event_id")): e for e in events if isinstance(e, dict) and e.get("event_id") is not None}
    selected_ids = set()
    for candidate in locator_result.get("station_candidates", locator_result.get("candidates", [])) + locator_result.get("shared_candidates", []):
        selected_ids.update(candidate.get("supporting_event_ids", []))
        selected_ids.update(candidate.get("contradicting_event_ids", []))
    return {
        "line_id": locator_result.get("line_id"), "incident_id": locator_result.get("incident_id"),
        "algorithm_version": locator_result.get("algorithm_version"),
        "data_source": locator_result.get("data_source"), "diagnosis_status": locator_result.get("diagnosis_status"),
        "verified_fault": False,
        "station_candidates": locator_result.get("station_candidates", locator_result.get("candidates", [])),
        "shared_candidates": locator_result.get("shared_candidates", []),
        "missing_data": locator_result.get("missing_data", []), "warnings": locator_result.get("warnings", []),
        "supporting_and_contradicting_events": [by_id[i] for i in sorted(selected_ids) if i in by_id],
    }


def run_management_cycle(locator_result: dict[str, Any], events: list[dict[str, Any]], dify: Any) -> dict[str, Any]:
    """Ask manager to select at most one expert; call it and request a summary."""
    payload = _evidence_payload(locator_result, events)
    if payload["diagnosis_status"] == "insufficient_data" or not payload["station_candidates"]:
        return {"status": "human_review", "decision": {"next_expert": "human_review", "reason": "定位证据不足，暂不自动调用专家。"},
                "expert_call": None, "expert_result": None, "management_summary": None, "locator": payload}
    try:
        decision = dify.manager("dispatch", payload)
    except IntegrationError as exc:
        return {"status": "management_agent_unavailable", "decision": None, "expert_call": None,
                "expert_result": None, "management_summary": None, "error": str(exc), "locator": payload}
    try:
        _validate_decision(decision)
    except (IntegrationError, AttributeError, TypeError):
        return {"status": "invalid_agent_response", "decision": None, "expert_call": None,
                "expert_result": None, "management_summary": None,
                "error": "Management Agent dispatch schema validation failed", "locator": payload}
    route = decision["next_expert"]
    if route not in {"human_review", "none"} and not _route_compatible(route, decision["candidate_station_id"], payload):
        return {"status": "incompatible_agent_route", "decision": decision, "expert_call": None,
                "expert_result": None, "management_summary": None,
                "error": "Selected expert is incompatible with the chosen locator candidate", "locator": payload}
    if route in {"none", "human_review"}:
        return {"status": "awaiting_review", "decision": decision, "expert_call": None,
                "expert_result": None, "management_summary": None, "locator": payload}
    expert_input = {"locator": payload, "selected_route": route, "decision_reason": decision.get("reason", "")}
    try:
        expert_result = dify.expert(route, expert_input)
    except IntegrationError as exc:
        return {"status": "expert_unavailable", "decision": decision, "expert_call": {"route": route, "called": False},
                "expert_result": None, "management_summary": None, "error": str(exc), "locator": payload}
    try:
        _validate_expert_result(expert_result)
    except (IntegrationError, AttributeError, TypeError):
        return {"status": "invalid_expert_response", "decision": decision, "expert_call": {"route": route, "called": True},
                "expert_result": None, "management_summary": None,
                "error": "Expert workflow failed schema validation", "locator": payload}
    summary_input = {"locator": payload, "decision": decision, "expert_route": route, "expert_result": expert_result}
    try:
        summary = dify.manager("summarize", summary_input)
    except IntegrationError as exc:
        return {"status": "summary_unavailable", "decision": decision, "expert_call": {"route": route, "called": True},
                "expert_result": expert_result, "management_summary": None, "error": str(exc), "locator": payload}
    try:
        _validate_summary(summary)
    except (IntegrationError, AttributeError, TypeError):
        return {"status": "invalid_agent_response", "decision": decision, "expert_call": {"route": route, "called": True},
                "expert_result": expert_result, "management_summary": None,
                "error": "Expert or management summary failed schema validation", "locator": payload}
    return {"status": "completed", "decision": decision, "expert_call": {"route": route, "called": True},
            "expert_result": expert_result, "management_summary": summary, "locator": payload}


def analyze(events, topology, signal_dictionary, alarm_dictionary, config, dify=None):
    """Run local deterministic locator first, then optional Dify management cycle."""
    locator_result = locate_faults(events, topology, signal_dictionary, alarm_dictionary, config)
    if dify is None:
        base_url = os.getenv("DIFY_BASE_URL", "")
        manager_key = os.getenv("LOVIS_V40_MANAGER_API_KEY", "")
        if not base_url or not manager_key:
            return {"status": "dify_not_configured", "locator": _evidence_payload(locator_result, events)}
        keys = {route: os.getenv(env_name, "") for route, env_name in EXPERT_ENV.items()}
        dify = DifyClient(base_url, manager_key, keys)
    cycle = run_management_cycle(locator_result, events, dify)
    cycle["locator_algorithm_version"] = locator_result.get("algorithm_version")
    return cycle

