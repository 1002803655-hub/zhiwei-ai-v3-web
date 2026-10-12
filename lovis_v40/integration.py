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
EXPERT_OPTIONAL_FIELDS = {"assessment"}
MAX_EXPERT_CALLS = 2
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
        elif stage == "followup":
            _validate_followup_decision(obj)
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
    if not re.search(r"[\u4e00-\u9fff]", obj["summary"]):
        raise IntegrationError("Management Agent summary must be in Simplified Chinese")
    if obj.get("verified_fault") is not False:
        raise IntegrationError("Management Agent summary cannot set verified_fault")


def _validate_followup_decision(obj: dict[str, Any]) -> None:
    required = {"next_action", "next_expert", "reason", "supporting_event_ids", "verified_fault"}
    if set(obj) != required:
        raise IntegrationError("Management Agent followup JSON has missing or unsupported fields")
    if obj.get("next_action") not in {"finish", "call_expert", "human_review"}:
        raise IntegrationError("Management Agent followup action is invalid")
    if not isinstance(obj.get("reason"), str) or not obj["reason"].strip():
        raise IntegrationError("Management Agent followup reason is required")
    if not isinstance(obj.get("supporting_event_ids"), list) or any(not isinstance(i, str) or not i for i in obj["supporting_event_ids"]):
        raise IntegrationError("Management Agent followup supporting_event_ids must be a string list")
    action, expert = obj["next_action"], obj["next_expert"]
    if action == "call_expert":
        if not isinstance(expert, str) or expert not in ALLOWED_ROUTES - {"human_review", "none"}:
            raise IntegrationError("Management Agent followup expert route is invalid")
        if not obj["supporting_event_ids"]:
            raise IntegrationError("Expert followup requires supporting event IDs")
    elif expert is not None or obj["supporting_event_ids"]:
        raise IntegrationError("Non-expert followup must not select an expert or cite routing evidence")
    if obj.get("verified_fault") is not False:
        raise IntegrationError("Management Agent cannot set verified_fault")


def _validate_expert_result(obj: dict[str, Any]) -> None:
    if not EXPERT_FIELDS <= set(obj) or set(obj) - EXPERT_FIELDS - EXPERT_OPTIONAL_FIELDS:
        raise IntegrationError("Expert workflow JSON has missing or unsupported fields")
    if obj.get("status") != "candidate_review" or not isinstance(obj.get("subsystem"), str) or not obj["subsystem"].strip():
        raise IntegrationError("Expert workflow JSON has invalid status or subsystem")
    for key in ("observations", "recommended_checks", "evidence_event_ids"):
        if not isinstance(obj.get(key), list) or any(not isinstance(item, str) or not item.strip() for item in obj[key]):
            raise IntegrationError(f"Expert workflow field {key} must be a list of non-empty strings")
    if obj.get("verified_fault") is not False:
        raise IntegrationError("Expert workflow cannot set verified_fault")
    if "assessment" in obj and obj["assessment"] not in {"abnormal", "normal", "inconclusive"}:
        raise IntegrationError("Expert workflow assessment is invalid")


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


def _route_evidence(locator_payload: dict[str, Any]) -> dict[str, list[str]]:
    """Evidence allowlist by expert, derived only from locator-supported raw event IDs."""
    event_by_id = {str(event.get("event_id")): event for event in locator_payload.get("supporting_and_contradicting_events", [])}
    allowed = {route: set() for route in ROUTE_COMPONENTS}
    for candidate in locator_payload.get("station_candidates", []):
        supported = set(candidate.get("supporting_event_ids", []))
        for event_id in supported:
            event = event_by_id.get(str(event_id))
            if not event:
                continue
            for route, components in ROUTE_COMPONENTS.items():
                if event.get("device_id") in components:
                    allowed[route].add(str(event_id))
    return {route: sorted(ids) for route, ids in allowed.items()}


def _validate_followup_authorization(decision: dict[str, Any], calls: list[dict[str, Any]],
                                     route_evidence: dict[str, list[str]]) -> str | None:
    """Backend authorization gate. A model decision alone never grants a route."""
    try:
        _validate_followup_decision(decision)
    except (IntegrationError, AttributeError, TypeError):
        return "invalid_followup_decision"
    if decision["next_action"] != "call_expert":
        return None
    expert = decision["next_expert"]
    already_called = {call.get("expert") for call in calls}
    if len(calls) >= MAX_EXPERT_CALLS:
        return "expert_call_limit_reached"
    if expert in already_called:
        return "repeated_expert_rejected"
    if expert not in route_evidence or not route_evidence[expert]:
        return "followup_without_locator_evidence"
    if not set(decision["supporting_event_ids"]) <= set(route_evidence[expert]):
        return "followup_evidence_not_in_locator_allowlist"
    return None


def _expert_conflicts(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    completed = [call for call in calls if call.get("status") == "succeeded" and isinstance(call.get("result"), dict)]
    for index, first in enumerate(completed):
        a = first["result"]
        if a.get("assessment") not in {"normal", "abnormal"}:
            continue
        for second in completed[index + 1:]:
            b = second["result"]
            shared = sorted(set(a.get("evidence_event_ids", [])) & set(b.get("evidence_event_ids", [])))
            if shared and b.get("assessment") in {"normal", "abnormal"} and a["assessment"] != b["assessment"]:
                conflicts.append({"experts": [first["expert"], second["expert"]], "evidence_event_ids": shared,
                                  "assessments": [a["assessment"], b["assessment"]]})
    return conflicts


def run_management_cycle(locator_result: dict[str, Any], events: list[dict[str, Any]], dify: Any) -> dict[str, Any]:
    """Run a bounded, evidence-authorized manager/expert feedback cycle."""
    payload = _evidence_payload(locator_result, events)
    empty = {"expert_call": None, "expert_result": None, "expert_calls": [],
             "management_summary": None, "locator": payload, "verified_fault": False}
    if payload["diagnosis_status"] == "insufficient_data" or not payload["station_candidates"]:
        return {**empty, "status": "human_review", "decision": {"next_expert": "human_review", "reason": "定位证据不足，暂不自动调用专家。"}}
    try:
        decision = dify.manager("dispatch", payload)
    except IntegrationError as exc:
        return {**empty, "status": "management_agent_unavailable", "decision": None, "error": str(exc)}
    try:
        _validate_decision(decision)
    except (IntegrationError, AttributeError, TypeError):
        return {**empty, "status": "invalid_agent_response", "decision": None, "error": "Management Agent dispatch schema validation failed"}
    route = decision["next_expert"]
    if route not in {"human_review", "none"} and not _route_compatible(route, decision["candidate_station_id"], payload):
        return {**empty, "status": "incompatible_agent_route", "decision": decision, "error": "Selected expert is incompatible with the chosen locator candidate"}
    if route in {"none", "human_review"}:
        return {**empty, "status": "awaiting_review", "decision": decision}

    route_evidence = _route_evidence(payload)
    if route not in route_evidence or not route_evidence[route]:
        return {**empty, "status": "incompatible_agent_route", "decision": decision,
                "error": "Selected expert has no locator-supported event evidence"}
    calls: list[dict[str, Any]] = []

    def call_expert(expert: str, reason: str, evidence_ids: list[str]) -> tuple[dict[str, Any] | None, str | None]:
        trace = {"expert": expert, "call_reason": reason, "supporting_event_ids": list(evidence_ids),
                 "result": None, "status": "running"}
        calls.append(trace)
        expert_input = {"locator": payload, "selected_route": expert, "decision_reason": reason,
                        "supporting_event_ids": list(evidence_ids)}
        try:
            result = dify.expert(expert, expert_input)
            _validate_expert_result(result)
            original_ids = {str(e.get("event_id")) for e in payload["supporting_and_contradicting_events"]}
            if not set(result["evidence_event_ids"]) <= original_ids:
                raise IntegrationError("Expert cited evidence absent from the locator result")
            trace["result"], trace["status"] = result, "succeeded"
            return result, None
        except IntegrationError as exc:
            trace["status"], trace["error"] = "failed", str(exc)
            return None, str(exc)
        except (TypeError, AttributeError):
            trace["status"], trace["error"] = "failed", "Expert workflow response failed schema validation"
            return None, trace["error"]

    first_evidence = route_evidence[route]
    first_result, first_error = call_expert(route, decision["reason"], first_evidence)
    if first_result is None:
        status = "invalid_expert_response" if "schema" in (first_error or "") or "cited evidence" in (first_error or "") else "expert_unavailable"
        return {**empty, "status": status, "decision": decision,
                "expert_call": {"route": route, "called": calls[0]["status"] == "running"},
                "expert_calls": calls, "error": first_error}

    available = {expert: ids for expert, ids in route_evidence.items() if ids and expert != route}
    followup_payload = {"locator": payload, "first_expert_result": first_result,
                        "experts_called": [route], "evidence_gaps": payload.get("missing_data", []),
                        "allowed_experts": available, "max_expert_calls": MAX_EXPERT_CALLS,
                        "expert_calls_remaining": MAX_EXPERT_CALLS - len(calls)}
    followup_decision = None
    cycle_status = "completed"
    try:
        followup_decision = dify.manager("followup", followup_payload)
        rejection = _validate_followup_authorization(followup_decision, calls, route_evidence)
        if rejection:
            cycle_status = rejection
        elif followup_decision["next_action"] == "human_review":
            cycle_status = "human_review"
        elif followup_decision["next_action"] == "call_expert":
            second_route = followup_decision["next_expert"]
            _, second_error = call_expert(second_route, followup_decision["reason"], followup_decision["supporting_event_ids"])
            if second_error:
                cycle_status = "partially_completed"
    except IntegrationError as exc:
        cycle_status = "invalid_followup_decision" if "followup" in str(exc).lower() else "followup_unavailable"
        followup_decision = {"error": str(exc)}

    conflicts = _expert_conflicts(calls)
    if conflicts:
        cycle_status = "human_review_conflict"
    summary_input = {"locator": payload, "decision": decision, "followup_decision": followup_decision,
                     "expert_calls": calls, "expert_conflicts": conflicts, "cycle_status": cycle_status}
    summary = None
    summary_error = None
    try:
        summary = dify.manager("summarize", summary_input)
        _validate_summary(summary)
    except IntegrationError as exc:
        summary_error = str(exc)
        if cycle_status == "completed":
            cycle_status = "summary_unavailable"
    except (AttributeError, TypeError):
        summary_error = "Management summary failed schema validation"
        if cycle_status == "completed":
            cycle_status = "invalid_agent_response"

    succeeded = [c for c in calls if c["status"] == "succeeded"]
    first = calls[0] if calls else None
    result = {**empty, "status": cycle_status, "decision": decision, "followup_decision": followup_decision,
              "expert_call": {"route": route, "called": bool(first and first["status"] == "succeeded")},
              "expert_result": first_result, "expert_calls": calls, "management_summary": summary,
              "expert_conflicts": conflicts}
    if summary_error:
        result["error"] = summary_error
    if cycle_status == "partially_completed":
        result["error"] = next((c.get("error") for c in calls if c["status"] == "failed"), "Second expert failed")
    return result


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

