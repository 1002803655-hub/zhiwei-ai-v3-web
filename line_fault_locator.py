"""Deterministic, evidence-oriented AL-01 incident triage (experimental rules)."""
from __future__ import annotations

from datetime import datetime, timezone
from collections import defaultdict
import re

ALGORITHM_VERSION = "4.0.0"


def _parse_time(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO 8601 string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return dt.astimezone(timezone.utc)


def _as_dict(value):
    if isinstance(value, dict):
        return value
    raise ValueError("configuration must be an object")


def locate_faults(events, topology, signal_dictionary, alarm_dictionary, config):
    """Return ranked investigation candidates; never asserts a verified fault.

    Events are dictionaries with event_id, line_id, cycle_id, timestamp, device_id,
    signal, value, unit, quality and (optionally) alarm_code/data_source.
    Scores are rule points, not probabilities.
    """
    topology, signals = _as_dict(topology), _as_dict(signal_dictionary)
    alarms, cfg = _as_dict(alarm_dictionary), _as_dict(config)
    line_id = topology.get("line_id", "AL-01")
    stations = topology.get("stations", [])
    station_by_id = {s["station_id"]: s for s in stations}
    devices = {s["station_id"]: set(s.get("components", [])) | {s["station_id"]} for s in stations}
    devices["PLC01"] = set(topology.get("shared_components", ["PLC01"])) | {"PLC01"}
    order = {s["station_id"]: i for i, s in enumerate(stations)}
    flow = topology.get("material_flow", [])
    successors = defaultdict(list)
    for edge in flow:
        successors[edge[0]].append(edge[1])
    weights = cfg.get("weights", {})
    threshold = cfg.get("minimum_evidence_points", 1)
    max_cycle_gap = cfg.get("cycle_time_window_seconds", 120)
    temporal_window = cfg.get("temporal_window_seconds", 30)
    validation_errors, warnings, missing = [], [], []
    valid = []
    seen = set()
    for index, event in enumerate(events if isinstance(events, list) else []):
        if not isinstance(event, dict):
            validation_errors.append(f"event[{index}] is not an object")
            continue
        required = ("event_id", "line_id", "cycle_id", "timestamp", "device_id", "signal", "value", "unit")
        absent = [k for k in required if k not in event]
        if absent:
            validation_errors.append(f"event[{index}] missing {','.join(absent)}")
            continue
        if event["event_id"] in seen:
            validation_errors.append(f"duplicate event_id: {event['event_id']}")
            continue
        seen.add(event["event_id"])
        if event["line_id"] != line_id:
            validation_errors.append(f"{event['event_id']}: unknown line {event['line_id']}")
            continue
        device = event["device_id"]
        station = next((sid for sid, comps in devices.items() if device in comps), None)
        if station is None:
            validation_errors.append(f"{event['event_id']}: unknown device/component {device}")
            continue
        spec = signals.get(event["signal"])
        if not spec:
            validation_errors.append(f"{event['event_id']}: unknown signal {event['signal']}")
            continue
        if event.get("unit") != spec.get("unit"):
            validation_errors.append(f"{event['event_id']}: unit mismatch for {event['signal']}")
            continue
        value_type = spec.get("value_type")
        invalid_value = (
            value_type == "number" and (isinstance(event["value"], bool) or not isinstance(event["value"], (int, float)))
            or value_type == "boolean" and not isinstance(event["value"], bool)
            or value_type == "string" and not isinstance(event["value"], str)
        )
        if invalid_value:
            validation_errors.append(f"{event['event_id']}: signal requires {value_type} value")
            continue
        alarm_code = event.get("alarm_code")
        if alarm_code:
            alarm_spec = alarms.get("alarms", {}).get(alarm_code)
            if alarm_spec is None:
                validation_errors.append(f"{event['event_id']}: unknown alarm code {alarm_code}")
                continue
            if device not in alarm_spec.get("device_ids", []) or event["signal"] not in alarm_spec.get("signals", []):
                validation_errors.append(f"{event['event_id']}: alarm {alarm_code} is incompatible with device/signal")
                continue
        try:
            dt = _parse_time(event["timestamp"])
        except (ValueError, TypeError) as exc:
            validation_errors.append(f"{event['event_id']}: invalid timestamp ({exc})")
            continue
        quality = event.get("quality", "good")
        if quality not in cfg.get("accepted_quality", ["good"]):
            warnings.append(f"{event['event_id']}: excluded sensor sample with quality={quality}")
            continue
        row = dict(event, _dt=dt, _station=station)
        valid.append(row)
    sorted_events = sorted(valid, key=lambda e: (e["_dt"], e["event_id"]))
    if valid != sorted_events:
        warnings.append("input events were out of timestamp order; analyzed in timestamp order")
    cycles = defaultdict(list)
    for e in sorted_events:
        cycles[e["cycle_id"]].append(e)
    if len(cycles) > 1:
        warnings.append("multiple cycle_id values supplied; cycles analyzed independently")
    candidates = defaultdict(lambda: {"score": 0.0, "breakdown": defaultdict(float), "support": set(), "contradict": set(), "paths": [], "components": set()})
    abnormal_by_cycle = defaultdict(list)
    alarms_by_code = alarms.get("alarms", {})
    local_points_used = defaultdict(float)
    propagation_pairs_scored = set()
    plc_edges_scored = set()
    for e in sorted_events:
        spec = signals[e["signal"]]
        abnormal = False
        rule = spec.get("abnormal")
        if rule == "false_is_abnormal": abnormal = e["value"] is False
        elif rule == "true_is_abnormal": abnormal = e["value"] is True
        elif rule == "outside_range": abnormal = e["value"] < spec["normal_min"] or e["value"] > spec["normal_max"]
        elif rule == "enum_not_normal": abnormal = e["value"] not in spec.get("normal_values", [])
        alarm = alarms_by_code.get(e.get("alarm_code")) if e.get("alarm_code") else None
        # Protective interlocks are control actions. Only an alarm explicitly
        # declared as a local fault can add abnormal evidence for its device.
        abnormal = abnormal or bool(alarm and alarm.get("local_abnormal", False))
        e["_abnormal"] = abnormal
        if abnormal:
            abnormal_by_cycle[e["cycle_id"]].append(e)
        c = candidates[e["_station"]]
        points = weights.get("local_abnormal", 0) if abnormal else weights.get("normal_counter_evidence", 0)
        key = "local_abnormal" if abnormal else "normal_counter_evidence"
        if abnormal:
            cap = cfg.get("max_local_abnormal_points_per_device_cycle", float("inf"))
            device_key = (e["cycle_id"], e["_station"])
            remaining = max(0.0, cap - local_points_used[device_key])
            points = min(points, remaining)
            local_points_used[device_key] += points
        if points:
            c["score"] += points
            c["breakdown"][key] += points
        (c["support"] if abnormal else c["contradict"]).add(e["event_id"])
        if abnormal and e["device_id"] != e["_station"]:
            c["components"].add(e["device_id"])
    for cycle_id, rows in abnormal_by_cycle.items():
        rows.sort(key=lambda e: (e["_dt"], e["event_id"]))
        if rows:
            earliest = rows[0]
            if earliest["_station"] == "PLC01":
                for e in rows:
                    if e["_station"] != "PLC01":
                        edge_key = (cycle_id, e["_station"])
                        if edge_key in plc_edges_scored:
                            continue
                        plc_edges_scored.add(edge_key)
                        c = candidates["PLC01"]
                        p = weights.get("shared_plc_precedes", 0)
                        c["score"] += p
                        c["breakdown"]["shared_plc_precedes"] += p
                        c["support"].add(earliest["event_id"])
                        c["paths"].append(["PLC01", e["_station"]])
            for source in rows:
                if source["_station"] == "PLC01":
                    continue
                source_station = source["_station"]
                for target in rows:
                    target_station = target["_station"]
                    if target_station not in order or source_station not in order or order[target_station] <= order[source_station]:
                        continue
                    gap = (target["_dt"] - source["_dt"]).total_seconds()
                    # Strictly later downstream observations may be consistent with
                    # propagation; simultaneous events do not establish ordering.
                    if 0 < gap <= temporal_window:
                        path = [source_station]
                        current = source_station
                        while current != target_station:
                            nxt = next((n for n in successors[current] if order.get(n, -1) <= order[target_station]), None)
                            if not nxt: break
                            path.append(nxt); current = nxt
                        if path[-1] == target_station:
                            pair_key = (cycle_id, source_station, target_station)
                            if pair_key in propagation_pairs_scored:
                                continue
                            propagation_pairs_scored.add(pair_key)
                            c = candidates[source_station]
                            p = weights.get("downstream_sequence", 0)
                            c["score"] += p
                            c["breakdown"]["downstream_sequence"] += p
                            c["support"].update([source["event_id"], target["event_id"]])
                            c["paths"].append(path)
                            target_c = candidates[target_station]
                            q = weights.get("downstream_can_be_propagated", 0)
                            target_c["score"] -= q
                            target_c["breakdown"]["downstream_can_be_propagated"] -= q
                            target_c["contradict"].add(source["event_id"])
    if not sorted_events:
        missing.append("no valid events")
    for device, entry in signals.items():
        if entry.get("required_for_location") and not any(e["signal"] == device for e in sorted_events):
            missing.append(f"required signal absent: {device}")
    if len(cycles) > 1:
        missing.append("events from multiple cycles cannot support a single causal sequence")
    results = []
    for device, c in candidates.items():
        if c["score"] < threshold:
            continue
        station = device if device in station_by_id or device == "PLC01" else next((s for s, ds in devices.items() if device in ds), device)
        component = next(iter(c["components"])) if len(c["components"]) == 1 else (None if device == station else device)
        results.append({"ranking_scope": "shared_controller" if station == "PLC01" else "station",
                        "station_id": station, "component_id": component,
                        "associated_component_ids": sorted(c["components"]),
                        "ranking_score": round(c["score"], 3), "score_breakdown": dict(sorted(c["breakdown"].items())),
                        "supporting_event_ids": sorted(c["support"]), "contradicting_event_ids": sorted(c["contradict"]),
                        "possible_propagation_paths": c["paths"]})
    results.sort(key=lambda c: (-c["ranking_score"], c["station_id"], c["component_id"] or ""))
    ambiguous = len(results) > 1 and results[0]["ranking_score"] - results[1]["ranking_score"] < cfg.get("ambiguity_margin_points", 1)
    enough = bool(results) and bool(sorted_events) and not missing
    status = "insufficient_data" if not enough else ("multiple_candidates" if ambiguous else "candidates_available")
    if validation_errors:
        warnings.extend(validation_errors)
    source = sorted({str(e.get("data_source", "unspecified")) for e in sorted_events})
    station_candidates = [c for c in results if c["station_id"] != "PLC01"]
    shared_candidates = [c for c in results if c["station_id"] == "PLC01"]
    for rank, candidate in enumerate(station_candidates, 1):
        candidate["station_rank"] = rank
    for rank, candidate in enumerate(shared_candidates, 1):
        candidate["shared_rank"] = rank
    return {"line_id": line_id, "incident_id": (sorted_events[0]["cycle_id"] if sorted_events else None),
            "algorithm_version": ALGORITHM_VERSION, "data_source": source, "diagnosis_status": status,
            # Keep `candidates` unchanged for backwards compatibility. These typed
            # lists are views over the same rule scores, not a new scoring model.
            "candidates": results, "station_candidates": station_candidates,
            "shared_candidates": shared_candidates,
            "missing_data": sorted(set(missing)), "warnings": warnings,
            "verified_fault": False}

