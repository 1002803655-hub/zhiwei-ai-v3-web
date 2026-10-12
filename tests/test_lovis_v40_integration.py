import json
import unittest
from pathlib import Path

from line_fault_locator import locate_faults
from lovis_v40.integration import (IntegrationError, analyze, run_management_cycle,
                                   _validate_followup_authorization, _route_evidence, _evidence_payload)

ROOT = Path(__file__).parent
def load(path): return json.loads(path.read_text(encoding="utf-8"))
CONFIG_ROOT = ROOT.parent / "config" / "v40"
TOPO = load(CONFIG_ROOT / "topology.json")
SIGNALS = load(CONFIG_ROOT / "signal_dictionary.json")
ALARMS = load(CONFIG_ROOT / "alarm_dictionary.json")
RULES = load(CONFIG_ROOT / "locator_config.json")
T02 = load(ROOT / "fixtures" / "t02_vacuum_gripper_simulation.json")

class FakeDify:
    def __init__(self, fail_expert=False, followup=None, assessments=None, fail_routes=None):
        self.calls=[]; self.fail_expert=fail_expert; self.followup=followup
        self.assessments=assessments or {}; self.fail_routes=set(fail_routes or [])
    def manager(self, stage, payload):
        self.calls.append(("manager",stage,payload))
        if stage == "dispatch":
            return {"next_expert":"robot_expert","reason":"S02-VG01真空夹爪异常在同周期先于S03缺料；需机器人专家复核。","candidate_station_id":"S02","verified_fault":False}
        if stage == "followup":
            return self.followup or {"next_action":"finish","next_expert":None,"reason":"当前证据支持完成本领域复核。","supporting_event_ids":[],"verified_fault":False}
        return {"summary":"机器人专家建议复核真空状态；结果仅为候选线索，尚未经现场验证。","verified_fault":False}
    def expert(self, route, payload):
        self.calls.append(("expert",route,payload))
        if self.fail_expert or route in self.fail_routes: raise IntegrationError(f"{route} API unavailable")
        is_robot=route == "robot_expert"
        evidence=["grip"] if is_robot else ["servo"]
        return {"status":"candidate_review","subsystem":"vacuum_gripper" if is_robot else "servo_motion","observations":["simulation observation"],"recommended_checks":["复核现场信号"],"evidence_event_ids":evidence,"verified_fault":False, **({"assessment":self.assessments[route]} if route in self.assessments else {})}

class IntegrationTests(unittest.TestCase):
    def with_servo_fault(self):
        return T02 + [{"event_id":"servo","line_id":"AL-01","cycle_id":"SIM-C01","timestamp":"2026-01-01T00:00:01Z","device_id":"S02-SV01","signal":"servo_status","value":"FAULT","unit":"code","quality":"good","alarm_code":"S02_SERVO_FAULT","data_source":"simulation"}]

    def run_t02_with(self, events, dify):
        located=locate_faults(events,TOPO,SIGNALS,ALARMS,RULES)
        return run_management_cycle(located,events,dify)

    def test_t02_level_views_separate_plc_and_rank_stations(self):
        result=locate_faults(T02,TOPO,SIGNALS,ALARMS,RULES)
        self.assertEqual([x["station_rank"] for x in result["station_candidates"]],[1,2])
        self.assertEqual(result["station_candidates"][0]["station_id"],"S02")
        self.assertEqual(result["station_candidates"][0]["component_id"],"S02-VG01")
        self.assertEqual(result["station_candidates"][0]["ranking_scope"],"station")
        self.assertEqual(result["station_candidates"][0]["associated_component_ids"],["S02-VG01"])
        self.assertEqual(result["station_candidates"][0]["ranking_score"],8)
        self.assertEqual(result["shared_candidates"],[])
    def test_t02_manager_decides_robot_expert_and_returns_structured_chain(self):
        dify=FakeDify(); result=analyze(T02,TOPO,SIGNALS,ALARMS,RULES,dify=dify)
        self.assertEqual(result["status"],"completed")
        self.assertEqual(result["decision"]["next_expert"],"robot_expert")
        self.assertEqual(result["expert_call"],{"route":"robot_expert","called":True})
        self.assertEqual(result["expert_result"]["status"],"candidate_review")
        self.assertFalse(result["expert_result"]["verified_fault"])
        self.assertFalse(result["locator"]["verified_fault"])
        self.assertIn("grip", [e["event_id"] for e in dify.calls[0][2]["supporting_and_contradicting_events"]])
        self.assertEqual([c[0] for c in dify.calls],["manager","expert","manager","manager"])
        self.assertEqual([c["expert"] for c in result["expert_calls"]],["robot_expert"])
        self.assertEqual(result["followup_decision"]["next_action"],"finish")
    def test_insufficient_data_does_not_call_agent_or_expert(self):
        dify=FakeDify(); result=analyze([T02[2]],TOPO,SIGNALS,ALARMS,RULES,dify=dify)
        self.assertEqual(result["status"],"human_review"); self.assertEqual(dify.calls,[])
    def test_expert_failure_is_reported_without_claiming_completion(self):
        result=run_management_cycle(locate_faults(T02,TOPO,SIGNALS,ALARMS,RULES),T02,FakeDify(fail_expert=True))
        self.assertEqual(result["status"],"expert_unavailable")
        self.assertEqual(result["expert_call"],{"route":"robot_expert","called":False})
        self.assertIsNone(result["expert_result"])
    def test_unsupported_agent_route_is_rejected(self):
        class Invalid(FakeDify):
            def manager(self, stage, payload): return {"next_expert":"unknown"}
        result=run_management_cycle(locate_faults(T02,TOPO,SIGNALS,ALARMS,RULES),T02,Invalid())
        self.assertEqual(result["status"],"invalid_agent_response"); self.assertIsNone(result["expert_call"])
    def test_management_agent_failure_is_structured_and_no_expert_call(self):
        class Broken(FakeDify):
            def manager(self, stage, payload): raise IntegrationError("Dify request failed")
        result=run_management_cycle(locate_faults(T02,TOPO,SIGNALS,ALARMS,RULES),T02,Broken())
        self.assertEqual(result["status"],"management_agent_unavailable"); self.assertIsNone(result["expert_call"])
    def test_motion_control_expert_branch_is_kept_available(self):
        class Motion(FakeDify):
            def manager(self, stage, payload):
                self.calls.append(("manager",stage,payload))
                return {"next_expert":"motion_control_expert","reason":"伺服信号线索","candidate_station_id":"S02","verified_fault":False}
        events=[e for e in T02 if e.get("device_id") != "S02-VG01" and e.get("device_id") != "S03-PE01"]
        events.append({"event_id":"servo","line_id":"AL-01","cycle_id":"SIM-C01","timestamp":"2026-01-01T00:00:02+00:00","device_id":"S02-SV01","signal":"servo_status","value":"FAULT","unit":"code","quality":"good","data_source":"simulation"})
        located=locate_faults(events,TOPO,SIGNALS,ALARMS,RULES)
        dify=Motion(); result=run_management_cycle(located,events,dify)
        self.assertEqual(result["expert_call"]["route"],"motion_control_expert")
        self.assertIn(("expert","motion_control_expert",dify.calls[1][2]),dify.calls)

    def test_t03_servo_only_calls_motion_expert(self):
        events=[e for e in T02 if e.get("device_id") not in {"S02-VG01","S03-PE01"}]
        events.append(self.with_servo_fault()[-1])
        class Motion(FakeDify):
            def manager(self, stage, payload):
                self.calls.append(("manager",stage,payload))
                if stage == "dispatch": return {"next_expert":"motion_control_expert","reason":"S02伺服异常证据","candidate_station_id":"S02","verified_fault":False}
                return super().manager(stage,payload)
        dify=Motion(); result=self.run_t02_with(events,dify)
        self.assertEqual([x["expert"] for x in result["expert_calls"]],["motion_control_expert"])

    def test_t07_two_domains_can_call_second_expert_from_raw_evidence(self):
        events=self.with_servo_fault()
        follow={"next_action":"call_expert","next_expert":"motion_control_expert","reason":"原始定位证据同时包含伺服异常","supporting_event_ids":["servo"],"verified_fault":False}
        dify=FakeDify(followup=follow); result=self.run_t02_with(events,dify)
        self.assertEqual([x["expert"] for x in result["expert_calls"]],["robot_expert","motion_control_expert"])
        self.assertEqual([x["status"] for x in result["expert_calls"]],["succeeded","succeeded"])
        self.assertTrue(all(x["supporting_event_ids"] for x in result["expert_calls"]))
        self.assertFalse(result["verified_fault"])

    def test_t08_expert_suggestion_without_raw_domain_evidence_cannot_route(self):
        follow={"next_action":"call_expert","next_expert":"motion_control_expert","reason":"机器人专家建议检查伺服","supporting_event_ids":["servo-suggested-only"],"verified_fault":False}
        dify=FakeDify(followup=follow); result=self.run_t02_with(T02,dify)
        self.assertEqual(result["status"],"followup_without_locator_evidence")
        self.assertEqual([x["expert"] for x in result["expert_calls"]],["robot_expert"])

    def test_t09_conflicting_expert_assessments_are_retained_for_review(self):
        events=self.with_servo_fault()
        follow={"next_action":"call_expert","next_expert":"motion_control_expert","reason":"存在独立伺服异常证据","supporting_event_ids":["servo"],"verified_fault":False}
        dify=FakeDify(followup=follow,assessments={"robot_expert":"abnormal","motion_control_expert":"normal"})
        # Both specialists may comment on the same locator event; preserve contradiction, do not adjudicate it.
        original=dify.expert
        def conflicting(route,payload):
            result=original(route,payload)
            if route=="robot_expert": result["evidence_event_ids"]=["servo"]
            return result
        dify.expert=conflicting
        result=self.run_t02_with(events,dify)
        self.assertEqual(result["status"],"human_review_conflict")
        self.assertEqual(result["expert_conflicts"][0]["evidence_event_ids"],["servo"])
        self.assertEqual(len(result["expert_calls"]),2)

    def test_t10_second_expert_failure_keeps_first_result_as_partial(self):
        events=self.with_servo_fault()
        follow={"next_action":"call_expert","next_expert":"motion_control_expert","reason":"存在伺服事件","supporting_event_ids":["servo"],"verified_fault":False}
        result=self.run_t02_with(events,FakeDify(followup=follow,fail_routes={"motion_control_expert"}))
        self.assertEqual(result["status"],"partially_completed")
        self.assertEqual(result["expert_calls"][0]["status"],"succeeded")
        self.assertEqual(result["expert_calls"][1]["status"],"failed")
        self.assertIsNotNone(result["expert_result"])
        self.assertFalse(result["verified_fault"])

    def test_t11_repeated_expert_route_is_rejected(self):
        follow={"next_action":"call_expert","next_expert":"robot_expert","reason":"重复尝试","supporting_event_ids":["grip"],"verified_fault":False}
        result=self.run_t02_with(T02,FakeDify(followup=follow))
        self.assertEqual(result["status"],"repeated_expert_rejected")
        self.assertEqual(len(result["expert_calls"]),1)

    def test_t12_call_beyond_maximum_is_rejected_by_backend_guard(self):
        decision={"next_action":"call_expert","next_expert":"motion_control_expert","reason":"第三次调用","supporting_event_ids":["servo"],"verified_fault":False}
        calls=[{"expert":"robot_expert"},{"expert":"press_expert"}]
        allow={"motion_control_expert":["servo"]}
        self.assertEqual(_validate_followup_authorization(decision,calls,allow),"expert_call_limit_reached")

    def test_manager_verified_true_or_missing_schema_is_rejected(self):
        for decision in (
            None,
            {"next_expert":"robot_expert","reason":"x","candidate_station_id":"S02","verified_fault":True},
            {"next_expert":"robot_expert","reason":"x"},
            {"next_expert":"robot_expert","reason":"x","candidate_station_id":"S02","verified_fault":False,"ranking_score":9999},
        ):
            class Bad(FakeDify):
                def manager(self, stage, payload): return decision
            result=run_management_cycle(locate_faults(T02,TOPO,SIGNALS,ALARMS,RULES),T02,Bad())
            self.assertEqual(result["status"],"invalid_agent_response")
            self.assertFalse(result["locator"]["verified_fault"])

    def test_incompatible_expert_route_is_rejected_before_call(self):
        class Wrong(FakeDify):
            def manager(self, stage, payload):
                return {"next_expert":"vision_expert","reason":"wrong equipment","candidate_station_id":"S02","verified_fault":False}
        dify=Wrong(); result=run_management_cycle(locate_faults(T02,TOPO,SIGNALS,ALARMS,RULES),T02,dify)
        self.assertEqual(result["status"],"incompatible_agent_route")
        self.assertFalse(any(call[0]=="expert" for call in dify.calls))

    def test_malformed_expert_or_expert_verified_true_never_completes(self):
        for expert_result in (
            "not json",
            {"status":"candidate_review","subsystem":"robot","observations":[],"recommended_checks":[],"verified_fault":False},
            {"status":"candidate_review","subsystem":"robot","observations":[],"recommended_checks":[],"evidence_event_ids":[],"verified_fault":True},
        ):
            class BadExpert(FakeDify):
                def expert(self, route, payload): return expert_result
            dify=BadExpert(); result=run_management_cycle(locate_faults(T02,TOPO,SIGNALS,ALARMS,RULES),T02,dify)
            self.assertNotEqual(result["status"],"completed")
            self.assertEqual([call[1] for call in dify.calls if call[0]=="manager"],["dispatch"])

    def test_management_summary_verified_true_or_malformed_never_completes(self):
        for summary in ({"summary":"done","verified_fault":True}, {"summary":"done"}, "bad"):
            class BadSummary(FakeDify):
                def manager(self, stage, payload):
                    if stage=="dispatch": return super().manager(stage,payload)
                    return summary
            result=run_management_cycle(locate_faults(T02,TOPO,SIGNALS,ALARMS,RULES),T02,BadSummary())
            self.assertNotEqual(result["status"],"completed")

if __name__ == "__main__": unittest.main()

