import json
import unittest
from pathlib import Path

from line_fault_locator import locate_faults
from lovis_v40.integration import IntegrationError, analyze, run_management_cycle

ROOT = Path(__file__).parent
def load(path): return json.loads(path.read_text(encoding="utf-8"))
CONFIG_ROOT = ROOT.parent / "config" / "v40"
TOPO = load(CONFIG_ROOT / "topology.json")
SIGNALS = load(CONFIG_ROOT / "signal_dictionary.json")
ALARMS = load(CONFIG_ROOT / "alarm_dictionary.json")
RULES = load(CONFIG_ROOT / "locator_config.json")
T02 = load(ROOT / "fixtures" / "t02_vacuum_gripper_simulation.json")

class FakeDify:
    def __init__(self, fail_expert=False):
        self.calls=[]; self.fail_expert=fail_expert
    def manager(self, stage, payload):
        self.calls.append(("manager",stage,payload))
        if stage == "dispatch":
            return {"next_expert":"robot_expert","reason":"S02-VG01真空夹爪异常在同周期先于S03缺料；需机器人专家复核。","candidate_station_id":"S02","verified_fault":False}
        return {"summary":"机器人专家建议复核真空保持与吸附状态；目前是候选调查线索，未确诊。","verified_fault":False}
    def expert(self, route, payload):
        self.calls.append(("expert",route,payload))
        if self.fail_expert: raise IntegrationError("robot_expert is selected but its Dify expert API key is not configured")
        return {"status":"candidate_review","subsystem":"vacuum_gripper","observations":["grip_ok=false"],"recommended_checks":["复核真空压力趋势","检查吸盘与管路泄漏"],"evidence_event_ids":["grip"],"verified_fault":False}

class IntegrationTests(unittest.TestCase):
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
        self.assertEqual([c[0] for c in dify.calls],["manager","expert","manager"])
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

