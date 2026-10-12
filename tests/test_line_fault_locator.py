import json, unittest
from pathlib import Path
from line_fault_locator import locate_faults

ROOT = Path(__file__).parent
CONFIG_ROOT = ROOT.parent / "config" / "v40" if (ROOT.parent / "config" / "v40").exists() else ROOT
def load(name): return json.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))
TOPO, SIG, ALARM, CFG = (load(n) for n in ("topology.json", "signal_dictionary.json", "alarm_dictionary.json", "locator_config.json"))

def ev(i, device, signal, value, unit, sec, alarm=None, quality="good"):
    d = {"event_id": i, "line_id": "AL-01", "cycle_id": "SIM-C01", "timestamp": f"2026-01-01T00:00:{sec:02d}+00:00", "device_id": device, "signal": signal, "value": value, "unit": unit, "quality": quality, "data_source": "simulation"}
    if alarm: d["alarm_code"] = alarm
    return d

class LocatorScenarios(unittest.TestCase):
    def run_case(self, events): return locate_faults(events, TOPO, SIG, ALARM, CFG)
    def top(self, result): return result["candidates"][0]["station_id"] if result["candidates"] else None
    def test_T01_simulation_feed_fault_propagates(self):
        r = self.run_case([ev("a","S01-FE01","feed_ready",False,"bool",1,"S01_NO_FEED"), ev("b","S03-PE01","part_present",False,"bool",8,"S03_PART_MISSING")])
        self.assertEqual(self.top(r), "S01"); self.assertIn(["S01","S02","S03"], r["candidates"][0]["possible_propagation_paths"])
    def test_T02_simulation_grip_then_shortage_and_interlock(self):
        r = self.run_case([ev("feed","S01-FE01","feed_ready",True,"bool",0), ev("grip","S02-VG01","grip_ok",False,"bool",2,"S02_GRIP_FAIL"), ev("short","S03-PE01","part_present",False,"bool",6,"S03_PART_MISSING"), ev("plc","PLC01","plc_interlock_active",True,"bool",9,"PLC_INTERLOCK")])
        self.assertEqual(self.top(r), "S02"); self.assertTrue(r["candidates"][0]["supporting_event_ids"])
    def test_T03_simulation_robot_servo_fault(self):
        self.assertEqual(self.top(self.run_case([ev("a","S01-FE01","feed_ready",True,"bool",0),ev("b","S02-SV01","servo_status","FAULT","code",2,"S02_SERVO_FAULT")])), "S02")
    def test_T04_simulation_press_force(self):
        self.assertEqual(self.top(self.run_case([ev("a","S01-FE01","feed_ready",True,"bool",0),ev("b","S03-LC01","press_force",15,"kN",2,"S03_PRESS_FORCE")])), "S03")
    def test_T05_simulation_vision_false_alarm(self):
        self.assertEqual(self.top(self.run_case([ev("a","S01-FE01","feed_ready",True,"bool",0),ev("b","S04-CAM01","vision_result","FAIL","enum",2,"S04_VISION_NG")])), "S04")
    def test_T06_simulation_independent_faults(self):
        r=self.run_case([ev("a","S01-FE01","feed_ready",False,"bool",1,"S01_NO_FEED"),ev("b","S03-LC01","press_force",20,"kN",1,"S03_PRESS_FORCE")])
        self.assertEqual({c["station_id"] for c in r["candidates"][:2]}, {"S01","S03"})
        self.assertFalse(any(c["possible_propagation_paths"] for c in r["candidates"]))
    def test_T07_simulation_out_of_order_input(self):
        r=self.run_case([ev("later","S03-PE01","part_present",False,"bool",8,"S03_PART_MISSING"),ev("early","S01-FE01","feed_ready",False,"bool",1,"S01_NO_FEED")])
        self.assertIn("out of timestamp order", " ".join(r["warnings"]))
    def test_T08_simulation_missing_required_signal(self):
        r=self.run_case([ev("b","S03-PE01","part_present",False,"bool",2,"S03_PART_MISSING")])
        self.assertEqual(r["diagnosis_status"], "insufficient_data"); self.assertTrue(r["missing_data"])
    def test_T09_simulation_invalid_unit_excluded(self):
        r=self.run_case([ev("a","S01-FE01","feed_ready",False,"kPa",1,"S01_NO_FEED")])
        self.assertEqual(r["diagnosis_status"], "insufficient_data"); self.assertIn("unit mismatch", " ".join(r["warnings"]))
    def test_T10_simulation_plc_precedes_equipment_alarms(self):
        r=self.run_case([ev("p","PLC01","plc_health","FAULT","code",1,"PLC_FAULT"),ev("s1","S01-FE01","feed_ready",False,"bool",4,"S01_NO_FEED"),ev("s3","S03-PE01","part_present",False,"bool",7,"S03_PART_MISSING")])
        self.assertEqual(self.top(r), "PLC01")
    def test_T11_simulation_downstream_alarm_alone_not_diagnosed(self):
        r=self.run_case([ev("b","S03-PE01","part_present",False,"bool",2,"S03_PART_MISSING")])
        self.assertEqual(r["diagnosis_status"], "insufficient_data"); self.assertFalse(r["verified_fault"])
    def test_T12_simulation_normal_evidence_contradicts_candidate(self):
        r=self.run_case([ev("bad","S02-VG01","grip_ok",False,"bool",1,"S02_GRIP_FAIL"),ev("normal","S02-VG01","grip_ok",True,"bool",4)])
        cand=next(c for c in r["candidates"] if c["station_id"]=="S02")
        self.assertIn("normal", cand["contradicting_event_ids"])

    def test_interlock_is_not_a_plc_fault(self):
        r=self.run_case([ev("feed","S01-FE01","feed_ready",True,"bool",0),ev("interlock","PLC01","plc_interlock_active",True,"bool",1,"PLC_INTERLOCK")])
        self.assertFalse(any(c["station_id"] == "PLC01" for c in r["candidates"]))

    def test_repeat_local_alarms_are_capped_but_evidence_is_retained(self):
        events=[ev(f"g{i}","S02-VG01","grip_ok",False,"bool",i+1,"S02_GRIP_FAIL") for i in range(8)]
        r=self.run_case(events); c=next(c for c in r["candidates"] if c["station_id"]=="S02")
        self.assertEqual(c["score_breakdown"]["local_abnormal"],CFG["max_local_abnormal_points_per_device_cycle"])
        self.assertEqual(set(c["supporting_event_ids"]),{e["event_id"] for e in events})

    def test_alarm_device_and_signal_pair_must_match(self):
        r=self.run_case([ev("wrong","S02-VG01","grip_ok",False,"bool",1,"S01_NO_FEED")])
        self.assertEqual(r["diagnosis_status"],"insufficient_data")
        self.assertIn("incompatible with device/signal"," ".join(r["warnings"]))

if __name__ == "__main__": unittest.main()

