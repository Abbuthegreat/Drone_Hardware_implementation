"""
Full ROS 2 / MAVLink autonomous mission (needs rclpy). Runs scripts/run_disaster_demo.py --no-video (5 processes + simulator) and checks the
offline score report: commands really travelled ROS -> MAVLink -> FC -> PyBullet, and the mission worked.
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = [pytest.mark.ros, pytest.mark.slow, pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="rclpy not available")]


def test_autonomous_mission_over_ros_and_mavlink(tmp_path):
    out = tmp_path / "demo"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_disaster_demo.py"), "--no-video", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=1500)
    rep = json.load(open(out / "score_report.json"))
    summ = json.load(open(out / "demo_summary.json"))
    assert summ["reason"] == "mission complete", summ["reason"]
    assert rep["flight"]["collision_samples_in_flight"] == 0
    assert rep["flight"]["unplanned_touchdown_samples"] == 0 and rep["flight"]["landed_disarmed_at_end"]
    assert rep["flight"]["max_altitude_m"] > 1.5
    assert rep["slam"]["ate_rmse_m"] < 0.8 and rep["slam"]["poses"] > 200
    found = {s["nearest_truth"] for s in rep["survivors"] if s["status"] == "CONFIRMED" and s["local_error_m"] < 1.0}
    assert found == {"S1", "S2"}
    assert rep["survivor_summary"]["false_or_duplicate_nodes"] == 0
    assert any(h["status"] == "CONFIRMED" for h in rep["hazards"])
    # the commands went over MAVLink to the simulated FC
    assert rep["mavlink"]["rx"]["SET_POSITION_TARGET_LOCAL_NED"] > 1000 and rep["mavlink"]["rx"]["COMMAND_LONG"] >= 2
    assert rep["navigation"]["vfh_blocked_ticks"] > 0
    assert {"TAKEOFF", "SEARCH", "HOLD_END"} <= set(rep["navigation"]["states"])
    # mission ends with RETURN -> LAND -> FC auto-disarm (config demo.return_home / end_action), and the rescue outputs exist
    assert summ["final_state"] == "DONE" and summ["end_reason"] == "mission complete" and not summ["skipped_waypoints"]
    assert summ["fc_final"]["armed"] is False and summ["mavlink_final"]["link_ok"]
    assert (out / "rescue_report.json").exists() and (out / "rescue_report.md").exists()
    assert sum(1 for _ in open(out / "command_centre" / "status.jsonl")) > 30
