"""
Closed-loop tests (slow: minutes each). They fly the simulated UAV through the FC's setpoint interface and run the real autonomy modules:
    SLAM test       -> tracking, drift, frame gaps, timestamps          (ground truth used OFFLINE only, to score)
    survivor test   -> RGB detection, depth, thermal, geo-tag, persistence
    navigation test -> A* on the live map, VFH+ deviation, reconnect to the route
    full mission    -> everything together (in-process; the ROS/MAVLink version is tests/test_ros_demo.py)
"""
import copy
import json
import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("backend", ["orbslam3", "orb_pnp_vo"])
def test_slam_flight_tracks_without_ground_truth(tmp_path, backend):
    from slam.orbslam3_backend import orbslam3_available
    if backend == "orbslam3" and not orbslam3_available():
        pytest.skip("ORB-SLAM3 runner not built (Linux-only: bash slam/build_orbslam3.sh)")
    from eval.rigs.slam_offline_eval import run
    rep = run(out=str(tmp_path), save_kit=False, verbose=False, backend_name=backend)
    assert rep["tracked_fraction"] >= 0.98 and rep["lost"] <= 2
    assert rep["ate_rmse_m"] < 0.3 and rep["drift_percent_of_path"] < 3.0
    assert rep["timestamps_monotonic"] and rep["max_frame_gap_s"] < 0.5
    assert rep["mean_process_ms"] < 200


def test_survivor_perception_pipeline(tmp_path):
    pytest.importorskip("onnxruntime")
    from eval.rigs.survivor_offline_test import run
    rep = run(out=str(tmp_path), verbose=False)
    nodes = [n for n in rep["survivor_nodes"] if n["status"] == "CONFIRMED"]
    assert len(rep["survivor_nodes"]) == 1 and len(nodes) == 1, "one survivor -> exactly one persistent node"
    n = nodes[0]
    assert n["nearest_gt"] == "S1" and n["local_error_to_gt_m"] < 1.0 and n["geo_error_to_gt_m"] < 1.5
    assert n["obs"] >= 10 and n["thermal_support_obs"] >= 5 and n["confidence"] > 0.6
    assert 30.0 < n["lat"] < 30.001 and 75.999 < n["lon"] < 76.001                                   # near the configured origin
    assert any(h["status"] == "CONFIRMED" and h["max_temp_c"] > 300 for h in rep["hazards"])         # the fire is detected thermally


def test_navigation_astar_and_vfh_around_the_wall_and_pole(tmp_path, cfg):
    from eval.rigs.inprocess_rig import run
    c = copy.deepcopy(cfg)
    c["demo"]["search_waypoints_local"] = [{"name": "WP1 debris gap", "xy": [0.0, 12.5]}]
    c["demo"]["hold_after_mission_s"] = 1.0
    c["demo"]["return_home"], c["demo"]["end_action"] = False, "hold"     # the single-leg navigation scenario this test was written for
    rep = run(seconds=70.0, out=str(tmp_path), cfg=c, verbose=False)
    tr = np.load(os.path.join(str(tmp_path), "trajectory.npy"), allow_pickle=True)
    gx, gy = tr[:, 1].astype(float), tr[:, 2].astype(float)
    assert rep["mission_done"], rep
    assert rep["collisions"] == 0
    assert rep["planner_stats"]["replans"] >= 3                                    # A* planned on the growing live map
    assert rep["planner_stats"]["vfh_deviations"] > 0                              # VFH+ deviated from the A* direction at least once
    assert gx.min() < -1.8 or gx.max() > 3.6                                       # went AROUND the 4.6 m wall (x from -1.5 to 3.1)
    assert math.hypot(gx[-1] - 0.0, gy[-1] - 12.5) < 1.5                           # and reconnected to the route / reached WP1
    assert rep["slam_tracked_fraction"] > 0.98


def test_full_mission_in_process(tmp_path, cfg):
    pytest.importorskip("onnxruntime")
    from eval.rigs.inprocess_rig import run
    c = copy.deepcopy(cfg)
    c["demo"]["return_home"] = False                                       # the search flight these SLAM thresholds were set for; ends with a LAND
    c["demo"]["end_action"] = "land"
    rep = run(seconds=160.0, out=str(tmp_path), cfg=c, verbose=False)
    assert rep["mission_done"] and rep["collisions"] == 0
    assert rep["final_state"] == "DONE" and rep["landed_at_end"] and not rep["fc_armed_at_end"]      # landed and auto-disarmed
    assert not rep["unplanned_touchdowns"], rep["unplanned_touchdowns"]
    assert rep["end_reason"] == "mission complete" and not rep["skipped_waypoints"]
    conf = [s for s in rep["survivors"] if s["status"] == "CONFIRMED"]
    assert {s["nearest_gt"] for s in conf if s["error_m"] < 1.0} == {"S1", "S2"}
    assert len([s for s in conf if s["error_m"] >= 1.5]) == 0                       # no false / duplicate confirmed nodes
    assert rep["slam_tracked_fraction"] > 0.98 and rep["slam_ate_rmse_m"] < 0.5
    assert any(h["status"] == "CONFIRMED" for h in rep["hazards"])
    assert rep["planner_stats"]["vfh_deviations"] > 0 and rep["planner_stats"]["replans"] > 10


def test_dynamic_scenario_mission_with_return_and_landing(tmp_path, cfg):
    """Dynamic world (secondary collapse + sliding debris): the UAV must perceive the changes through RGB-D -> map and still finish
    search -> survivors -> return home -> LAND without a collision."""
    pytest.importorskip("onnxruntime")
    from eval.rigs.inprocess_rig import run
    c = copy.deepcopy(cfg)
    c["world"]["scenario"] = "dynamic"
    c["demo"]["return_home"], c["demo"]["end_action"] = True, "land"
    rep = run(seconds=300.0, out=str(tmp_path), cfg=c, verbose=False)
    assert rep["scenario"] == "dynamic"
    assert any(name == "aftershock_slab" and what == "collapsed" for _, name, what in rep["world_events"]), rep["world_events"]
    assert rep["mission_done"] and rep["final_state"] == "DONE" and rep["landed_at_end"] and not rep["fc_armed_at_end"]
    assert rep["collisions"] == 0, rep["collision_log"]
    assert not rep["unplanned_touchdowns"], rep["unplanned_touchdowns"]
    assert not rep["skipped_waypoints"] and rep["end_reason"] == "mission complete"
    conf = [s for s in rep["survivors"] if s["status"] == "CONFIRMED"]
    assert {s["nearest_gt"] for s in conf if s["error_m"] < 1.0} == {"S1", "S2"}
    assert len([s for s in conf if s["error_m"] >= 1.5]) == 0
    assert rep["slam_tracked_fraction"] > 0.98
    assert rep["planner_stats"]["vfh_deviations"] > 0 and rep["planner_stats"]["replans"] > 10
