"""
Run the in-process closed-loop rig (sim + SLAM + perception + map + A* + VFH+ + mission, no ROS processes) with outputs kept on disk.

    python -m eval.rigs.run_inprocess --scenario dynamic --seconds 300 --out logs/final/inproc_dyn [--slam orbslam3|orb_pnp_vo]
Ground truth is used only inside the rig for offline scoring.
"""
import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="static", choices=["static", "dynamic"])
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dyn-memory", action="store_true", help="enable map.dynamic_memory")
    ap.add_argument("--slam", default="auto", choices=["auto", "orbslam3", "orb_pnp_vo"])
    a = ap.parse_args()
    os.environ["UAV_RIG_SLAM"] = a.slam
    from eval.rigs.inprocess_rig import run
    from sim.config import load_config
    c = copy.deepcopy(load_config())
    c["world"]["scenario"] = a.scenario
    c["demo"]["return_home"], c["demo"]["end_action"] = True, "land"
    if a.dyn_memory:
        c["map"]["dynamic_memory"]["enabled"] = True
    rep = run(seconds=a.seconds, out=a.out, cfg=c, verbose=False)
    keys = ("slam_backend", "scenario", "sim_time_s", "final_state", "landed_at_end", "fc_armed_at_end", "collisions", "collision_log", "unplanned_touchdowns",
            "slam_ate_rmse_m", "slam_ate_max_m", "slam_tracked_fraction", "planner_stats", "world_events")
    print(json.dumps({k: rep[k] for k in keys}, indent=1, default=float))


if __name__ == "__main__":
    main()
