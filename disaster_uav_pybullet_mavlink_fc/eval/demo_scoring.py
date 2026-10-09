"""
OFFLINE scoring of a demo run against ground truth (EVALUATION ONLY - runs after the flight, never during it).

    python -m eval.demo_scoring logs/demo/run_...

Inputs written by the run: sim/ground_truth.csv, sim/world_truth.json, slam/slam_pose.csv, perception/survivor_map.jsonl,
nav/nav_log.csv, sim/fc_mavlink.log, sim/fc_mavlink_summary.json.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.config import load_config  # noqa: E402
from common.geo import GeoFrame  # noqa: E402
from eval.ground_truth import load_ground_truth_csv  # noqa: E402


def _last_json_line(path):
    last = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def score(run):
    cfg = load_config()
    geo = GeoFrame.from_config(cfg)
    gt = load_ground_truth_csv(os.path.join(run, "sim", "ground_truth.csv"))
    world = json.load(open(os.path.join(run, "sim", "world_truth.json")))
    # ground truth in ENU (x East, y North, z Up); local frame origin = take-off position (first sample with the UAV on the ground)
    gx, gy, gz = gt["pe"], gt["pn"], -gt["pd"]
    start = np.array([gx[0], gy[0], gz[0]])
    rep = {"run": run}
    # ---- SLAM trajectory error
    sp = os.path.join(run, "slam", "slam_pose.csv")
    if os.path.exists(sp):
        rows = list(csv.DictReader(open(sp)))
        if rows:
            t = np.array([float(r["t"]) for r in rows])
            sx = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
            g = np.stack([np.interp(t, gt["t"], gx) - start[0], np.interp(t, gt["t"], gy) - start[1], np.interp(t, gt["t"], gz) - start[2]], 1)
            err = np.linalg.norm(sx - g, axis=1)
            path = float(np.sum(np.linalg.norm(np.diff(g, axis=0), axis=1)))
            rep["slam"] = {"poses": len(t), "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))), "ate_max_m": float(err.max()), "final_error_m": float(err[-1]),
                           "gt_path_m": path, "drift_pct_of_path": float(err[-1] / max(path, 1e-6) * 100), "first_t": float(t[0]), "last_t": float(t[-1])}
    # ---- survivors
    snap = _last_json_line(os.path.join(run, "perception", "survivor_map.jsonl")) if os.path.exists(os.path.join(run, "perception", "survivor_map.jsonl")) else None
    truth = {h["id"]: np.array(h["xy_enu"]) for h in world["humans"]}
    rep["survivors"] = []
    if snap:
        for s in snap["survivors"]:
            k, d = min(((k, float(np.linalg.norm(np.array([s["local_x"], s["local_y"]]) - (v - start[:2])))) for k, v in truth.items()), key=lambda kv: kv[1])
            lat_t, lon_t, _ = geo.ned_to_geodetic([truth[k][1], truth[k][0], 0.0])
            dn = (s["latitude"] - lat_t) * 111_000.0
            de = (s["longitude"] - lon_t) * 111_000.0 * math.cos(math.radians(lat_t))
            rep["survivors"].append({"track_id": s["track_id"], "status": s["status"], "observations": s["observation_count"], "confidence": round(s["confidence"], 3),
                                     "thermal_support_obs": s["thermal_support"], "nearest_truth": k, "local_error_m": round(d, 2),
                                     "geo_error_m": round(math.hypot(dn, de), 2), "lat": s["latitude"], "lon": s["longitude"], "alt": s["altitude"],
                                     "first_seen_s": round(s["first_seen"], 1), "confirmed_at_s": round(s["confirmed_at"], 1)})
        rep["hazards"] = []
        for h in snap["hazards"]:
            best = min(world.get("hazards", []) or [{"id": "?", "xy_enu": [0, 0]}], key=lambda z: np.linalg.norm(np.array([h["x"], h["y"]]) - (np.array(z["xy_enu"]) - start[:2])))
            rep["hazards"].append({"id": h["hazard_id"], "status": h["status"], "kind": h["kind"], "nearest_truth": best["id"],
                                   "error_m": round(float(np.linalg.norm(np.array([h["x"], h["y"]]) - (np.array(best["xy_enu"]) - start[:2]))), 2),
                                   "max_temp_c": h["max_temp_c"]})
    confirmed = [s for s in rep["survivors"] if s["status"] == "CONFIRMED"]
    matched = {s["nearest_truth"] for s in confirmed if s["local_error_m"] < 1.5}
    rep["survivor_summary"] = {"truth_count": len(truth), "confirmed_nodes": len(confirmed), "truth_survivors_found": sorted(matched),
                               "false_or_duplicate_nodes": len([s for s in confirmed if s["local_error_m"] >= 1.5]) + max(0, len(confirmed) - len(matched) - len([s for s in confirmed if s["local_error_m"] >= 1.5]))}
    # ---- navigation / flight
    nl = os.path.join(run, "nav", "nav_log.csv")
    if os.path.exists(nl):
        rows = list(csv.DictReader(open(nl)))
        blocked = [r for r in rows if r["vfh_blocked"] == "1"]
        rep["navigation"] = {"ticks": len(rows), "vfh_blocked_ticks": len(blocked),
                             "vfh_max_deviation_deg": max([abs(float(r["vfh_dev_deg"])) for r in blocked], default=0.0),
                             "max_speed_mps": max([float(r["speed"]) for r in rows], default=0.0),
                             "states": sorted(set(r["state"] for r in rows)), "targets": list(dict.fromkeys(r["target"] for r in rows))}
    air = gz - start[2] > 0.5
    contacts = gt["contacts"] > 0
    been_up = np.maximum.accumulate(gz - start[2] > 1.0)
    offboard_touchdown = been_up & (gt["landed"] > 0) & (gt["armed"] > 0) & (gt["mode"] == "OFFBOARD")    # on the ground while still flying the mission
    rep["flight"] = {"sim_duration_s": float(gt["t"][-1]), "max_altitude_m": float(np.max(gz - start[2])),
                     "unplanned_touchdown_samples": int(np.sum(offboard_touchdown)),
                     "landed_disarmed_at_end": bool(gt["landed"][-1] > 0 and gt["armed"][-1] == 0),
                     "final_gt_local_xy": [round(float(gx[-1] - start[0]), 2), round(float(gy[-1] - start[1]), 2)],
                     "collision_samples_in_flight": int(np.sum(air & contacts)), "gt_path_m": float(np.sum(np.hypot(np.diff(gx), np.diff(gy))))}
    sm = os.path.join(run, "sim", "fc_mavlink_summary.json")
    if os.path.exists(sm):
        rep["mavlink"] = json.load(open(sm))
    elif os.path.exists(os.path.join(run, "sim", "fc_mavlink.log")):        # simulator killed before writing its summary: count the FC log
        import collections
        import re
        rx = collections.Counter(m.group(1) for m in (re.search(r"RX (\w+)", ln) for ln in open(os.path.join(run, "sim", "fc_mavlink.log"))) if m)
        rep["mavlink"] = {"rx": dict(rx), "source": "fc_mavlink.log (summary file missing)"}
    json.dump(rep, open(os.path.join(run, "score_report.json"), "w"), indent=1, default=float)
    return rep


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1]), indent=1, default=float))
