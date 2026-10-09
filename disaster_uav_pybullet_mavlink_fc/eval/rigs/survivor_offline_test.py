"""
Perception-only test (no A*): the UAV is flown by a scripted FC command sequence past a survivor while the full
SLAM -> detection -> depth -> thermal fusion -> geo-semantic map chain runs on RGB-D + thermal + IMU + GPS only.
Ground truth (survivor position, UAV pose) is used ONLY after the run to score it.

    python -m eval.rigs.survivor_offline_test [--out logs/survivor_test]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from common.frames import euler_from_R  # noqa: E402
from common.geo import GeoFrame  # noqa: E402
from mapping.geo_anchor import GpsAnchor  # noqa: E402
from mapping.pipeline import PerceptionMappingPipeline  # noqa: E402
from sim.config import load_config  # noqa: E402
from sim.rgbd_camera import RGBDSensor  # noqa: E402
from sim.sim_core import UavSim  # noqa: E402
from sim.thermal_camera import ThermalSensor  # noqa: E402
from slam.adapter import SlamFrontEnd  # noqa: E402
from slam.vo_backend import OrbPnpVO  # noqa: E402
from tests.helpers import Flyer  # noqa: E402


def run(out=None, verbose=True, seconds_scale=1.0, save_kit_run_dir=None):
    cfg = load_config()
    out = out or os.path.join(ROOT, "logs", "survivor_test")
    os.makedirs(out, exist_ok=True)
    s = UavSim(cfg, world=True)
    cam, th = RGBDSensor(s), ThermalSensor(s, s.world_info["thermal_c"])
    Krgb, Kth = cam.intrinsics(), th.intrinsics()
    fe = SlamFrontEnd(cfg, OrbPnpVO(Krgb, cfg["slam"]))
    pipe = PerceptionMappingPipeline(cfg, Krgb, Kth, evidence_dir=os.path.join(out, "evidence"))
    anchor = GpsAnchor(cfg["sensor_extrinsics"]["gps"]["xyz_body_flu_m"])
    fl = Flyer(s)
    cam_div = int(round(s.hz / cam.rate_hz))
    s.hooks.append(lambda sim: (fe.feed_imu(sim.imu_sample.t, sim.imu_sample.gyro, sim.imu_sample.accel) if sim.new_imu else None,
                                anchor.add(sim.gps_sample.lat_deg, sim.gps_sample.lon_deg, sim.gps_sample.alt_m, sim.gps_sample.fix_type)
                                if sim.new_gps else None))
    s.run(0.2)
    heading0 = math.pi / 2 - euler_from_R(s.estimate.R_ned_frd)[2]
    fe.set_initial_heading(euler_from_R(s.estimate.R_ned_frd)[2])
    start_enu = s.vehicle.read_state().pos_enu.copy()
    timeline, frames_with_person = [], 0
    kit_frames, kit_traj = [], []
    frames_dropped = 0

    def grab():
        nonlocal frames_with_person
        rgb, depth = cam.capture()
        T = th.capture()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pose = fe.process(s.t, bgr, depth)
        if not pipe.map.geo.ready:
            res = anchor.try_finish(heading0)
            if res:
                pipe.map.geo.set_anchor(*res)
        if pose.T_map_body is not None:
            kit_frames.append((s.t, bgr, depth))
            kit_traj.append((s.t, pose.T_map_cam))
        if pose.T_map_body is None or not pipe.map.geo.ready:
            return
        r = pipe.process(s.t, bgr, depth, T, pose.T_map_body)
        if r["persons"]:
            frames_with_person += 1
        timeline.append((s.t, len(r["persons"]), sum(1 for p in r["persons"] if p.get("range")), len(pipe.map.survivors.tracks)))

    def stream(sec, send):
        for i in range(int(sec * seconds_scale / fl.period)):
            send()
            for _ in range(int(fl.period * s.hz)):
                s.step()
                if s.step_count % cam_div == 0:
                    grab()

    stream(2.0, lambda: fl.vel(0, 0, 0))                                   # still: gravity + GPS anchor
    assert fl.fc.arm(s.t) == 0 and fl.fc.set_mode("OFFBOARD", s.t) == 0
    n0, e0, d0 = s.state.pos_ned
    stream(9.0, lambda: fl.pos(n0, e0, d0 - cfg["demo"]["flight_altitude_m"], 0.0))
    stream(5.0, lambda: fl.vel(0, -0.6, 0))                                # west 3 m, around the wall
    stream(19.0, lambda: fl.vel(0.8, 0, 0))                                # north
    stream(5.0, lambda: fl.vel(0, 0, 0, yaw_rate=0.25))                    # turn right toward S1
    stream(7.0, lambda: fl.vel(0.0, 0, 0))                                 # observe
    stream(4.0, lambda: fl.vel(0.3, 0.3, 0))
    stream(3.0, lambda: fl.vel(0, 0, 0))

    geo = GeoFrame.from_config(cfg)
    gt = {h["id"]: h["xy_enu"] for h in s.world_info["humans"]}
    rep = {"frames_processed": len(timeline), "frames_with_person_detection": frames_with_person, "survivor_nodes": []}
    for tr in pipe.map.survivors.tracks.values():
        # nearest ground-truth survivor (OFFLINE scoring only)
        gx = {k: np.array(v) - start_enu[:2] for k, v in gt.items()}
        k, d = min(((k, float(np.linalg.norm(np.array([tr.local_x, tr.local_y]) - v))) for k, v in gx.items()), key=lambda kv: kv[1])
        lat_t, lon_t, _ = geo.ned_to_geodetic([gt[k][1], gt[k][0], 0.0])
        dn = (tr.latitude - lat_t) * 111_000.0
        de = (tr.longitude - lon_t) * 111_000.0 * math.cos(math.radians(lat_t))
        rep["survivor_nodes"].append({"track_id": tr.track_id, "status": tr.status, "obs": tr.observation_count,
                                      "confidence": round(tr.confidence, 3), "thermal_support_obs": tr.thermal_support,
                                      "local_xyz": [round(tr.local_x, 2), round(tr.local_y, 2), round(tr.local_z, 2)],
                                      "lat": tr.latitude, "lon": tr.longitude, "alt": tr.altitude,
                                      "position_uncertainty_m": round(tr.position_uncertainty, 2),
                                      "nearest_gt": k, "local_error_to_gt_m": round(d, 2), "geo_error_to_gt_m": round(math.hypot(dn, de), 2),
                                      "first_seen": round(tr.first_seen, 1), "last_seen": round(tr.last_seen, 1)})
    rep["hazards"] = [{"id": h.hazard_id, "status": h.status, "xy": [round(h.x, 2), round(h.y, 2)], "radius": round(h.radius_m, 2),
                       "max_temp_c": round(h.max_temp_c, 0), "obs": h.observation_count} for h in pipe.map.hazards.items.values()]
    rep["geo_anchor"] = pipe.map.geo.anchor
    rep["true_geo_origin_of_start_body_approx"] = list(geo.ned_to_geodetic([start_enu[1], start_enu[0], 0]))
    rep["timeline_tail"] = timeline[-5:]
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    if save_kit_run_dir:
        from eval.rigs.slam_offline_eval import save_kit_run
        # the kit expects camera poses in ITS world (first camera frame): re-express our map-frame camera poses relative to the first one
        T0 = np.linalg.inv(kit_traj[0][1])
        save_kit_run(save_kit_run_dir, kit_frames, cam, cfg, [(t, T0 @ T) for t, T in kit_traj], up_cam=fe.up_cam)
        rep["kit_run_dir"] = save_kit_run_dir
        rep["camera0_in_map"] = kit_traj[0][1][:3, 3].tolist()
        json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    s.close()
    if verbose:
        print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.out)
