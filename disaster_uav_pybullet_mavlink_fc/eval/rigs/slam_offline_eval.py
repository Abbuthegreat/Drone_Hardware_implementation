"""
SLAM test rig (calibration/validation, not autonomy).

    python -m eval.rigs.slam_offline_eval [--seconds 60] [--out logs/slam_test] [--backend orb_pnp_vo]

Flies: take-off -> slow forward -> yaw rotation -> lateral move -> stop, in the disaster scene, through the FC's
command interface. The SLAM front-end receives ONLY RGB-D + IMU + the FC heading at start. After the flight the
estimated trajectory is compared with the simulator's ground truth - strictly OFFLINE, for the report.
Also writes a kit-format run folder (rgb/, depth/, associations.txt, meta.json, orbslam.yaml, CameraTrajectory.txt) so the
attached kit's own postprocess.py can process the very same flight.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from common.frames import quat_from_R  # noqa: E402
from sim.config import load_config  # noqa: E402
from sim.rgbd_camera import RGBDSensor  # noqa: E402
from sim.sim_core import UavSim  # noqa: E402
from slam import kit  # noqa: E402
from slam.adapter import SlamFrontEnd  # noqa: E402
from slam.vo_backend import OrbPnpVO  # noqa: E402
from tests.helpers import Flyer  # noqa: E402


def save_kit_run(run_dir, frames, cam, cfg, traj_cam, up_cam=None):
    """Write a run folder in the attached kit's format (as record.py would)."""
    os.makedirs(os.path.join(run_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "depth"), exist_ok=True)
    rec = kit.record_module()
    with open(os.path.join(run_dir, "frames.csv"), "w") as f:
        f.write("idx,t,t_color_ms,t_depth_ms\n")
        for i, (t, bgr, depth) in enumerate(frames):
            cv2.imwrite(os.path.join(run_dir, "rgb", f"{i:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(run_dir, "depth", f"{i:06d}.png"), np.nan_to_num(depth * 1000.0, nan=0).astype(np.uint16))
            f.write(f"{i},{t:.6f},{t*1000:.3f},{t*1000:.3f}\n")
    rec.write_associations(run_dir)
    k = cam.intrinsics()
    intr = dict(width=k["width"], height=k["height"], fx=k["fx"], fy=k["fy"], ppx=k["cx"], ppy=k["cy"], model="none", coeffs=[0] * 5)
    rec.write_orbslam_yaml(os.path.join(run_dir, "orbslam.yaml"), intr, 0.001, int(cam.rate_hz), 0.05, cfg["slam"]["n_features"])
    meta = dict(device="PyBullet simulated RGB-D", width=k["width"], height=k["height"], fps=cam.rate_hz, depth_scale=0.001,
                color_intrinsics=intr, up_camera_frame=None if up_cam is None else [float(v) for v in up_cam])
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=1)
    with open(os.path.join(run_dir, "CameraTrajectory.txt"), "w") as f:        # EuRoC style like ORB-SLAM3: t_ns tx ty tz qx qy qz qw
        for t, T in traj_cam:
            q = quat_from_R(T[:3, :3])
            f.write(f"{t*1e9:.0f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} {q[0]:.6f}\n")


def run(seconds=60.0, out=None, world=True, save_kit=True, verbose=True, backend_name=None):
    cfg = load_config()
    out = out or os.path.join(ROOT, "logs", "slam_test")
    os.makedirs(out, exist_ok=True)
    s = UavSim(cfg, world=world)
    cam = RGBDSensor(s)
    k = cam.intrinsics()
    backend_name = backend_name or cfg["slam"]["backend"]
    if backend_name == "orbslam3":
        from slam.orbslam3_backend import OrbSlam3Backend
        backend = OrbSlam3Backend(k, cfg)
    else:
        backend = OrbPnpVO(k, cfg["slam"])
    fe = SlamFrontEnd(cfg, backend)
    fl = Flyer(s)
    cam_div = int(round(s.hz / cam.rate_hz))

    # --- FC heading at start (this is legitimate MAVLink ATTITUDE data) and gravity from the IMU while on the ground
    from common.frames import euler_from_R
    s.run(0.2)
    fe.set_initial_heading(euler_from_R(s.estimate.R_ned_frd)[2])
    start_enu = s.vehicle.read_state().pos_enu.copy()

    log, frames, traj_cam = [], [], []
    stats = {"frames": 0, "ok": 0, "lost": 0, "dt_ms": []}
    phase_t = [0.0]

    def imu_hook(sim_):
        if sim_.new_imu:
            im = sim_.imu_sample
            fe.feed_imu(im.t, im.gyro, im.accel)
    s.hooks.append(imu_hook)

    def grab():
        rgb, depth = cam.capture()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        t = s.t
        t0 = time.perf_counter()
        pose = fe.process(t, bgr, depth)
        dt = (time.perf_counter() - t0) * 1e3
        st = s.vehicle.read_state()                       # ground truth: RECORDED for offline evaluation only
        gt = st.pos_enu - start_enu
        frames.append((t, bgr, depth))
        stats["frames"] += 1
        stats["dt_ms"].append(dt)
        if pose.T_map_body is not None:
            stats["ok"] += 1
            traj_cam.append((t, pose.T_map_cam))
            e = pose.T_map_body[:3, 3]
            log.append([t, *e, *gt, pose.info.get("inliers", 0)])
        else:
            stats["lost"] += 1
            log.append([t, *([np.nan] * 3), *gt, 0])

    def stream(sec, send):
        n = int(sec / fl.period)
        for i in range(n):
            send()
            for _ in range(int(fl.period * s.hz)):
                s.step()
                if s.step_count % cam_div == 0:
                    grab()

    # --- scripted flight through the FC command interface
    stream(1.5, lambda: fl.vel(0, 0, 0))                           # sit still: gravity + heading + first frames
    assert fl.fc.arm(s.t) == 0 and fl.fc.set_mode("OFFBOARD", s.t) == 0
    n0, e0, d0 = s.state.pos_ned
    alt = cfg["demo"]["flight_altitude_m"]
    stream(9.0, lambda: fl.pos(n0, e0, d0 - alt, 0.0))            # take-off
    stream(9.0, lambda: fl.vel(0.6, 0, 0))                          # slow forward (North)
    stream(4.0, lambda: fl.vel(0, 0, 0))
    yaw0 = math.atan2(0, 1)
    stream(5.0, lambda: fl.vel(0, 0, 0, yaw_rate=0.3))              # rotate ~85 deg (clockwise)
    stream(6.0, lambda: fl.vel(0, 0.5, 0))                          # lateral (East) - heading is now ~85 deg so this is a mix
    stream(3.0, lambda: fl.vel(0, 0, 0))
    truth_final = s.vehicle.read_state().pos_enu - start_enu

    L = np.array(log, float)
    ok = np.isfinite(L[:, 1])
    err = np.linalg.norm(L[ok, 1:4] - L[ok, 4:7], axis=1)
    t_ok = L[ok, 0]
    path_len = float(np.sum(np.linalg.norm(np.diff(L[ok, 4:7], axis=0), axis=1)))
    ts = np.diff(L[:, 0])
    rep = {
        "backend": backend.name, "frames": stats["frames"], "tracked": stats["ok"], "lost": stats["lost"],
        "tracked_fraction": stats["ok"] / max(stats["frames"], 1),
        "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))) if ok.any() else None,
        "ate_max_m": float(err.max()) if ok.any() else None,
        "final_error_m": float(err[-1]) if ok.any() else None,
        "gt_path_length_m": path_len,
        "drift_percent_of_path": float(err[-1] / path_len * 100) if ok.any() and path_len > 0 else None,
        "mean_process_ms": float(np.mean(stats["dt_ms"])),
        "max_frame_gap_s": float(ts.max()) if len(ts) else None,
        "timestamps_monotonic": bool(np.all(ts > 0)),
        "truth_final_m": truth_final.tolist(),
    }
    with open(os.path.join(out, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "slam_x", "slam_y", "slam_z", "gt_x", "gt_y", "gt_z", "inliers"])
        w.writerows(L.tolist())
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
    if save_kit:
        save_kit_run(os.path.join(out, "kit_run"), frames, cam, cfg, traj_cam)
    s.close()
    if verbose:
        print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default=None, choices=["orb_pnp_vo", "orbslam3"])
    ap.add_argument("--no-kit-run", action="store_true")
    a = ap.parse_args()
    run(out=a.out, save_kit=not a.no_kit_run, backend_name=a.backend)
