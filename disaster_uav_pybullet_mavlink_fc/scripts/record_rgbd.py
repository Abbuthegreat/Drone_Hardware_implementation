"""
Record an offline dataset for SLAM / VIWO development from a scripted flight.

    python scripts/record_rgbd.py --seconds 40 --out data/run1

Writes (autonomy-legal inputs)      data/run1/rgb/000000.png, depth/000000.png (uint16, millimetres, 0 = invalid),
                                    imu.csv (t, gyro xyz, accel xyz  [FRD]), gps.csv, camera_info.json, frames.csv
Writes (EVALUATION ONLY, separate)  data/run1/eval_only/ground_truth_pose.csv  (camera pose in the map frame)
A SLAM stack must be fed only the first group. The flight is scripted through the FC's command interface
(runs non-real-time in-process; it is a dataset generator, not the autonomy path).
"""
import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.rgbd_camera import RGBDSensor  # noqa: E402
from sim.sim_core import UavSim  # noqa: E402
from tests.helpers import Flyer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "run1"))
    a = ap.parse_args()
    for d in ("rgb", "depth", "eval_only"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    s = UavSim(world=True)
    cam = RGBDSensor(s)
    fl = Flyer(s)
    n0, e0 = fl.takeoff(4.0, settle_s=8.0)
    d0 = s.state.pos_ned[2]
    json.dump(cam.intrinsics(), open(os.path.join(a.out, "camera_info.json"), "w"), indent=1)
    imu_f = open(os.path.join(a.out, "imu.csv"), "w", newline="")
    gps_f = open(os.path.join(a.out, "gps.csv"), "w", newline="")
    gt_f = open(os.path.join(a.out, "eval_only", "ground_truth_pose.csv"), "w", newline="")
    fr_f = open(os.path.join(a.out, "frames.csv"), "w", newline="")
    imu_w, gps_w, gt_w, fr_w = (csv.writer(f) for f in (imu_f, gps_f, gt_f, fr_f))
    imu_w.writerow(["t", "gx", "gy", "gz", "ax", "ay", "az"])
    gps_w.writerow(["t", "fix", "lat", "lon", "alt", "hdop", "sats"])
    gt_w.writerow(["t", "x_east", "y_north", "z_up", "qx", "qy", "qz", "qw"])
    fr_w.writerow(["idx", "t", "rgb", "depth"])
    from common.frames import quat_from_R
    cam_div = int(round(s.hz / cam.rate_hz))
    idx = 0
    t_end = s.t + a.seconds
    while s.t < t_end:
        fl.vel(1.0 * math.cos(0.3 * s.t), 1.0 * math.sin(0.3 * s.t), 0.0, yaw_rate=0.15)      # gentle looping flight
        for _ in range(int(fl.period * s.hz)):
            s.step()
            if s.new_imu:
                im = s.imu_sample
                imu_w.writerow([f"{im.t:.5f}", *map(lambda v: f"{v:.6f}", im.gyro), *map(lambda v: f"{v:.6f}", im.accel)])
            if s.new_gps and s.gps_sample.fix_type:
                g = s.gps_sample
                gps_w.writerow([f"{g.t:.3f}", g.fix_type, f"{g.lat_deg:.8f}", f"{g.lon_deg:.8f}", f"{g.alt_m:.3f}", g.hdop, g.satellites])
            if s.step_count % cam_div == 0:
                rgb, depth = cam.capture()
                cv2.imwrite(os.path.join(a.out, "rgb", f"{idx:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(a.out, "depth", f"{idx:06d}.png"), np.nan_to_num(depth * 1000.0, nan=0.0).astype(np.uint16))
                pos, R_wo = cam._pose_world_truth()
                q = quat_from_R(R_wo)
                gt_w.writerow([f"{s.t:.4f}", *map(lambda v: f"{v:.5f}", pos), q[1], q[2], q[3], q[0]])
                fr_w.writerow([idx, f"{s.t:.4f}", f"rgb/{idx:06d}.png", f"depth/{idx:06d}.png"])
                idx += 1
    for f in (imu_f, gps_f, gt_f, fr_f):
        f.close()
    s.close()
    print(f"wrote {idx} RGB-D frames to {a.out}")


if __name__ == "__main__":
    main()
