"""
Live diagnostic of the ROS 2 sensor + SLAM chain (run while the system is up, e.g. during scripts/run_disaster_demo.py):

    python tools/diagnose_slam.py [--seconds 8] [--out logs/diag]

CAMERA / DEPTH / IMU / TF / SLAM / POSE / GPS: PASS|FAIL, with the numbers behind each verdict, plus an RGB|depth montage
(proof of camera orientation) and a ground-truth-firewall check (the SLAM node must not subscribe to the FC pose).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, NavSatFix
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from ros2_bridge.ros_util import img_to_np, stamp_to_s
from ros2_bridge.tf_util import urdf_fixed_joints


class Diag(Node):
    def __init__(self):
        super().__init__("diagnose_slam")
        self.d = {}
        self.cnt = {}
        self.poses = []
        for name, typ, topic in (("rgb", Image, "/uav/camera/color/image_raw"), ("depth", Image, "/uav/camera/depth/image_raw"),
                                 ("rgb_info", CameraInfo, "/uav/camera/color/camera_info"), ("depth_info", CameraInfo, "/uav/camera/depth/camera_info"),
                                 ("imu", Imu, "/uav/imu"), ("gps", NavSatFix, "/uav/gps/fix"), ("status", String, "/uav/slam/status")):
            self.create_subscription(typ, topic, lambda m, n=name: self.on(n, m), 20)
        self.create_subscription(PoseStamped, "/uav/slam/pose", lambda m: self.poses.append((stamp_to_s(m.header.stamp), m.pose.position.x, m.pose.position.y, m.pose.position.z, m.header.frame_id)), 20)
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)

    def on(self, n, m):
        self.cnt[n] = self.cnt.get(n, 0) + 1
        self.d.setdefault(n, []).append(m)
        self.d[n] = self.d[n][-200:]


def verdict(res, name, ok, msg):
    res[name] = {"pass": bool(ok), "detail": msg}
    print(f"{name:7s}: {'PASS' if ok else 'FAIL'}   {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "diag"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    n = Diag()
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.05)
    res, T = {}, a.seconds

    # CAMERA
    rgb, info = n.d.get("rgb", []), n.d.get("rgb_info", [])
    if rgb and info:
        m, ci = rgb[-1], info[-1]
        ok = m.encoding == "rgb8" and ci.header.frame_id == m.header.frame_id == "rgbd_camera_optical_link" and (ci.width, ci.height) == (m.width, m.height) and ci.k[0] > 0
        verdict(res, "CAMERA", ok, f"{m.width}x{m.height} {m.encoding} frame={m.header.frame_id} fx={ci.k[0]:.2f} fy={ci.k[4]:.2f} cx={ci.k[2]:.2f} cy={ci.k[5]:.2f} "
                                   f"{n.cnt['rgb'] / T:.1f} Hz wall stamp={stamp_to_s(m.header.stamp):.3f}s")
    else:
        verdict(res, "CAMERA", False, "no color image / camera_info received")

    # DEPTH
    dep = n.d.get("depth", [])
    if dep and rgb:
        dm = dep[-1]
        arr = img_to_np(dm)
        valid = arr[np.isfinite(arr) & (arr > 0)]
        same_stamp = any(abs(stamp_to_s(r.header.stamp) - stamp_to_s(dm.header.stamp)) < 1e-6 for r in rgb)
        ok = dm.encoding == "32FC1" and valid.size > 0.3 * arr.size and 0.1 < np.median(valid) < 50 and same_stamp
        verdict(res, "DEPTH", ok, f"{dm.encoding} metres, valid {100 * valid.size / arr.size:.0f}%, median {np.median(valid):.2f} m, range {valid.min():.2f}-{valid.max():.2f} m, "
                                   f"same stamp as an RGB frame: {same_stamp}  (ORB-SLAM3 gets uint16 mm = depth x 1000, DepthMapFactor 1000)")
        r = [m for m in rgb if abs(stamp_to_s(m.header.stamp) - stamp_to_s(dm.header.stamp)) < 1e-6]
        if r:
            c = cv2.resize(img_to_np(r[-1]), (arr.shape[1], arr.shape[0]))
            dv = np.clip(np.nan_to_num(arr, nan=0), 0, 15) / 15.0
            dc = cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(a.out, "rgb_depth_montage.png"), np.hstack([cv2.cvtColor(c, cv2.COLOR_RGB2BGR), dc]))
    else:
        verdict(res, "DEPTH", False, "no depth image received")

    # IMU
    im = n.d.get("imu", [])
    if im:
        acc = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in im])
        mag = np.linalg.norm(acc, axis=1).mean()
        ts = np.array([stamp_to_s(m.header.stamp) for m in im])
        rate = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-9)                 # in SIMULATION time (stamps), not wall time (sim may run at rtf < 1)
        verdict(res, "IMU", rate > 50 and 7 < mag < 14, f"{rate:.0f} Hz (sim time), |accel| mean {mag:.2f} m/s^2 (m/s^2, gravity ~9.8), frame={im[-1].header.frame_id}")
    else:
        verdict(res, "IMU", False, "no IMU received")

    # TF: numeric check of base_link -> rgbd_camera_optical_link against the URDF, and duplicate-broadcaster check
    try:
        tf = n.buf.lookup_transform("base_link", "rgbd_camera_optical_link", rclpy.time.Time())
        j = {c: (R, t) for _, c, R, t in urdf_fixed_joints()}
        R1, t1 = j["rgbd_camera_link"]
        R2, t2 = j["rgbd_camera_optical_link"]
        t_exp = t1 + R1 @ t2
        t_got = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        err = float(np.linalg.norm(t_exp - t_got))
        pubs_static = n.count_publishers("/tf_static")
        pubs_dyn = n.count_publishers("/tf")
        ok = err < 1e-6 and pubs_static == 1 and pubs_dyn == 1
        verdict(res, "TF", ok, f"base_link->rgbd_camera_optical_link t={np.round(t_got, 4).tolist()} (URDF {np.round(t_exp, 4).tolist()}, err {err:.1e} m); "
                               f"publishers on /tf_static={pubs_static}, /tf={pubs_dyn} (must be 1 each)")
        try:
            n.buf.lookup_transform("slam_map", "rgbd_camera_optical_link", rclpy.time.Time())
        except Exception as e:
            verdict(res, "TF", False, f"slam_map -> camera lookup failed: {e}")
    except Exception as e:
        verdict(res, "TF", False, f"lookup base_link->rgbd_camera_optical_link failed: {e}")

    # SLAM
    st = n.d.get("status", [])
    if st:
        s = json.loads(st[-1].data)
        verdict(res, "SLAM", s.get("backend") == "orbslam3" and s.get("state") == "OK", f"backend={s.get('backend')} state={s.get('state')} tracked={s.get('tracked')} lost={s.get('lost')} "
                                                                                       f"frames={s.get('frames')} latency={s.get('latency_ms')} ms")
    else:
        verdict(res, "SLAM", False, "no /uav/slam/status")

    # POSE
    P = np.array([p[1:4] for p in n.poses]) if n.poses else np.zeros((0, 3))
    if len(P) >= 2:
        verdict(res, "POSE", n.poses[-1][4] == "slam_map" and np.isfinite(P).all(), f"{len(P)} poses in {T:.0f} s, frame={n.poses[-1][4]}, last={np.round(P[-1], 2).tolist()}, "
                                                                                    f"moved {np.linalg.norm(P[-1] - P[0]):.2f} m during the window")
    else:
        verdict(res, "POSE", False, f"{len(P)} poses received")

    # GPS
    g = n.d.get("gps", [])
    verdict(res, "GPS", bool(g) and g[-1].status.status >= 0 and abs(g[-1].latitude) > 0, f"{n.cnt.get('gps', 0)} fixes, last lat={g[-1].latitude:.7f} lon={g[-1].longitude:.7f} alt={g[-1].altitude:.1f}" if g else "no GPS fix")

    # firewall: who subscribes to ground-truth-like topics
    fw = {t: [i.node_name for i in n.get_subscriptions_info_by_topic(t)] for t in ("/uav/odom",)}
    leak = [x for x in fw["/uav/odom"] if "slam" in x or "perception" in x or "nav" in x]
    verdict(res, "FIREWALL", not leak, f"/uav/odom (FC pose) subscribers: {fw['/uav/odom']} (SLAM / mapping / nav nodes must not be among them)")
    json.dump(res, open(os.path.join(a.out, "diagnose_slam.json"), "w"), indent=1)
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if all(v["pass"] for v in res.values()) else 1)


if __name__ == "__main__":
    main()
