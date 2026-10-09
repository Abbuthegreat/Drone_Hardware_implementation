"""
SLAM node: RGB-D + IMU -> estimated UAV pose.

Subscribes : /uav/camera/color/image_raw (rgb8), /uav/camera/depth/image_raw (32FC1 m), /uav/camera/color/camera_info,
             /uav/imu (FLU), /uav/telemetry (only the FC heading at start, for map yaw alignment)
Publishes  : /uav/slam/pose    geometry_msgs/PoseStamped  UAV BODY pose in `slam_map` (x East, y North, z Up, origin = take-off point)
             /uav/slam/status  std_msgs/String JSON  {state, inliers, features, keyframes, tracked, lost, latency_ms, backend, ...}
             /uav/slam/path    nav_msgs/Path (decimated)
Never subscribes to ground truth. Backend: config slam.backend (orb_pnp_vo runs here; orbslam3 = the kit's ORB-SLAM3, Linux build).
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from ros2_bridge.ros_util import img_to_np, matrix_to_pose, s_to_stamp, stamp_to_s
from ros2_bridge.tf_util import make_tf, static_transforms
from common.config import load_config
from slam.adapter import SlamFrontEnd
from slam.vo_backend import OrbPnpVO


def make_backend(cfg, K, kind=None):
    kind = kind or cfg["slam"]["backend"]
    if kind == "orb_pnp_vo":
        return OrbPnpVO(K, cfg["slam"])
    if kind == "orbslam3":
        from slam.orbslam3_backend import OrbSlam3Backend
        return OrbSlam3Backend(K, cfg)
    raise ValueError(kind)


class SlamNode(Node):
    def __init__(self):
        super().__init__("uav_slam")
        self.cfg = load_config()
        self.log_dir = self.declare_parameter("log_dir", "").value
        # backend: "" = config slam.backend; "auto" = configured backend if it can run here, else the orb_pnp_vo fallback (Windows)
        from slam.orbslam3_backend import resolve_backend
        self.backend_kind, self.backend_note = resolve_backend(self.cfg["slam"]["backend"], self.declare_parameter("backend", "").value)
        if self.backend_note:
            self.get_logger().warn(self.backend_note)
        self.fe = None
        self.K = None
        self.frames = {}                   # stamp -> {rgb, depth}
        self.n_lost = self.n_ok = self.n_frames = 0
        self.path = Path()
        self.path.header.frame_id = "slam_map"
        self.last_path_t = -1.0
        self.imu_count = 0
        self.first_imu = self.first_frame = None
        self.create_subscription(CameraInfo, "/uav/camera/color/camera_info", self.on_info, 2)
        self.create_subscription(Image, "/uav/camera/color/image_raw", lambda m: self.on_img(m, "rgb"), 5)
        self.create_subscription(Image, "/uav/camera/depth/image_raw", lambda m: self.on_img(m, "depth"), 5)
        self.create_subscription(Imu, "/uav/imu", self.on_imu, 400)
        self.create_subscription(String, "/uav/telemetry", self.on_tel, 5)
        self.pub_pose = self.create_publisher(PoseStamped, "/uav/slam/pose", 10)
        self.pub_status = self.create_publisher(String, "/uav/slam/status", 10)
        self.pub_path = self.create_publisher(Path, "/uav/slam/path", 2)
        # TF: the only broadcaster in the system. URDF static transforms + slam_map -> base_link from the SLAM pose.
        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_static.sendTransform(static_transforms(self.get_clock().now().to_msg()))
        self.tf_dyn = TransformBroadcaster(self)
        self.cg_in_base = np.array(self.cfg["mass_properties"]["cg_in_base_link_m"], float)   # SLAM pose is the CG pose; URDF base_link is not the CG
        self.csv = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.csv = open(os.path.join(self.log_dir, "slam_pose.csv"), "w", buffering=1)
            self.csv.write("t,state,x,y,z,qx,qy,qz,qw,inliers,features,keyframes,latency_ms\n")
            self.fcsv = open(os.path.join(self.log_dir, "slam_frames.csv"), "w", buffering=1)      # every frame, incl. INIT / LOST (analysis only)
            self.fcsv.write("t,state,orb_state,map_id,new_map,n_map_resets,runner_crashed,latency_ms\n")
        self.get_logger().info("SLAM node up (waiting for camera_info)")

    def on_info(self, m: CameraInfo):
        if self.fe is None:
            self.K = {"fx": m.k[0], "fy": m.k[4], "cx": m.k[2], "cy": m.k[5], "width": m.width, "height": m.height}
            self.fe = SlamFrontEnd(self.cfg, make_backend(self.cfg, self.K, self.backend_kind))
            self.get_logger().info(f"backend {self.fe.backend.name}, K={self.K}")

    def on_tel(self, m: String):
        if self.fe is not None and self.fe.heading_enu is None:
            d = json.loads(m.data)
            self.fe.set_initial_heading(d["roll_pitch_yaw_ned_frd"][2])      # FC heading (MAVLink ATTITUDE yaw) at start
            self.get_logger().info(f"initial heading (NED yaw) {d['roll_pitch_yaw_ned_frd'][2]:.3f} rad")

    def on_imu(self, m: Imu):
        if self.fe is None:
            return
        g, a = m.angular_velocity, m.linear_acceleration
        # REP-103 FLU -> FRD (the front-end takes MAVLink-style FRD)
        self.fe.feed_imu(stamp_to_s(m.header.stamp), (g.x, -g.y, -g.z), (a.x, -a.y, -a.z))
        self.imu_count += 1

    def on_img(self, m: Image, kind):
        t = stamp_to_s(m.header.stamp)
        f = self.frames.setdefault(round(t, 4), {})
        f[kind] = m
        if "rgb" in f and "depth" in f and self.fe is not None:
            self.frames.pop(round(t, 4))
            for k in [k for k in self.frames if k < t - 2.0]:
                self.frames.pop(k)
            self.process(t, f["rgb"], f["depth"])

    def process(self, t, rgb_msg, depth_msg):
        rgb = img_to_np(rgb_msg)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = img_to_np(depth_msg)
        t0 = time.perf_counter()
        pose = self.fe.process(t, bgr, depth)
        ms = (time.perf_counter() - t0) * 1e3
        self.n_frames += 1
        if pose.T_map_body is not None:
            self.n_ok += 1
            self.pub_pose.publish(matrix_to_pose(pose.T_map_body, t, "slam_map"))
            T_mb = pose.T_map_body.copy()
            T_mb[:3, 3] = pose.T_map_body[:3, :3] @ (-self.cg_in_base) + pose.T_map_body[:3, 3]      # base_link origin in slam_map
            self.tf_dyn.sendTransform(make_tf("slam_map", "base_link", T_mb[:3, :3], T_mb[:3, 3], s_to_stamp(t)))
            if t - self.last_path_t > 0.5:
                self.last_path_t = t
                self.path.poses.append(matrix_to_pose(pose.T_map_body, t, "slam_map"))
                self.path.header.stamp = s_to_stamp(t)
                self.pub_path.publish(self.path)
            if self.csv:
                T = pose.T_map_body
                from common.frames import quat_from_R
                q = quat_from_R(T[:3, :3])
                self.csv.write(f"{t:.4f},{pose.state},{T[0,3]:.4f},{T[1,3]:.4f},{T[2,3]:.4f},{q[1]:.5f},{q[2]:.5f},{q[3]:.5f},{q[0]:.5f},"
                               f"{pose.info.get('inliers',0)},{pose.info.get('features',0)},{pose.info.get('keyframes',0)},{ms:.1f}\n")
        else:
            self.n_lost += 1
        if self.csv:
            i = pose.info
            self.fcsv.write(f"{t:.4f},{pose.state},{i.get('orb_state', '')},{i.get('map_id', '')},{int(bool(i.get('new_map', False)))},"
                            f"{i.get('n_map_resets', '')},{int(bool(i.get('runner_crashed', False)))},{ms:.1f}\n")
        st = {"t": t, "state": pose.state, "aligned": self.fe.M is not None, "tracked": self.n_ok, "lost": self.n_lost, "frames": self.n_frames,
              "latency_ms": round(ms, 1), "backend": self.fe.backend.name, "backend_note": self.backend_note, "imu_samples": self.imu_count,
              "heading_enu": self.fe.heading_enu,
              **{k: v for k, v in pose.info.items() if isinstance(v, (int, float, str, bool))}}
        self.pub_status.publish(String(data=json.dumps(st)))


def main(args=None):
    rclpy.init(args=args)
    n = SlamNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
