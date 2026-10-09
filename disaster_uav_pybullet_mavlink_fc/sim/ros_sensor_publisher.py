"""
ROS 2 publisher for the simulated camera sensors (runs INSIDE the simulation process, because the images can
only come from PyBullet; on the real vehicle the RealSense / thermal-camera ROS drivers take this role).

All headers are stamped with SIMULATION time (the same time base as the FC's MAVLink timestamps and the bridge's
IMU/GPS/odometry), so SLAM can pair images with IMU exactly.

Topics (frame_id in brackets):
    /uav/camera/color/image_raw          sensor_msgs/Image  rgb8                  [rgbd_camera_optical_link]
    /uav/camera/depth/image_raw          sensor_msgs/Image  32FC1  metres, NaN=invalid, registered to colour
    /uav/camera/color/camera_info        sensor_msgs/CameraInfo
    /uav/camera/depth/camera_info        sensor_msgs/CameraInfo
    /uav/camera/thermal/image_raw        sensor_msgs/Image  32FC1  degrees Celsius (SYNTHETIC thermal)  [thermal_camera_optical_link]
    /uav/camera/thermal/camera_info      sensor_msgs/CameraInfo
    /demo/main_view/image_raw            sensor_msgs/Image  bgr8   third-person chase view for the demo VIDEO ONLY
                                         (never consumed by autonomy)
RGB, depth and thermal of one frame are captured in the same simulation step (same stamp, same body pose).
"""
from __future__ import annotations

import array
import math
import time

import numpy as np
import pybullet as p
import rclpy
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time

from sim.rgbd_camera import RGBDSensor
from sim.thermal_camera import ThermalSensor


def _stamp(t):
    m = Time()
    m.sec, m.nanosec = int(t), int((t % 1.0) * 1e9)
    return m


def _image(arr, encoding, stamp, frame_id):
    m = Image()
    m.header.stamp, m.header.frame_id = stamp, frame_id
    m.height, m.width, m.encoding = arr.shape[0], arr.shape[1], encoding
    m.step = int(arr.strides[0])
    ar = array.array("B")                    # fast path: assigning bytes makes rclpy validate every element (~100 ms per image)
    ar.frombytes(np.ascontiguousarray(arr).tobytes())
    m.data = ar
    return m


def _info(k, stamp, frame_id):
    m = CameraInfo()
    m.header.stamp, m.header.frame_id = stamp, frame_id
    m.width, m.height = k["width"], k["height"]
    m.distortion_model = "plumb_bob"
    m.d = [0.0] * 5
    m.k = [k["fx"], 0.0, k["cx"], 0.0, k["fy"], k["cy"], 0.0, 0.0, 1.0]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.p = [k["fx"], 0.0, k["cx"], 0.0, 0.0, k["fy"], k["cy"], 0.0, 0.0, 0.0, 1.0, 0.0]
    return m


class RosSensorPublisher:
    def __init__(self, sim, with_main_view=True):
        if not rclpy.ok():
            rclpy.init()
        self.sim = sim
        self.cam = RGBDSensor(sim)
        table = (sim.world_info or {}).get("thermal_c", {})
        self.thermal = ThermalSensor(sim, table)
        self.node = rclpy.create_node("uav_sim_sensors")
        c = self.node.create_publisher
        self.p_rgb = c(Image, "/uav/camera/color/image_raw", 5)
        self.p_depth = c(Image, "/uav/camera/depth/image_raw", 5)
        self.p_th = c(Image, "/uav/camera/thermal/image_raw", 5)
        self.p_irgb = c(CameraInfo, "/uav/camera/color/camera_info", 5)
        self.p_idepth = c(CameraInfo, "/uav/camera/depth/camera_info", 5)
        self.p_ith = c(CameraInfo, "/uav/camera/thermal/camera_info", 5)
        self.div = max(1, int(round(sim.hz / self.cam.rate_hz)))
        self.frames = 0
        self.main = None
        vcfg = sim.cfg.get("video", {}).get("main_view")
        if with_main_view and vcfg:
            self.main = vcfg
            self.p_main = c(Image, "/demo/main_view/image_raw", 5)
            fps = sim.cfg["video"]["fps"]
            self.main_div = max(1, int(round(sim.hz / fps)))
            self.main_proj = p.computeProjectionMatrixFOV(vcfg["fov_deg"], vcfg["width"] / vcfg["height"], 0.3, 120.0)
            self._eye = None
        self.n_main = 0
        self.prof = {"rgbd": 0.0, "thermal": 0.0, "publish": 0.0, "main": 0.0, "n_cam": 0, "n_main": 0}
        self._last_prof = time.perf_counter()

    # ------------------------------------------------------------------ chase camera (video only)
    def _main_view(self):
        v = self.main
        st = self.sim.vehicle.read_state()
        yaw = math.atan2(st.R_enu_flu[1, 0], st.R_enu_flu[0, 0])           # heading in ENU
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        target = st.pos_enu + fwd * v["look_ahead_m"]
        eye = st.pos_enu - fwd * v["distance_m"] + np.array([0, 0, v["height_m"]])
        self._eye = eye if self._eye is None else 0.85 * self._eye + 0.15 * eye     # smooth camera motion
        view = p.computeViewMatrix(self._eye.tolist(), target.tolist(), [0, 0, 1], physicsClientId=self.sim.pb)
        img = p.getCameraImage(v["width"], v["height"], view, self.main_proj, renderer=p.ER_TINY_RENDERER,
                               flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=self.sim.pb)
        rgb = np.asarray(img[2], np.uint8).reshape(v["height"], v["width"], 4)[:, :, :3]
        depth = np.asarray(img[3], np.float32).reshape(v["height"], v["width"])
        out = rgb[:, :, ::-1].copy()
        sky = depth >= 0.9999
        if sky.any():
            g = np.linspace(0.0, 1.0, v["height"], dtype=np.float32)[:, None]
            grad = np.stack([225 + 25 * g, 185 + 45 * g, 150 + 70 * g], axis=2).repeat(v["width"], axis=1)   # BGR sky
            out[sky] = grad[sky].astype(np.uint8)
        return out

    def on_step(self, sim):
        t = sim.t
        if sim.step_count % self.div == 0:
            t0 = time.perf_counter()
            rgb, depth = self.cam.capture()
            t1 = time.perf_counter()
            T = self.thermal.capture()
            t2 = time.perf_counter()
            st = _stamp(t)
            self.p_rgb.publish(_image(rgb, "rgb8", st, "rgbd_camera_optical_link"))
            self.p_depth.publish(_image(depth, "32FC1", st, "rgbd_camera_optical_link"))
            self.p_th.publish(_image(T, "32FC1", st, "thermal_camera_optical_link"))
            k = self.cam.intrinsics()
            self.p_irgb.publish(_info(k, st, "rgbd_camera_optical_link"))
            self.p_idepth.publish(_info(k, st, "rgbd_camera_optical_link"))
            self.p_ith.publish(_info(self.thermal.intrinsics(), st, "thermal_camera_optical_link"))
            self.frames += 1
            t3 = time.perf_counter()
            self.prof["rgbd"] += t1 - t0
            self.prof["thermal"] += t2 - t1
            self.prof["publish"] += t3 - t2
            self.prof["n_cam"] += 1
        if self.main is not None and sim.step_count % self.main_div == 0:
            t0 = time.perf_counter()
            self.p_main.publish(_image(self._main_view(), "bgr8", _stamp(t), "chase_view"))
            self.prof["main"] += time.perf_counter() - t0
            self.prof["n_main"] += 1
            self.n_main += 1
        rclpy.spin_once(self.node, timeout_sec=0.0)
        now = time.perf_counter()
        if now - self._last_prof > 10.0:
            P = self.prof
            print(f"[sensors] sim t={t:.1f}s  per RGB-D frame: render {1e3*P['rgbd']/max(P['n_cam'],1):.0f} ms, thermal {1e3*P['thermal']/max(P['n_cam'],1):.0f} ms, "
                  f"publish {1e3*P['publish']/max(P['n_cam'],1):.0f} ms | chase view {1e3*P['main']/max(P['n_main'],1):.0f} ms", flush=True)
            self._last_prof = now
