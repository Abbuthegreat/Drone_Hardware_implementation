"""
ROS 2 sensor interface of the simulation (needs rclpy): topic names, encodings, frames, camera_info, synchronised stamps.
Starts the simulator with --ros-sensors as a separate process and listens with a plain rclpy node.
"""
import importlib.util
import os
import time

import numpy as np
import pytest

from tests.fc_process import FcProcess

pytestmark = [pytest.mark.ros, pytest.mark.skipif(importlib.util.find_spec("rclpy") is None, reason="rclpy not available")]
os.environ.setdefault("ROS_DOMAIN_ID", "66")


def test_camera_topics_encodings_frames_and_sync(tmp_path):
    import rclpy
    from sensor_msgs.msg import CameraInfo, Image
    fc = FcProcess(14741, 14761, log_dir=str(tmp_path), world=True, ros_sensors=True)
    try:
        fc.wait_ready(120)
        rclpy.init()
        node = rclpy.create_node("sensor_test")
        got = {}
        topics = {"/uav/camera/color/image_raw": Image, "/uav/camera/depth/image_raw": Image, "/uav/camera/thermal/image_raw": Image,
                  "/uav/camera/color/camera_info": CameraInfo, "/uav/camera/depth/camera_info": CameraInfo,
                  "/uav/camera/thermal/camera_info": CameraInfo, "/demo/main_view/image_raw": Image}
        for t, ty in topics.items():
            node.create_subscription(ty, t, lambda m, t=t: got.setdefault(t, []).append(m), 10)
        t_end = time.time() + 60
        while time.time() < t_end and not all(len(got.get(t, [])) >= 3 for t in topics):
            rclpy.spin_once(node, timeout_sec=0.1)
        missing = [t for t in topics if len(got.get(t, [])) < 3]
        assert not missing, missing
        c, d, th = got["/uav/camera/color/image_raw"][1], got["/uav/camera/depth/image_raw"][1], got["/uav/camera/thermal/image_raw"][1]
        assert (c.encoding, c.width, c.height) == ("rgb8", 640, 480) and c.header.frame_id == "rgbd_camera_optical_link"
        assert (d.encoding, d.width, d.height) == ("32FC1", 640, 480) and d.header.frame_id == "rgbd_camera_optical_link"
        assert (th.encoding, th.width, th.height) == ("32FC1", 160, 120) and th.header.frame_id == "thermal_camera_optical_link"
        depth = np.frombuffer(d.data, np.float32).reshape(480, 640)
        assert np.isfinite(depth).any() and np.nanmin(depth) >= 0.2 and np.nanmax(depth) <= 12.0         # metres
        assert (c.header.stamp.sec, c.header.stamp.nanosec) in {(x.header.stamp.sec, x.header.stamp.nanosec) for x in got["/uav/camera/depth/image_raw"]}
        stamps = [m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in got["/uav/camera/color/image_raw"]]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        ci = got["/uav/camera/color/camera_info"][0]
        assert ci.k[0] == pytest.approx(432.97, abs=0.1) and ci.k[2] == 320.0 and ci.k[5] == 240.0
        cth = got["/uav/camera/thermal/camera_info"][0]
        assert cth.width == 160 and cth.k[2] == 80.0
        node.destroy_node()
        rclpy.shutdown()
    finally:
        fc.stop()
