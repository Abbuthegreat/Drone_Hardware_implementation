"""
Test M - ROS 2 communication (ROS command changes vehicle motion; telemetry returns to ROS).

* The frame-conversion rules of /uav/cmd_vel are unit-tested without ROS.
* The end-to-end test runs scripts/comm_test.py (ROS 2 node + bridge + FC + PyBullet, real `ros2` CLI)
  and requires an environment with rclpy; it is SKIPPED (not passed) when rclpy is missing.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys

import pytest

from common.frames import ros_twist_to_ned

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_RCLPY = importlib.util.find_spec("rclpy") is not None


def test_cmd_vel_enu_semantics():
    # East, North, Up, CCW yaw  ->  NED: vn = North, ve = East, vd = -Up, yaw_rate(cw) = -CCW
    assert ros_twist_to_ned("enu", (1.0, 0.0, 0.0), 0.0, 0.0) == (0.0, 1.0, -0.0, -0.0)        # +x = East
    assert ros_twist_to_ned("enu", (0.0, 2.0, 0.0), 0.0, 0.0)[:2] == (2.0, 0.0)              # +y = North
    assert ros_twist_to_ned("enu", (0.0, 0.0, 1.5), 0.0, 0.0)[2] == -1.5                     # +z = Up = negative Down
    assert ros_twist_to_ned("enu", (0, 0, 0), 0.7, 0.0)[3] == -0.7                             # CCW positive in ROS
    assert ros_twist_to_ned("ned", (1, 2, 3), 0.4, 0.0) == (1, 2, 3, 0.4)


def test_cmd_vel_body_frame_follows_heading():
    vn, ve, vd, yr = ros_twist_to_ned("body_flu", (1.0, 0.0, 0.0), 0.0, 0.0)             # forward, heading North
    assert (vn, ve) == pytest.approx((1.0, 0.0))
    vn, ve, *_ = ros_twist_to_ned("body_flu", (1.0, 0.0, 0.0), 0.0, math.pi / 2)         # forward, heading East
    assert (vn, ve) == pytest.approx((0.0, 1.0), abs=1e-9)
    vn, ve, *_ = ros_twist_to_ned("body_flu", (0.0, 1.0, 0.0), 0.0, 0.0)                 # LEFT while heading North = West
    assert (vn, ve) == pytest.approx((0.0, -1.0), abs=1e-9)
    with pytest.raises(ValueError):
        ros_twist_to_ned("bogus", (0, 0, 0), 0, 0)


@pytest.mark.ros
@pytest.mark.skipif(not HAS_RCLPY, reason="rclpy not available: install ROS 2 (e.g. RoboStack humble) and run in that env")
def test_ros2_end_to_end_comm_test(tmp_path):
    out = tmp_path / "comm"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "comm_test.py"), "--no-world", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=400)
    report = json.load(open(out / "report.json"))
    failed = [s for s in report["steps"] if s["result"] != "PASS"]
    assert not failed, failed
    assert r.returncode == 0 and len(report["steps"]) == 16
