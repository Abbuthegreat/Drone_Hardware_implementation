"""
Ground-truth firewall: the autonomy stack must not be able to reach simulator internals or the evaluation channel.
Enforced statically (imports and topic names) for every autonomy package.
"""
import ast
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTONOMY_DIRS = ["algo", "companion", "ros2_bridge", "slam", "perception", "mapping", "planning", "mission", "video"]
FORBIDDEN_PREFIXES = ("pybullet", "pybullet_data", "sim", "worlds", "eval", "tests")
NODES = ["ros2_bridge/slam_node.py", "ros2_bridge/perception_mapping_node.py", "ros2_bridge/nav_node.py", "video/compositor_node.py"]


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.append(n.module)
    return out


def _autonomy_files():
    for d in AUTONOMY_DIRS:
        for dp, _, fs in os.walk(os.path.join(ROOT, d)):
            for f in fs:
                if f.endswith(".py"):
                    yield os.path.join(dp, f)


def test_autonomy_code_never_imports_simulator_or_ground_truth():
    bad = []
    for path in _autonomy_files():
        for mod in _imports(path):
            if mod.split(".")[0] in FORBIDDEN_PREFIXES:
                bad.append((os.path.relpath(path, ROOT), mod))
    assert not bad, f"autonomy code imports simulator/ground-truth modules: {bad}"


def test_autonomy_code_does_not_mention_ground_truth_channel():
    for path in _autonomy_files():
        text = open(path, encoding="utf-8").read()
        assert "eval.ground_truth" not in text and "world_truth" not in text and "ground_truth.csv" not in text, path


def test_ground_truth_module_documents_the_firewall():
    text = open(os.path.join(ROOT, "eval", "ground_truth.py"), encoding="utf-8").read()
    assert "GROUND-TRUTH FIREWALL" in text


def test_survivor_positions_not_in_any_autonomy_facing_channel():
    """The MAVLink server / bridge must not carry survivor coordinates; the sensor publisher must not publish survivor topics."""
    for f in ("sim/mavlink_fc_server.py", "ros2_bridge/uav_mavlink_bridge.py"):
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert "humans" not in text and "world_info" not in text, f
    pub = open(os.path.join(ROOT, "sim", "ros_sensor_publisher.py"), encoding="utf-8").read()
    topics = re.findall(r'"(/[A-Za-z0-9_/]+)"', pub)
    assert topics and not any("human" in t or "survivor" in t or "truth" in t for t in topics), topics


def test_autonomy_nodes_only_use_legitimate_input_topics():
    """SLAM, perception/mapping, navigation and the video compositor subscribe to sensor / SLAM / map / MAVLink-telemetry topics only."""
    allowed_prefixes = ("/uav/camera/", "/uav/imu", "/uav/gps/", "/uav/telemetry", "/uav/armed", "/uav/mode", "/uav/slam/", "/uav/map/",
                        "/uav/nav/", "/uav/perception/", "/uav/cmd_vel", "/uav/arm", "/uav/offboard", "/uav/land", "/demo/")
    for f in NODES:
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        for topic in set(re.findall(r'"(/(?:uav|demo|eval)[A-Za-z0-9_/]*)"', text)):
            assert topic.startswith(allowed_prefixes), (f, topic)
        assert "/uav/odom" not in text or "publish" in text, f
    nav = open(os.path.join(ROOT, "ros2_bridge", "nav_node.py"), encoding="utf-8").read()
    assert "/uav/odom" not in nav.split("Subscribes")[-1].split("Publishes")[0] and "Odometry" not in nav


def test_slam_and_map_do_not_read_flight_controller_pose():
    """The SLAM / mapping / nav nodes must not subscribe to the FC pose estimate (/uav/odom): pose = SLAM only."""
    for f in ("ros2_bridge/slam_node.py", "ros2_bridge/perception_mapping_node.py", "ros2_bridge/nav_node.py"):
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        assert "create_subscription(Odometry" not in text and "/uav/odom" not in text, f


def test_autonomy_interface_carries_no_truth_fields():
    from companion.mavlink_client import Telemetry
    fields = set(Telemetry.__dataclass_fields__)
    assert not any("truth" in f or "ground" in f for f in fields)
    from mapping.geo_semantic_map import SurvivorNode
    assert not any("truth" in f or "gt_" in f or "body_id" in f for f in SurvivorNode.__dataclass_fields__)
