"""ORB-SLAM3 integration tests (kit ORB-SLAM3 via slam/rgbd_stream). Skipped when the binary has not been built (bash slam/build_orbslam3.sh)."""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORB_DIR = os.environ.get("ORB_SLAM3_DIR", os.path.expanduser("~/slam_deps/ORB_SLAM3"))
HAVE_ORB = os.path.exists(os.path.join(ORB_DIR, "Examples", "RGB-D", "rgbd_stream")) and os.path.exists(os.path.join(ORB_DIR, "Vocabulary", "ORBvoc.txt"))


def test_simulation_yaml_matches_simulated_camera_and_depth_scale():
    from sim.config import load_config
    from sim.rgbd_camera import RGBDSensor
    from sim.sim_core import UavSim
    cfg = load_config()
    s = UavSim(cfg, world=False)
    k = RGBDSensor(s).intrinsics()
    s.close()
    txt = open(os.path.join(ROOT, "config", "orbslam3", "pybullet_rgbd.yaml")).read()
    get = lambda key: float(re.search(rf"^{re.escape(key)}:\s*([-0-9.eE]+)", txt, re.M).group(1))
    for a, b in (("fx", "Camera1.fx"), ("fy", "Camera1.fy"), ("cx", "Camera1.cx"), ("cy", "Camera1.cy")):
        assert abs(k[a] - get(b)) < 1e-4
    assert get("Camera.width") == k["width"] and get("Camera.height") == k["height"]
    assert get("RGBD.DepthMapFactor") == 1000.0            # sim metres -> uint16 mm -> /1000


def test_depth_metres_to_orbslam_millimetres_roundtrip():
    d = np.array([0.2, 1.234, 7.5, 12.0, np.nan], np.float32)
    d16 = np.nan_to_num(d * 1000.0, nan=0.0).clip(0, 65535).astype(np.uint16)       # same line as slam/orbslam3_backend.py
    assert np.allclose(d16 / 1000.0, np.nan_to_num(d, nan=0.0), atol=1.5e-3)


def test_tf_static_transforms_match_urdf_and_optical_convention():
    from ros2_bridge.tf_util import urdf_fixed_joints
    j = {c: (p, R, t) for p, c, R, t in urdf_fixed_joints()}
    assert j["rgbd_camera_link"][0] == "base_link" and j["rgbd_camera_optical_link"][0] == "rgbd_camera_link"
    R1, t1 = j["rgbd_camera_link"][1], j["rgbd_camera_link"][2]
    R2 = j["rgbd_camera_optical_link"][1]
    R_bo = R1 @ R2                                           # optical -> base_link
    # optical z (forward) must point along camera x (forward), pitched 10 deg down; optical x (right) -> body -y; optical y (down) -> body -z-ish
    assert np.allclose(R_bo @ [0, 0, 1], [np.cos(np.radians(10)), 0, -np.sin(np.radians(10))], atol=1e-5)
    assert np.allclose(R_bo @ [1, 0, 0], [0, -1, 0], atol=1e-5)
    assert (R_bo @ [0, 1, 0])[2] < -0.9
    from common.config import load_config
    cfg = load_config()
    R_cfg = np.array(cfg["sensor_extrinsics"]["rgbd_optical"]["R_body_from_optical"])
    assert np.allclose(R_bo, R_cfg, atol=1e-5)               # URDF and the config used by the SLAM adapter agree
    cg = np.array(cfg["mass_properties"]["cg_in_base_link_m"])
    assert np.allclose(t1 - cg + 0, np.array(cfg["sensor_extrinsics"]["rgbd_optical"]["xyz_body_flu_m"]), atol=1e-5)


def test_slam_node_does_not_subscribe_to_ground_truth_or_fc_pose():
    src = open(os.path.join(ROOT, "ros2_bridge", "slam_node.py")).read()
    assert "/uav/odom" not in src and "ground_truth" not in src and "pybullet" not in src


@pytest.mark.skipif(not HAVE_ORB, reason="ORB-SLAM3 not built: bash slam/build_orbslam3.sh")
@pytest.mark.slow
def test_orbslam3_tracks_simulated_rgbd_flight_and_is_not_ground_truth(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "test_slam_pipeline.py"), "--out", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
    rep = json.load(open(tmp_path / "report.json"))
    assert rep["backend"] == "orbslam3"
    assert rep["tracked_fraction"] >= 0.95 and rep["timestamps_monotonic"]
    assert rep["gt_path_length_m"] > 8.0
    assert 0.0 < rep["ate_rmse_m"] < 0.5 and rep["drift_percent_of_path"] < 5.0           # real estimate: small but non-zero error
    assert r.returncode == 0, r.stdout[-500:]
