"""
Scene validation BEFORE autonomy: UAV loads, no initial collision, cameras point correctly, RGB / depth / thermal / IMU / GPS are sane,
survivors and debris are visible, obstacles collide, and PyBullet dynamics stay stable in the disaster world.
(Ground truth is used here only to place the camera and to check the result - this is an evaluation test.)
"""
import math

import numpy as np
import pybullet as p
import pytest

from perception.thermal_fusion import ThermalFusion
from sim.rgbd_camera import RGBDSensor
from sim.thermal_camera import ThermalSensor
from tests.helpers import Flyer


@pytest.fixture(scope="module")
def world_sim(cfg):
    from sim.sim_core import UavSim
    s = UavSim(cfg, world=True)
    yield s
    s.close()


def _place(s, xyz, yaw_ned_deg=0.0):
    s.vehicle.reset(xyz, yaw_ned_deg)


def test_uav_loads_with_2_2_kg_and_no_initial_collision_with_obstacles(world_sim, cfg):
    s = world_sim
    assert s.model.total_mass == pytest.approx(2.2, abs=1e-6)
    s.run(0.5)
    obstacle_like = set(s.world_info["obstacle_ids"]) | {b for ids in s.world_info["human_body_ids"].values() for b in ids}
    contacts = s.vehicle.contacts_with_world()
    assert all(cp[2] not in obstacle_like for cp in contacts), "vehicle spawned inside an obstacle"
    assert all(cp[5][2] < 0.08 for cp in contacts), "vehicle touches something above the ground at spawn (only the skids may touch the pad)"


def test_camera_frame_and_optical_axes(world_sim):
    s = world_sim
    _place(s, (0.0, -3.0, 2.0), 0.0)
    cam = RGBDSensor(s)
    pos, R_wo = cam._pose_world_truth()
    fwd = R_wo[:, 2]
    assert fwd[1] > 0.95 and math.degrees(math.asin(-fwd[2])) == pytest.approx(10.0, abs=0.5)     # forward = North, 10 deg down
    assert R_wo[:, 0] @ np.array([1.0, 0, 0]) > 0.99                                                # optical x = right = East
    assert R_wo[2, 1] < -0.9                                                                        # optical y = down


def test_rgb_depth_thermal_images_are_valid(world_sim, cfg):
    s = world_sim
    _place(s, (0.0, -3.0, 2.0), 0.0)
    cam, th = RGBDSensor(s), ThermalSensor(s, s.world_info["thermal_c"])
    rgb, depth = cam.capture()
    T = th.capture()
    assert rgb.shape == (480, 640, 3) and rgb.std() > 15
    assert rgb[:60].mean(axis=(0, 1))[2] > rgb[:60].mean(axis=(0, 1))[0]                             # sky region is blue-ish
    assert depth.dtype == np.float32 and 0.2 <= np.nanmin(depth) and np.nanmax(depth) < 12.0
    # the leaning wall (world y=4.25 face) is 7.1 m ahead: depth at the image centre is metres along the optical axis
    assert np.nanmedian(depth[230:250, 310:330]) == pytest.approx(7.17, abs=0.35)
    assert T.shape == (120, 160) and T.dtype == np.float32
    assert T.min() < -10 and 5 < np.median(T) < 30                                                    # cold sky, ambient scene


def test_thermal_shows_humans_and_fire_aligned_with_rgb(world_sim, cfg):
    s = world_sim
    _place(s, (4.6, 7.0, 2.0), 0.0)                                                                   # 4.5 m south of S1 (standing)
    cam, th = RGBDSensor(s), ThermalSensor(s, s.world_info["thermal_c"])
    rgb, depth = cam.capture()
    T = th.capture()
    K = cam.intrinsics()
    fusion = ThermalFusion(cfg, K, th.intrinsics())
    # S1 torso point (ground truth, only to check alignment) -> RGB optical frame -> thermal pixel and RGB pixel
    pos, R_wo = cam._pose_world_truth()
    torso_w = np.array([4.6, 11.5, 1.1])
    P_rgb = R_wo.T @ (torso_w - pos)
    u_r, v_r = K["fx"] * P_rgb[0] / P_rgb[2] + K["cx"], K["fy"] * P_rgb[1] / P_rgb[2] + K["cy"]
    assert 0 <= u_r < 640 and 0 <= v_r < 480
    chk = fusion.person_thermal_check(P_rgb, T)
    assert chk["in_view"] and chk["support"], chk                                                     # warm body where the RGB camera sees the person
    r, g, b = (rgb[int(v_r) - 6:int(v_r) + 6, int(u_r) - 4:int(u_r) + 4, i].astype(int).mean() for i in range(3))
    assert r > g > b                                                                                  # orange shirt in RGB at the same spot
    # fire
    _place(s, (-3.0, 12.0, 2.0), 0.0)
    T2 = th.capture()
    assert T2.max() > 300
    # background stays cool
    assert np.percentile(T, 50) < 25


def test_imu_and_gps_at_rest(world_sim, cfg):
    s = world_sim
    s.vehicle.reset((0.0, -3.0, 0.17), 0.0)
    s.run(1.5)
    im = s.imu_sample
    # specific force at rest in FRD points UP = -z (FRD z is down): (0, 0, -g), within bias/noise
    assert im.accel[2] == pytest.approx(-9.81, abs=0.4) and abs(im.accel[0]) < 0.4 and abs(im.accel[1]) < 0.4
    assert np.linalg.norm(im.gyro) < 0.05
    g = s.gps_sample
    assert g.fix_type == 3 and g.satellites >= 10
    from common.geo import GeoFrame
    ned = GeoFrame.from_config(cfg).geodetic_to_ned(g.lat_deg, g.lon_deg, g.alt_m)
    assert abs(ned[0] - (-3.0)) < 4.0 and abs(ned[1]) < 4.0                                            # within GPS noise of the spawn point


def test_obstacles_have_collisions_and_are_solid(world_sim):
    s = world_sim
    for bid, name in s.world_info["obstacle_ids"].items():
        assert p.getCollisionShapeData(bid, -1, physicsClientId=s.pb), name
    # push the vehicle into the leaning wall: it must register a contact with that body
    wall = [b for b, n in s.world_info["obstacle_ids"].items() if n == "leaning_wall"][0]
    s.vehicle.reset((0.8, 4.5, 1.5), 0.0)
    p.stepSimulation(physicsClientId=s.pb)
    assert any(cp[2] == wall for cp in s.vehicle.contacts_with_world())


def test_survivor_and_debris_visible_to_rgb_detector(world_sim, cfg):
    pytest.importorskip("onnxruntime")
    from slam import kit
    det_mod = kit.detector_module()
    det = det_mod.Detector(kit.kit_model_path("slam_kit_3/slam_kit/models/yolov8n.onnx"), conf=0.35, backend="onnxruntime")
    s = world_sim
    cam = RGBDSensor(s)
    hits = {}
    for name, pos, yaw in (("S1", (4.6, 7.5, 2.0), 0.0), ("S2", (-8.5, 20.5, 2.0), 0.0)):
        _place(s, pos, yaw)
        rgb, _ = cam.capture()
        import cv2
        d = det(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        hits[name] = d
    assert hits["S1"] and hits["S1"][0][4] > 0.35, hits
    assert hits["S2"] and hits["S2"][0][4] > 0.35, hits


def test_dynamics_stable_in_the_disaster_world(cfg):
    from sim.sim_core import UavSim
    s = UavSim(cfg, world=True)
    try:
        fl = Flyer(s)
        n0, e0 = fl.takeoff(2.0, settle_s=8.0)
        z = []
        for _ in range(60):
            fl.pos(n0, e0, s.fc.home_down - 2.0, 0.0)
            s.run(0.05)
            z.append(s.state.pos_enu[2])
        assert np.ptp(z) < 0.15
        r, pch, _ = s.state.euler_ned
        assert abs(r) < math.radians(1.5) and abs(pch) < math.radians(1.5)
        assert s.motor_thrusts.sum() == pytest.approx(21.582, rel=0.03)
    finally:
        s.close()
