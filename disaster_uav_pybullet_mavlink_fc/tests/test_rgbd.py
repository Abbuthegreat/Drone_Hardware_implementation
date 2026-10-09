"""Test N - RGB-D camera: mounted on the physical body, metric depth, consistent intrinsics."""
import math

import numpy as np
import pybullet as p
import pytest

from sim.rgbd_camera import RGBDSensor
from tests.helpers import Flyer


@pytest.fixture
def cam_sim(make_sim):
    s = make_sim(pos_enu=(0, 0, 3.0))
    return s, RGBDSensor(s)


def _ground_depth_expected(s, cam, u, v):
    """Analytic z-depth of the ground plane z=0 along pixel (u,v), from the (truth) camera pose."""
    pos, R_wo = cam._pose_world_truth()
    ray_opt = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    ray_w = R_wo @ ray_opt
    t = -pos[2] / ray_w[2]                    # ray param where z_world = 0 (ray_w z-component per unit optical z)
    return t                                   # optical-z depth, because ray_opt has z = 1


def test_rgb_and_depth_shapes_types_and_metres(cam_sim):
    s, cam = cam_sim
    rgb, depth = cam.capture()
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    assert depth.shape == (480, 640) and depth.dtype == np.float32
    valid = depth[np.isfinite(depth)]
    assert valid.size > 1000
    assert cam.near <= valid.min() and valid.max() < cam.far


def test_intrinsics_are_explicit_and_consistent(cam_sim):
    _, cam = cam_sim
    k = cam.intrinsics()
    assert k["fx"] == k["fy"] and k["cx"] == 320 and k["cy"] == 240
    assert 2 * math.degrees(math.atan(240 / k["fy"])) == pytest.approx(k["vfov_deg"])
    assert 2 * math.degrees(math.atan(320 / k["fx"])) == pytest.approx(k["hfov_deg"])
    assert (k["near_m"], k["far_m"]) == (0.2, 12.0)


def test_depth_values_match_analytic_ground_geometry(make_sim):
    """Flat ground: depth at several pixels must equal the ray/plane intersection distance (metres)."""
    s = make_sim(pos_enu=(0, 0, 3.0))
    cam = RGBDSensor(s)
    cam.noise = 0.0
    _, depth = cam.capture()
    checked = 0
    for (u, v) in [(320, 400), (100, 440), (540, 300), (320, 460)]:
        exp = _ground_depth_expected(s, cam, u + 0.5, v + 0.5)
        if 0.3 < exp < cam.far - 0.5 and np.isfinite(depth[v, u]):
            assert depth[v, u] == pytest.approx(exp, rel=0.02), (u, v)
            checked += 1
    assert checked >= 3


def test_camera_is_rigidly_attached_and_moves_with_the_uav(make_sim):
    s = make_sim()
    cam = RGBDSensor(s)
    cam.noise = 0.0
    fl = Flyer(s)
    pos0, R0 = cam._pose_world_truth()
    n0, e0 = fl.takeoff(2.0, settle_s=7.0)
    _, d_low = cam.capture()
    pos1, _ = cam._pose_world_truth()
    st = s.vehicle.read_state()
    # camera position == body pose composed with the URDF extrinsics (no independent camera state)
    expect = st.pos_enu + st.R_enu_flu @ s.model.cam_optical_pos_flu
    assert np.allclose(pos1, expect, atol=1e-6)
    assert pos1[2] - pos0[2] > 1.5                        # it climbed with the vehicle
    fl.stream(6.0, lambda: fl.pos(n0, e0, s.fc.home_down - 5.0, 0.0))   # higher
    _, d_high = cam.capture()
    u, v = 320, 450
    assert d_high[v, u] > d_low[v, u] * 1.4               # same pixel now sees farther ground
    # yaw: rotating the vehicle rotates the camera forward axis
    _, R_a = cam._pose_world_truth()
    fl.stream(4.0, lambda: fl.pos(n0, e0, s.fc.home_down - 5.0, math.radians(90)))
    _, R_b = cam._pose_world_truth()
    fwd_a, fwd_b = R_a[:, 2], R_b[:, 2]
    ang = math.degrees(math.acos(np.clip(fwd_a[:2] @ fwd_b[:2] / (np.linalg.norm(fwd_a[:2]) * np.linalg.norm(fwd_b[:2])), -1, 1)))
    assert ang == pytest.approx(90, abs=5)


def test_camera_pitch_extrinsic_is_ten_degrees_down(make_sim):
    s = make_sim(pos_enu=(0, 0, 3.0), yaw_ned_deg=0.0)
    cam = RGBDSensor(s)
    _, R_wo = cam._pose_world_truth()
    fwd = R_wo[:, 2]
    assert math.degrees(math.asin(-fwd[2])) == pytest.approx(10.0, abs=0.5)        # looks down 10 deg
    assert fwd[1] > 0.9                                                             # towards North (+Y)


def test_survivors_are_visible_in_rgb(make_sim, cfg):
    """A survivor is a real PyBullet object: a camera 4.5 m south of S1 (standing, orange shirt) must see the shirt colour."""
    s = make_sim(world=True, pos_enu=(4.6, 7.0, 2.0), yaw_ned_deg=0.0)
    cam = RGBDSensor(s)
    rgb, depth = cam.capture()
    r, g, b = (rgb[:, :, i].astype(int) for i in range(3))
    orange = (r > 140) & (r > 1.6 * g) & (g > 1.5 * b) & (b < 90)          # orange shirt (0.85, 0.30, 0.10) under shading
    assert orange.sum() > 30, "survivor S1 not visible to the RGB-D camera"
