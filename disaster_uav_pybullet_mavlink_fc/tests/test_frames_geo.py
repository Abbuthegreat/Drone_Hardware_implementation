import math

import numpy as np
import pytest

from common import frames as F
from common.geo import GeoFrame, geodetic_to_ecef, ecef_to_geodetic


def test_ned_enu_involution_and_axes():
    v = np.array([1.0, 2.0, 3.0])            # N, E, D
    enu = F.ned_to_enu(v)
    assert np.allclose(enu, [2.0, 1.0, -3.0])  # x=East, y=North, z=Up
    assert np.allclose(F.enu_to_ned(enu), v)


def test_body_frames_and_attitude_conversion():
    # level vehicle heading North: FLU x -> ENU +y ; FRD x -> NED +x
    yaw_enu = math.pi / 2
    R_enu_flu = F.R_from_euler(0, 0, yaw_enu)
    R_ned_frd = F.R_ned_frd_from_enu_flu(R_enu_flu)
    r, p, y = F.euler_from_R(R_ned_frd)
    assert (r, p, y) == pytest.approx((0, 0, 0), abs=1e-9)          # heading North -> yaw_ned = 0
    # heading East in ENU (yaw_enu = 0) -> yaw_ned = +90 deg (clockwise from North)
    _, _, y = F.euler_from_R(F.R_ned_frd_from_enu_flu(np.eye(3)))
    assert y == pytest.approx(math.pi / 2)
    # nose-up: rotation about FLU +y is nose DOWN, so nose-up is negative -> positive pitch in FRD
    _, p, _ = F.euler_from_R(F.R_ned_frd_from_enu_flu(F.R_from_euler(0, -0.2, yaw_enu)))
    assert p == pytest.approx(0.2)
    # right-wing-down: FLU roll about +x positive => FRD roll positive (same axis)
    r, _, _ = F.euler_from_R(F.R_ned_frd_from_enu_flu(F.R_from_euler(0.2, 0, yaw_enu)))
    assert r == pytest.approx(0.2)


def test_yaw_rate_sign_ccw_enu_is_negative_mavlink():
    """ENU CCW rotation about +z(up) == negative rotation about NED +z(down)."""
    w_enu = np.array([0, 0, 1.0])
    assert F.enu_to_ned(w_enu)[2] == pytest.approx(-1.0)


def test_euler_quat_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r, p, y = rng.uniform(-1, 1), rng.uniform(-1.2, 1.2), rng.uniform(-3, 3)
        R = F.R_from_euler(r, p, y)
        assert np.allclose(F.euler_from_R(R), (r, p, y), atol=1e-9)
        assert np.allclose(F.R_from_quat(F.quat_from_R(R)), R, atol=1e-9)


def test_optical_frame_axes():
    # optical z (forward) == link x ; optical x (right) == link -y ; optical y (down) == link -z
    R = F.R_LINK_OPTICAL
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(R @ [0, 1, 0], [0, 0, -1])
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_geo_roundtrip_and_scale():
    g = GeoFrame(30.0, 76.0, 250.0)
    lat, lon, alt = g.ned_to_geodetic([0, 0, 0])
    assert (lat, lon, alt) == pytest.approx((30.0, 76.0, 250.0), abs=1e-9)
    lat, lon, alt = g.ned_to_geodetic([111.0, 0, 0])            # 111 m north
    assert lat - 30.0 == pytest.approx(111.0 / 110_850.0, rel=2e-3)   # ~ 1 deg = 110.85 km at 30 N
    assert alt == pytest.approx(250.0, abs=0.01)
    lat, lon, alt = g.ned_to_geodetic([0, 100.0, -5.0])          # 100 m east, 5 m up
    assert lon > 76.0 and alt == pytest.approx(255.0, abs=0.01)
    for ned in ([10, -20, -3], [-500, 800, 40]):
        back = g.geodetic_to_ned(*g.ned_to_geodetic(ned))
        assert np.allclose(back, ned, atol=1e-6)
    x = geodetic_to_ecef(12.3, 45.6, 789.0)
    assert ecef_to_geodetic(x) == pytest.approx((12.3, 45.6, 789.0), abs=1e-7)
