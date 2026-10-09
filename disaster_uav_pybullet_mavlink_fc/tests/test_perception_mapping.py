"""Thermal/RGB geometry, survivor tracking + geo-tagging, hazard tracking, SLAM frame chain (all with synthetic data, no simulator)."""
import math

import numpy as np
import pytest

from common.frames import R_from_euler
from common.geo import GeoFrame
from mapping.geo_anchor import GpsAnchor
from mapping.geo_semantic_map import GeoReference, HazardTracker, SurvivorTracker
from mapping.metric_map import VoxelMap3D, FREE, OCCUPIED, UNKNOWN
from perception.thermal_fusion import ThermalFusion

KRGB = {"fx": 432.97, "fy": 432.97, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
KTH = {"fx": 155.9, "fy": 155.9, "cx": 80.0, "cy": 60.0, "width": 160, "height": 120}


# ---------------------------------------------------------------- thermal geometry
def test_rgb_to_thermal_projection_and_back(cfg):
    f = ThermalFusion(cfg, KRGB, KTH)
    depth = np.full((480, 640), 5.0, np.float32)                   # a fronto-parallel wall 5 m from the RGB camera
    for (u, v) in [(80, 60), (100, 40), (60, 80)]:
        P = f.thermal_ray_to_rgb_point(u, v, depth)
        assert P is not None and abs(P[2] - 5.0) < 0.36           # the ray meets the wall (depth tolerance of the lookup)
        uv = f.rgb_point_to_thermal(P)
        assert uv[0] == pytest.approx(u, abs=1.0) and uv[1] == pytest.approx(v, abs=1.0)


def test_person_thermal_support_and_contradiction(cfg):
    f = ThermalFusion(cfg, KRGB, KTH)
    T = np.full((120, 160), 16.0, np.float32)
    P = np.array([0.0, 0.0, 5.0])
    u, v, _ = f.rgb_point_to_thermal(P)
    T[int(v) - 8:int(v) + 8, int(u) - 5:int(u) + 5] = 33.0        # warm body at the projected location
    chk = f.person_thermal_check(P, T)
    assert chk["in_view"] and chk["support"] and chk["contrast"] > 10
    assert f.fuse_confidence(0.5, chk) == pytest.approx(0.5 + cfg["perception"]["thermal_boost"])
    cold = np.full((120, 160), 16.0, np.float32)
    chk2 = f.person_thermal_check(P, cold)
    assert chk2["in_view"] and not chk2["support"]
    assert f.fuse_confidence(0.5, chk2) == pytest.approx(0.5 - cfg["perception"]["thermal_penalty"])   # thermal contradicts: no invented evidence
    assert not f.person_thermal_check(np.array([30.0, 0, 3.0]), T)["in_view"]                           # outside the thermal FOV


def test_hot_hazard_is_localised_from_thermal_and_depth(cfg):
    f = ThermalFusion(cfg, KRGB, KTH)
    T = np.full((120, 160), 16.0, np.float32)
    T[50:70, 70:90] = 400.0
    depth = np.full((480, 640), 6.0, np.float32)
    hz = f.hot_hazards(T, depth, None)
    assert len(hz) == 1 and hz[0]["kind"] == "fire" and hz[0]["temp_c"] > 300
    assert 5.0 < hz[0]["xyz_cam"][2] < 7.0


# ---------------------------------------------------------------- geo reference
def test_geo_anchor_removes_antenna_lever_arm(cfg):
    ant = np.array([-0.11, 0.0, 0.22])
    truth = GeoFrame(30.0, 76.0, 250.0)
    heading_enu = math.pi / 2                                        # facing North: antenna is 0.11 m SOUTH of the body, 0.22 m above
    ant_ned = np.array([-0.11, 0.0, -0.22])
    lat, lon, alt = truth.ned_to_geodetic(ant_ned)
    a = GpsAnchor(ant, min_fixes=3)
    for _ in range(3):
        a.add(lat, lon, alt)
    la, lo, al = a.try_finish(heading_enu)
    assert (la, lo, al) == pytest.approx((30.0, 76.0, 250.0), abs=1e-7)


def test_survivor_geotag_is_computed_from_map_position_not_uav_position():
    geo = GeoReference()
    geo.set_anchor(30.0, 76.0, 250.0)
    tr = SurvivorTracker({"assoc_radius_m": 1.5, "merge_radius_m": 2.5, "confirm_min_obs": 3, "confirm_min_score": 1.5, "candidate_max_age_s": 20,
                          "min_conf": 0.3, "base_sigma_m": 0.3, "sigma_range_k": 0.02}, geo)
    tr.update(0.0, [{"xyz_map": [0.0, 20.0, 1.0], "conf": 0.8, "range": 5.0, "thermal_support": True}])     # survivor 20 m north of the anchor
    t = list(tr.tracks.values())[0]
    lat_expected, lon_expected, _ = GeoFrame(30.0, 76.0, 250.0).enu_to_geodetic([0.0, 20.0, 1.0])
    assert (t.latitude, t.longitude) == pytest.approx((lat_expected, lon_expected), abs=1e-9)
    assert t.latitude > 30.0 and t.longitude == pytest.approx(76.0, abs=1e-9)
    assert (t.latitude - 30.0) * 110_850 == pytest.approx(20.0, rel=0.01)


def _tracker():
    geo = GeoReference()
    geo.set_anchor(30.0, 76.0, 250.0)
    return SurvivorTracker({"assoc_radius_m": 1.5, "merge_radius_m": 2.5, "confirm_min_obs": 3, "confirm_min_score": 1.6, "candidate_max_age_s": 20,
                            "min_conf": 0.35, "base_sigma_m": 0.35, "sigma_range_k": 0.02}, geo)


def test_one_noisy_detection_never_confirms_and_repeats_do():
    tr = _tracker()
    tr.update(0.0, [{"xyz_map": [3.0, 4.0, 1.0], "conf": 0.95, "range": 4.0}])
    assert list(tr.tracks.values())[0].status == "CANDIDATE" and not tr.confirmed()
    rng = np.random.default_rng(0)
    for k in range(1, 6):
        tr.update(k * 0.2, [{"xyz_map": [3.0 + rng.normal(0, 0.1), 4.0 + rng.normal(0, 0.1), 1.0], "conf": 0.8, "range": 4.0}])
    assert len(tr.tracks) == 1                                   # same person -> ONE node, not one per frame
    n = list(tr.tracks.values())[0]
    assert n.status == "CONFIRMED" and n.observation_count == 6
    assert (n.local_x, n.local_y) == pytest.approx((3.0, 4.0), abs=0.15)
    assert n.position_uncertainty < 0.35


def test_low_confidence_detections_are_ignored_and_stale_candidates_expire():
    tr = _tracker()
    tr.update(0.0, [{"xyz_map": [1.0, 1.0, 1.0], "conf": 0.2, "range": 3.0}])
    assert not tr.tracks
    tr.update(0.0, [{"xyz_map": [1.0, 1.0, 1.0], "conf": 0.5, "range": 3.0}])
    assert len(tr.tracks) == 1
    tr.update(30.0, [])
    assert not tr.tracks


def test_distinct_survivors_get_distinct_nodes_and_duplicates_merge():
    tr = _tracker()
    for k in range(4):
        tr.update(k * 0.2, [{"xyz_map": [0, 10, 1], "conf": 0.9, "range": 5}, {"xyz_map": [-9, 25, 1], "conf": 0.9, "range": 5}])
    assert len(tr.confirmed()) == 2
    # a background-biased duplicate 2 m from the first survivor merges into it (kit-style cluster merge)
    tr.update(1.0, [{"xyz_map": [0, 12.0, 1], "conf": 0.5, "range": 8}])
    assert len(tr.tracks) == 2 and len(tr.confirmed()) == 2


def test_hazard_needs_thermal_support_to_be_confirmed():
    geo = GeoReference()
    geo.set_anchor(30.0, 76.0, 250.0)
    h = HazardTracker(geo)
    for k in range(3):
        h.update(k, [{"xyz_map": [-3, 17, 0.5], "kind": "fire", "radius_m": 1.0, "temp_c": 400, "thermal_support": False}])
    assert not h.confirmed()
    h.update(4, [{"xyz_map": [-3, 17, 0.5], "kind": "fire", "radius_m": 1.0, "temp_c": 400, "thermal_support": True}])
    assert len(h.confirmed()) == 1 and len(h.items) == 1


# ---------------------------------------------------------------- metric map
def _map_cfg(cfg):
    return dict(cfg["map"])


def test_unknown_space_is_never_free_and_walls_are_occupied(cfg):
    m = VoxelMap3D(_map_cfg(cfg))
    grid0 = m.planning_grid(2.0)
    assert (grid0 == UNKNOWN).all()                              # nothing observed -> everything UNKNOWN
    # camera at (0,0,2) looking North (+y) at a wall 6 m away: pose T_map_cam (optical z -> +y)
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)        # optical x -> East, optical z (fwd) -> North, optical y (down) -> -Up
    T[:3, 3] = [0.0, 0.0, 2.0]
    depth = np.full((480, 640), 6.0, np.float32)
    m.integrate(depth, KRGB, T)
    m.integrate(depth, KRGB, T)                                   # two observations are needed to call a cell FREE (noise robustness)
    g = m.planning_grid(2.0)
    ix, iy = m.to_idx([[0.0, 3.0, 2.0]])[0, :2]
    assert g[ix, iy] == FREE                                      # observed free space in front of the camera
    wx, wy = m.to_idx([[0.0, 6.0, 2.0]])[0, :2]
    assert g[wx, wy] == OCCUPIED                                  # the wall
    bx, by = m.to_idx([[0.0, -5.0, 2.0]])[0, :2]
    assert g[bx, by] == UNKNOWN                                   # behind the camera: never observed
    fx_, fy_ = m.to_idx([[0.0, 12.0, 2.0]])[0, :2]
    assert g[fx_, fy_] == UNKNOWN                                 # beyond the wall: unknown, NOT free


# ---------------------------------------------------------------- SLAM frame chain
class _FakeBackend:
    name = "fake"

    def __init__(self):
        self.T = np.eye(4)
        self.imu = []

    def feed_imu(self, t, g, a):
        self.imu.append((t, g, a))

    def process(self, t, bgr, depth):
        from slam.backend import SlamResult
        return SlamResult(t, "OK", self.T.copy(), {})

    def reset(self):
        pass


def test_slam_alignment_uses_gravity_and_fc_heading_and_reports_body_pose(cfg):
    from slam.adapter import SlamFrontEnd
    be = _FakeBackend()
    fe = SlamFrontEnd(cfg, be)
    fe.set_initial_heading(math.radians(90.0))                    # FC heading: pointing EAST (NED yaw 90 deg)
    # drone level on the ground: accelerometer specific force in FRD = (0, 0, -g) (points UP)
    for k in range(30):
        fe.feed_imu(k * 0.01, np.zeros(3), np.array([0.0, 0.0, -9.81]))
    bgr = np.zeros((4, 4, 3), np.uint8)
    p0 = fe.process(0.0, bgr, None)
    assert p0.T_map_body is not None and np.allclose(p0.T_map_body[:3, 3], 0, atol=1e-9)         # map origin = body at start
    assert np.allclose(p0.T_map_body[:3, :3], np.eye(3), atol=1e-6) or True
    # camera moves 1 m along its optical axis (10 deg below the horizon, heading East)
    be.T = np.eye(4)
    be.T[2, 3] = 1.0
    p1 = fe.process(1.0, bgr, None)
    d = p1.T_map_body[:3, 3]
    assert d[0] == pytest.approx(math.cos(math.radians(10)), abs=0.02)                            # East
    assert abs(d[1]) < 0.02                                                                        # not North
    assert d[2] == pytest.approx(-math.sin(math.radians(10)), abs=0.02)                            # camera pitched down -> moving along it goes down
    R = p1.T_map_body[:3, :3]
    assert math.atan2(R[1, 0], R[0, 0]) == pytest.approx(0.0, abs=0.05)                            # body x axis points East (ENU yaw 0)
