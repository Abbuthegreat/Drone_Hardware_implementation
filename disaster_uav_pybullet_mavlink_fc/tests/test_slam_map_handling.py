"""ORB-SLAM3 multi-map handling in the adapter: after a tracking loss ORB-SLAM3 may continue in a NEW map with its own origin.
The UAV pose must stay continuous (new map anchored at the last accepted pose); a later merge back into the old map must restore the old frame."""
import math

import numpy as np
import pytest

from slam.adapter import SlamFrontEnd
from slam.backend import SlamResult


class _MapBackend:
    name = "fake_multimap"

    def __init__(self):
        self.T, self.map_id, self.lost = np.eye(4), 0, False

    def feed_imu(self, *a):
        pass

    def process(self, t, bgr, depth):
        if self.lost:
            return SlamResult(t, "LOST", None, {})
        return SlamResult(t, "OK", self.T.copy(), {"map_id": self.map_id})

    def reset(self):
        pass


def _fe(cfg):
    be = _MapBackend()
    fe = SlamFrontEnd(cfg, be)
    fe.set_initial_heading(math.radians(90.0))
    for k in range(30):
        fe.feed_imu(k * 0.01, np.zeros(3), np.array([0.0, 0.0, -9.81]))
    return fe, be


def _at_z(be, z):
    be.T = np.eye(4)
    be.T[2, 3] = z


def test_new_orbslam_map_is_anchored_at_last_accepted_pose_and_old_frame_returns_after_merge(cfg):
    fe, be = _fe(cfg)
    img = np.zeros((4, 4, 3), np.uint8)
    t = 0.0
    for z in np.linspace(0.0, 4.0, 21):                         # map 0: 4 m forward along the optical axis
        _at_z(be, z); t += 0.2
        last = fe.process(t, img, None)
    p_before = last.T_map_body[:3, 3].copy()
    be.lost = True
    for _ in range(5):
        t += 0.2
        assert fe.process(t, img, None).T_map_body is None
    be.lost, be.map_id = False, 1                               # recovery in a NEW map: backend pose restarts at the origin
    _at_z(be, 0.0); t += 0.2
    p_new = fe.process(t, img, None)
    assert p_new.info.get("new_map") and fe.n_map_resets == 1
    assert np.linalg.norm(p_new.T_map_body[:3, 3] - p_before) < 1e-6          # continuous: anchored where the UAV was
    _at_z(be, 0.5); t += 0.2                                    # keeps moving 0.5 m in the new map
    p2 = fe.process(t, img, None)
    assert np.linalg.norm(p2.T_map_body[:3, 3] - p_before) == pytest.approx(0.5, abs=0.02)
    be.map_id = 0                                               # maps merged: backend reports the old map again, in the old frame
    _at_z(be, 4.6); t += 0.2
    p3 = fe.process(t, img, None)
    assert not p3.info.get("new_map") and fe.n_map_resets == 1
    assert np.linalg.norm(p3.T_map_body[:3, 3] - p_before) == pytest.approx(0.6, abs=0.05)


def test_impossible_jump_inside_the_same_map_is_treated_as_a_reset(cfg):
    fe, be = _fe(cfg)
    img = np.zeros((4, 4, 3), np.uint8)
    for k, z in enumerate((0.0, 0.1, 0.2)):
        _at_z(be, z)
        last = fe.process(0.2 * (k + 1), img, None)
    _at_z(be, 0.0)                                              # ORB-SLAM3 reset the active map (same id) but the true motion was tiny
    be.T[2, 3] = 25.0                                           # ... and here it jumped 25 m in 0.2 s
    p = fe.process(0.8, img, None)
    assert p.info.get("new_map")
    assert np.linalg.norm(p.T_map_body[:3, 3] - last.T_map_body[:3, 3]) < 1e-6
