"""Sensor-frame front-clearance guard: depth-image corridor clearance (no SLAM pose / map) + the mission reaction. Synthetic inputs only."""
import math

import numpy as np

from perception.front_clearance import front_clearance
from planning.astar import FREE
from tests.test_mission_safety import C, Pilot, _grid  # noqa: F401  (fixtures / helpers)

K = {"fx": 432.97, "fy": 432.97, "cx": 320.0, "cy": 240.0}


def _depth(z=0.0, box=None):
    d = np.full((480, 640), 4.0, np.float32)
    if box:
        u0, u1, v0, v1, zb = box
        d[v0:v1, u0:u1] = zb
    return d


def test_front_clearance_sees_object_in_corridor_and_ignores_lateral_ones():
    ahead = front_clearance(_depth(box=(280, 360, 200, 280, 1.2)), K)
    assert ahead is not None and abs(ahead - 1.2) < 1e-3
    side = front_clearance(_depth(box=(0, 10, 200, 280, 0.9)), K)          # far left of the corridor: X = (10-320)*0.9/433 = -0.64 m
    assert side is not None and side > 3.9
    assert front_clearance(np.zeros((480, 640), np.float32), K) is None      # nothing valid -> no measurement (guard inactive)
    assert front_clearance(np.full((480, 640), np.nan, np.float32), K) is None


def _fly(pilot, xy, front, age=0.0, yaw=math.pi / 2):
    pilot.t += 0.1
    return pilot.mc.step(pilot.t, np.array([xy[0], xy[1], pilot.alt]), yaw, pilot.grid, 1, True, 0.0, pilot.armed, pilot.mode, [], True,
                         fc_link_ok=True, grid_age_s=0.0, front_clear_m=front, front_age_s=age)


def test_guard_stops_forward_motion_then_backs_away(C):
    p = Pilot(C, _grid(FREE))
    p.to_search()
    st = _fly(p, (0.0, 0.0), None)                                           # no measurement: normal flight north (WP1 is at +y)
    assert st.cmd.vy > 0.2
    st = _fly(p, (0.0, 0.0), 0.9)                                            # 0.75 < 0.9 < 1.0 -> hover in the forward direction
    assert abs(st.cmd.vy) < 1e-6 and p.mc.events[-1]["event"] == "FRONT_CLEARANCE_HOLD"
    st = _fly(p, (0.0, 0.0), 0.6)                                            # < 0.75 -> back away along the body axis (south, since yaw = north)
    assert st.status["state"] == "AVOID_RETREAT" and st.cmd.vy < -0.2 and abs(st.cmd.vx) < 1e-6
    assert p.mc.events[-1]["event"] == "FRONT_CLEARANCE_RETREAT"
    st = _fly(p, (0.0, 0.0), 3.0)                                            # clear again -> normal flight
    assert st.cmd.vy > 0.2


def test_guard_ignores_stale_measurement(C):
    p = Pilot(C, _grid(FREE))
    p.to_search()
    st = _fly(p, (0.0, 0.0), 0.5, age=5.0)
    assert st.cmd.vy > 0.2
