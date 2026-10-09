"""A* on a live-map style grid and VFH+ avoidance (synthetic grids; the closed-loop tests use the real map)."""
import math

import numpy as np
import pytest

from planning.astar import AStarPlanner, FREE, OCC, UNK
from planning.vfh_plus import VfhPlus

V = 0.25


def _grid(nx=100, ny=100):
    return np.full((nx, ny), FREE, np.int8)


@pytest.fixture(scope="module")
def C(cfg):
    return cfg


def test_astar_straight_path_in_free_space(C):
    g = _grid()
    p = AStarPlanner(C["astar"], V, (-12.5, -5.0))
    r = p.plan(g, (0.0, 0.0), (0.0, 15.0))
    assert r.found and not r.partial
    assert math.hypot(r.path[-1][0], r.path[-1][1] - 15.0) < 0.5
    assert len(r.path) <= 3                                # simplified to (nearly) a straight line


def test_astar_goes_around_a_wall_with_clearance(C):
    g = _grid()
    p = AStarPlanner(C["astar"], V, (-12.5, -5.0))
    ix0, iy = p.to_cell(-2.0, 7.0)[0], p.to_cell(0, 7.0)[1]
    g[ix0:ix0 + int(6 / V), iy:iy + 2] = OCC               # 6 m wide wall at y = 7 from x = -2 to 4
    r = p.plan(g, (0.0, 0.0), (0.0, 15.0))
    assert r.found and not r.partial
    xs = np.array([q[0] for q in r.path])
    assert xs.min() < -2.0 - C["astar"]["clearance_m"] + 0.3 or xs.max() > 4.0 + C["astar"]["clearance_m"] - 0.3   # detours beyond the wall end
    # every path point keeps the clearance to the occupied cells
    dist = p.traversable(g)
    for x, y in r.path:
        i, j = p.to_cell(x, y)
        assert dist[i, j]


def test_astar_never_plans_through_unknown(C):
    g = _grid()
    p = AStarPlanner(C["astar"], V, (-12.5, -5.0))
    j0 = p.to_cell(0, 8.0)[1]
    g[:, j0:] = UNK                                         # everything beyond y = 8 is unknown
    r = p.plan(g, (0.0, 0.0), (0.0, 15.0))
    assert r.found and r.partial                            # progress toward the goal only inside known-free space
    assert max(q[1] for q in r.path) < 8.0


def test_astar_reports_failure_when_boxed_in(C):
    g = _grid()
    p = AStarPlanner(C["astar"], V, (-12.5, -5.0))
    i, j = p.to_cell(0, 0)
    g[i - 12:i + 12, j - 12:j + 12] = OCC
    g[i - 2:i + 2, j - 2:j + 2] = FREE
    r = p.plan(g, (0.0, 0.0), (0.0, 15.0))
    assert not r.found or r.partial


def test_vfh_free_space_goes_straight_to_target(C):
    g = _grid()
    v = VfhPlus(C["vfh"], V, 0.5)
    r = v.choose(g, (-12.5, -5.0), (0.0, 0.0), math.pi / 2, math.pi / 2)
    assert abs(r.deviation_deg) < 6 and not r.blocked_target and r.speed_scale == pytest.approx(1.0)


def test_vfh_steers_around_an_obstacle_in_front(C):
    g = _grid()
    o = (-12.5, -5.0)
    i, j = int((0.0 - o[0]) / V), int((2.2 - o[1]) / V)
    g[i - 2:i + 3, j - 1:j + 2] = OCC                         # 1.25 m-wide obstacle 2.2 m ahead (North)
    v = VfhPlus(C["vfh"], V, 0.5)
    r = v.choose(g, o, (0.0, 0.0), math.pi / 2, math.pi / 2)
    assert r.blocked_target
    assert abs(r.deviation_deg) > 15                           # a real deviation
    assert r.speed_scale < 1.0
    # the chosen direction must not point into the obstacle
    x, y = 2.2 * math.cos(r.direction) / max(math.sin(r.direction), 1e-3), 2.2
    assert not (abs(x) < 0.9 and math.sin(r.direction) > 0)


def test_vfh_reports_no_opening_when_surrounded(C):
    g = _grid()
    o = (-12.5, -5.0)
    ci, cj = int((0.0 - o[0]) / V), int((0.0 - o[1]) / V)
    for k in range(-8, 9):
        for m in (-8, 8):
            g[ci + k, cj + m] = OCC
            g[ci + m, cj + k] = OCC
    v = VfhPlus(C["vfh"], V, 0.6)
    r = v.choose(g, o, (0.0, 0.0), 0.0, 0.0)
    assert r.speed_scale < 0.5 or r.free_openings == 0
