"""Moving-obstacle memory in the metric map: confidently occupied -> later observed free = something moved; keep the region blocked for planning."""
import copy

import numpy as np

from mapping.metric_map import DYNAMIC, FREE, OCCUPIED, VoxelMap3D


def _map(cfg, enabled=True):
    c = copy.deepcopy(cfg["map"])
    c["dynamic_memory"] = {"enabled": enabled, "occ_l": 1.6, "min_cells": 3, "hold_s": 30.0, "radius_m": 0.5}
    return VoxelMap3D(c)


def _patch(m, ix, iy, val, z=2.0):
    m.L[ix - 1:ix + 2, iy - 1:iy + 2, :] = val                 # whole column, so the whole flight band is covered
    m.seen[ix - 3:ix + 4, iy - 3:iy + 4] = True


def test_cleared_obstacle_stays_blocked_for_hold_time_then_expires(cfg):
    m = _map(cfg)
    m.planning_grid(2.0)                                       # sets the flight band
    ix, iy = m.shape[0] // 2, m.shape[1] // 2
    m.t_now = 10.0
    _patch(m, ix, iy, 3.0)                                     # confidently occupied
    m._update_dynamic()
    assert m.planning_grid(2.0)[ix, iy] == OCCUPIED
    m.t_now = 20.0
    _patch(m, ix, iy, -2.0)                                    # the obstacle moved away: seen free now
    m._update_dynamic()
    g = m.planning_grid(2.0)
    assert g[ix, iy] == DYNAMIC and g[ix + 2, iy] == DYNAMIC     # remembered (planning layer only), with the dilation radius
    assert m.n_dynamic_events == 1
    m.t_now = 60.0                                             # hold_s (30 s) expired
    assert m.planning_grid(2.0)[ix, iy] == FREE


def test_disabled_memory_changes_nothing_and_isolated_flicker_is_ignored(cfg):
    m = _map(cfg, enabled=False)
    m.planning_grid(2.0)
    ix, iy = m.shape[0] // 2, m.shape[1] // 2
    _patch(m, ix, iy, 3.0); m._update_dynamic(); _patch(m, ix, iy, -2.0); m._update_dynamic()
    assert m.planning_grid(2.0)[ix, iy] == FREE
    m2 = _map(cfg)
    m2.planning_grid(2.0)
    k = int(round((2.0 - m2.origin[2]) / m2.v))
    m2.L[ix, iy, k] = 3.0; m2.seen[ix, iy] = True; m2._update_dynamic()
    m2.L[ix, iy, k] = -2.0; m2._update_dynamic()               # single cell < min_cells -> ignored
    assert m2.n_dynamic_events == 0


def test_astar_prefers_a_detour_around_remembered_moving_region_but_can_still_cross_it(cfg):
    from planning.astar import AStarPlanner, DYN
    v = 0.25
    ca = dict(cfg["astar"], dynamic_mode="penalty")
    pl = AStarPlanner(ca, v, (0.0, 0.0))
    start, goal = (1.0, 7.5), (14.0, 7.5)                         # grid i = x / v, j = y / v
    g = np.full((64, 64), FREE, np.int8)
    g[28:34, 20:44] = DYN                                         # remembered band (x 7..8.5, y 5..11) across the direct route; free space around it
    res = pl.plan(g, start, goal)
    assert res.found and not res.partial
    assert not any(7.0 <= x <= 8.5 and 5.0 <= y <= 11.0 for x, y in res.path), "A* should detour around the remembered region when it can"
    g2 = np.full((64, 64), FREE, np.int8)
    g2[28:34, :] = DYN                                            # the band spans the whole map: no way around -> crossing is allowed (soft cost, not a wall)
    res2 = pl.plan(g2, start, goal)
    assert res2.found and not res2.partial


def test_astar_block_mode_never_enters_remembered_region(cfg):
    from planning.astar import AStarPlanner, DYN
    pl = AStarPlanner(dict(cfg["astar"], dynamic_mode="block"), 0.25, (0.0, 0.0))
    g = np.full((64, 64), FREE, np.int8)
    g[28:34, :] = DYN                                             # spans the map: in block mode there is no complete route
    res = pl.plan(g, (1.0, 7.5), (14.0, 7.5))
    assert res.partial and all(x < 7.0 for x, _ in res.path)


def test_mission_waits_then_crosses_a_remembered_moving_region_instead_of_skipping_the_waypoint(cfg):
    import copy as _copy
    from mission.disaster_demo import MissionController  # noqa: F401
    from tests.test_mission_safety import Pilot, _cell
    from planning.astar import DYN as _DYN
    c = _copy.deepcopy(cfg)
    c["demo"]["search_waypoints_local"] = [{"name": "WP1", "xy": [0.0, 8.0]}]
    c["demo"]["return_home"], c["demo"]["end_action"], c["demo"]["hold_after_mission_s"] = False, "land", 1.0
    c["nav"]["dynamic_wait_s"] = 10.0                           # the crossing safeguard is OFF by default (it let the UAV collide with the debris)
    g = np.full((200, 228), FREE, np.int8)
    i0, j0 = _cell(-25.0, 3.5)
    i1, j1 = _cell(25.0, 4.5)
    g[i0:i1, j0:j1] = _DYN                                        # remembered band across the whole map between start and WP1
    p = Pilot(c, g)
    p.to_search()
    events = set()
    for _ in range(200):                                          # 20 s: > dynamic_wait_s (10 s) but < no_route_timeout_s + wait
        st = p.step((0.0, 0.0))
        events |= {e["event"] for e in p.mc.events}
    assert "DYNAMIC_REGION_CROSSING" in events
    assert "WAYPOINT_SKIPPED" not in events and not p.mc.skipped
    assert p.mc.planner.dynamic_mode_override == "penalty"


def test_crossing_safeguard_is_off_by_default(cfg):
    assert cfg["nav"].get("dynamic_wait_s") is None and cfg["astar"]["dynamic_mode"] == "block"
