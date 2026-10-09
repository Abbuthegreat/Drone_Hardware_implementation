"""Mission-level safety behaviour of mission.disaster_demo.MissionController on synthetic live-map grids (no simulator, no ground truth)."""
import copy
import math

import numpy as np
import pytest

from mission.disaster_demo import Command, MissionController
from planning.astar import FREE, OCC, UNK

V = 0.25
ORIGIN = (-25.0, -12.0)


@pytest.fixture
def C(cfg):
    c = copy.deepcopy(cfg)
    c["demo"]["search_waypoints_local"] = [{"name": "WP1", "xy": [0.0, 8.0]}, {"name": "WP2", "xy": [4.0, 8.0]}]
    c["demo"]["return_home"] = False
    c["demo"]["end_action"] = "land"
    c["demo"]["hold_after_mission_s"] = 1.0
    return c


def _grid(fill=FREE):
    return np.full((200, 228), fill, np.int8)


def _cell(x, y):
    return int(math.floor((x - ORIGIN[0]) / V)), int(math.floor((y - ORIGIN[1]) / V))


class Pilot:
    """Feeds the controller like the nav node does; t advances 0.1 s per tick."""

    def __init__(self, cfg, grid):
        self.mc = MissionController(cfg, V, ORIGIN)
        self.grid, self.t = grid, 0.0
        self.alt = cfg["demo"]["flight_altitude_m"]
        self.armed, self.mode = False, "HOLD"

    def step(self, xy=(0.0, 0.0), z=None, yaw=math.pi / 2, slam_ok=True, age=0.0, link=True, grid_age=0.0, grid=None, confirmed=()):
        self.t += 0.1
        z = self.alt if z is None else z
        return self.mc.step(self.t, np.array([xy[0], xy[1], z]), yaw, self.grid if grid is None else grid, 1, slam_ok, age,
                            self.armed, self.mode, list(confirmed), True, fc_link_ok=link, grid_age_s=grid_age)

    def to_search(self, xy=(0.0, 0.0)):
        st = self.step(xy, z=0.0)                      # WAIT_SLAM -> ARM
        st = self.step(xy, z=0.0)
        assert "ARM" in st.actions
        self.armed = True
        st = self.step(xy, z=0.0)                      # -> OFFBOARD
        st = self.step(xy, z=0.0)
        assert "OFFBOARD" in st.actions
        self.mode = "OFFBOARD"
        self.step(xy, z=0.0)                           # -> TAKEOFF
        st = self.step(xy)                             # at altitude -> SEARCH
        assert self.mc.state == "SEARCH", self.mc.state
        return st


def test_nominal_search_flies_along_the_astar_route(C):
    p = Pilot(C, _grid())
    p.to_search()
    st = p.step()
    assert st.status["state"] == "SEARCH" and len(st.path) >= 2
    assert st.cmd.vy > 0.3 and abs(st.cmd.vx) < 0.2           # toward WP1 (North) through free space


def test_no_astar_route_hovers_instead_of_flying_at_the_goal_and_skips_after_timeout(C):
    g = _grid(UNK)
    i, j = _cell(0.0, 0.0)
    g[i - 3:i + 4, j - 3:j + 4] = OCC                         # boxed in by debris within 0.75 m
    g[i, j] = FREE
    p = Pilot(C, g)
    p.to_search()
    st = p.step()
    assert st.status["state"] == "NO_ROUTE_HOLD"
    assert st.cmd.vx == 0.0 and st.cmd.vy == 0.0                # no horizontal motion toward the goal
    for _ in range(int(C["nav"]["no_route_timeout_s"] / 0.1) + 5):
        st = p.step()
    assert any(e["event"] == "WAYPOINT_SKIPPED" and e["target"] == "WP1" for e in p.mc.events)
    assert p.mc.skipped and p.mc.skipped[0] == "WP1"


@pytest.mark.parametrize("side", [+1, -1])
def test_obstacle_too_close_backs_away_from_it_whichever_side_it_is_on(C, side):
    g = _grid()
    i, j = _cell(side * 0.5, 0.0)
    g[i, j - 1:j + 2] = OCC                                   # debris 0.5 m East (side=+1) or West (side=-1) of the UAV
    p = Pilot(C, g)
    p.to_search()
    st = p.step()
    assert st.status["state"] == "AVOID_RETREAT"
    assert np.sign(st.cmd.vx) == -side and abs(st.cmd.vx) > 0.2  # moves AWAY from the obstacle (direction from the map, not a fixed side)
    assert math.hypot(st.cmd.vx, st.cmd.vy) <= C["nav"]["retreat_speed_mps"] + 1e-6


def test_fc_failsafe_hold_leads_to_landing_then_done(C):
    p = Pilot(C, _grid())
    p.to_search()
    p.mode = "FAILSAFE_HOLD"                                   # FC setpoint-loss watchdog fired (reported over MAVLink)
    st = p.step()
    assert st.status["state"] == "LANDING" and "LAND" in st.actions
    assert st.cmd.vx == st.cmd.vy == st.cmd.vz == 0.0
    p.mode = "LAND"
    st = p.step()
    assert "LAND" not in st.actions and st.status["state"] == "LANDING"
    p.armed = False                                            # FC auto-disarms after touchdown
    st = p.step()
    assert p.mc.state == "DONE" and p.mc.done
    assert any(e["event"] == "LANDED" for e in p.mc.events) and p.mc.stats["failsafe_events"] == 1


def test_unexpected_disarm_in_flight_aborts(C):
    p = Pilot(C, _grid())
    p.to_search()
    p.armed = False
    p.step()
    assert p.mc.state == "ABORTED" and p.mc.done


def test_localisation_loss_holds_then_lands(C):
    p = Pilot(C, _grid())
    p.to_search()
    st = p.step(slam_ok=False)
    assert st.status["state"] == "LOST_HOLD" and st.cmd.vx == 0.0 and st.cmd.vy == 0.0
    for _ in range(int(C["nav"]["lost_land_s"] / 0.1) + 3):
        st = p.step(slam_ok=False, age=5.0)
    assert p.mc.state == "LANDING" and "LAND" in st.actions
    assert any(e["event"] == "LOCALIZATION_LOST" for e in p.mc.events)


def test_localisation_recovery_resumes_search(C):
    p = Pilot(C, _grid())
    p.to_search()
    for _ in range(5):
        p.step(slam_ok=False)
    st = p.step()
    assert st.status["state"] == "SEARCH" and any(e["event"] == "LOCALIZATION_RECOVERED" for e in p.mc.events)


def test_mavlink_link_loss_and_stale_map_hover(C):
    p = Pilot(C, _grid())
    p.to_search()
    st = p.step(link=False)
    assert st.status["state"] == "LINK_LOST" and st.cmd == Command()
    st = p.step(grid_age=C["nav"]["grid_stale_s"] + 1.0)
    assert st.status["state"] == "MAP_STALE" and st.cmd.vx == 0.0 and st.cmd.vy == 0.0
    st = p.step()
    assert st.status["state"] == "SEARCH"


def test_invalid_command_is_replaced_by_a_hover():
    stats = {"invalid_cmds": 0}
    out = MissionController._sanitize(Command(float("nan"), 1.0, 0.0, 0.0), stats)
    assert out == Command() and stats["invalid_cmds"] == 1
    ok = Command(0.5, 0.2, 0.1, 0.0)
    assert MissionController._sanitize(ok, stats) is ok


def test_mission_completion_lands_and_finishes(C):
    p = Pilot(C, _grid())
    p.to_search()
    p.step((0.0, 8.0))                                         # at WP1
    p.step((4.0, 8.0))                                         # at WP2 -> end of mission
    assert p.mc.state == "HOLD_END"
    st = None
    for _ in range(int(C["demo"]["hold_after_mission_s"] / 0.1) + 3):
        st = p.step((4.0, 8.0))
    assert p.mc.state == "LANDING" and "LAND" in st.actions and p.mc.stats["failsafe_events"] == 0
    p.mode, p.armed = "LAND", False
    p.step((4.0, 8.0), z=0.05)
    assert p.mc.done and p.mc.state == "DONE" and p.mc.end_reason == "mission complete"
