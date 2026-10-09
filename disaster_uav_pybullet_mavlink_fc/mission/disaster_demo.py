"""
Deterministic disaster-search mission logic. The SEARCH OBJECTIVE is scripted (an ordered list of search waypoints, then return),
but every metre of motion is generated online by A* (global, on the live metric map) + VFH+ (local avoidance).

Inputs  : SLAM pose (T_map_body), planning grid of the live geo-semantic map, SLAM status, FC state (armed/mode/link) from MAVLink
          telemetry, survivor events
Output  : velocity command in the map/ENU frame:  vx = East, vy = North, vz = Up [m/s], yaw_rate CCW-positive [rad/s]
          (the navigation node publishes it as geometry_msgs/Twist on /uav/cmd_vel with cmd_frame = enu)
          + actions for the MAVLink bridge: "ARM", "OFFBOARD", "LAND"
Never used: simulator state, ground truth, survivor/obstacle truth.

States  : WAIT_SLAM -> ARM -> OFFBOARD -> TAKEOFF -> SEARCH(k) -> RETURN -> HOLD_END -> LANDING -> DONE
          pauses   : INSPECT (new confirmed survivor), LOST_HOLD (localisation lost / stale), MAP_STALE, NO_ROUTE_HOLD (no A* route),
                     VFH_BLOCKED (no free VFH+ opening), AVOID_RETREAT (obstacle inside the minimum distance), LINK_LOST (no MAVLink telemetry)
          guard    : FRONT_CLEARANCE_HOLD / FRONT_CLEARANCE_RETREAT (depth-image clearance straight ahead, independent of the SLAM pose)
          safe exits: LANDING (localisation lost too long / FC failsafe / end of mission), ABORTED (vehicle disarmed in flight)
A waypoint that stays unreachable (no A* route or no VFH+ opening for longer than the timeout) is skipped and reported; the mission
never falls back to flying straight at a goal through unknown space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from planning.astar import AStarPlanner, DYN, FREE, OCC, UNK
from planning.vfh_plus import VfhPlus


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0


@dataclass
class Step:
    cmd: Command
    actions: List[str] = field(default_factory=list)      # "ARM", "OFFBOARD", "LAND"
    status: dict = field(default_factory=dict)
    path: List[tuple] = field(default_factory=list)


class MissionController:
    def __init__(self, cfg: dict, voxel_m: float, map_origin_xy):
        self.cfg = cfg
        self.d = cfg["demo"]
        self.n = cfg["nav"]
        self.v = voxel_m
        self.origin = np.array(map_origin_xy, float)
        self.planner = AStarPlanner(cfg["astar"], voxel_m, map_origin_xy)
        self.vfh = VfhPlus(cfg["vfh"], voxel_m, vehicle_radius_m=0.45)
        self.wps = [(w["name"], tuple(w["xy"])) for w in self.d["search_waypoints_local"]]
        if self.d.get("return_home", True):
            self.wps.append(("RETURN home", (0.0, 0.0)))
        self.state = "WAIT_SLAM"
        self.idx = 0
        self.path: List[tuple] = []
        self.path_partial = False
        self.last_plan_t = -1e9
        self.last_map_version = -1
        self.pause_until = -1.0
        self.pause_reason = ""
        self.hold_until = None
        self.last_good = None
        self.lost_since = None
        self.stats = {"replans": 0, "vfh_deviations": 0, "vfh_active_ticks": 0, "max_vfh_deviation_deg": 0.0, "lost_ticks": 0,
                      "astar_failures": 0, "blocked_path_replans": 0, "no_route_ticks": 0, "vfh_no_opening_ticks": 0, "retreat_ticks": 0,
                      "map_stale_ticks": 0, "link_lost_ticks": 0, "invalid_cmds": 0, "skipped_waypoints": 0, "failsafe_events": 0}
        self.events: List[dict] = []                         # safety / mission events (also logged by the nav node)
        self.skipped: List[str] = []
        self.known_confirmed = set()
        self.tick = 0
        self.t_start = None
        self.done = False
        self.mission_time = 0.0
        self.log = []
        self.no_route_since = None
        self.blocked_since = None
        self.end_reason = ""
        self.end_action = self.d.get("end_action", "hold")   # "land" | "hold"
        self.was_airborne = False
        self._paused = None
        self.dyn_block_since = None
        self.dyn_cross_until = -1.0
        self._front_guard_level = None                       # None | "hold" | "retreat" (sensor-frame front guard)

    # ------------------------------------------------------------------ helpers
    def _carrot(self, pos):
        L = self.n["carrot_lookahead_m"]
        pts = self.path
        if len(pts) < 2:
            return pts[-1] if pts else None
        # closest point on the polyline, then walk L metres along it
        best, bi, bt = 1e9, 0, 0.0
        for i in range(len(pts) - 1):
            a, b = np.array(pts[i]), np.array(pts[i + 1])
            ab = b - a
            t = float(np.clip(np.dot(pos - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
            d = float(np.linalg.norm(a + t * ab - pos))
            if d < best:
                best, bi, bt = d, i, t
        rem = L
        p = np.array(pts[bi]) + bt * (np.array(pts[bi + 1]) - np.array(pts[bi]))
        i = bi
        while rem > 0 and i < len(pts) - 1:
            nxt = np.array(pts[i + 1])
            seg = float(np.linalg.norm(nxt - p))
            if seg >= rem:
                return tuple(p + (nxt - p) / max(seg, 1e-9) * rem)
            rem -= seg
            p, i = nxt, i + 1
        return tuple(pts[-1])

    def _path_blocked(self, grid):
        free = self.planner.inflated
        if free is None or not self.path:
            return False
        for x, y in self.path[:: max(1, int(0.5 / self.v))]:
            i, j = self.planner.to_cell(x, y)
            if 0 <= i < free.shape[0] and 0 <= j < free.shape[1] and grid[i, j] == OCC:
                return True
        return False


    def target(self):
        return self.wps[min(self.idx, len(self.wps) - 1)]

    def _event(self, t, kind, **kw):
        ev = {"t": round(float(t), 2), "event": kind, **kw}
        self.events.append(ev)
        return ev

    def _advance_waypoint(self, t, reason):
        name = self.target()[0]
        self.stats["skipped_waypoints"] += 1
        self.skipped.append(name)
        self._event(t, "WAYPOINT_SKIPPED", target=name, reason=reason)
        self.idx += 1
        self.path, self.path_partial = [], False
        self.no_route_since = self.blocked_since = None

    def _finish(self, t, reason):
        """End of the search (normal or forced): land if configured / required, otherwise hover."""
        self.end_reason = reason
        self.state = "HOLD_END"
        self.hold_until = None
        self._event(t, "MISSION_END", reason=reason, end_action=self.end_action)

    def _start_landing(self, t, reason, failsafe=True):
        if self.state != "LANDING":
            self.state = "LANDING"
            self.end_reason = reason if failsafe or not self.end_reason else self.end_reason
            if failsafe:
                self.stats["failsafe_events"] += 1
            self._event(t, "LAND_REQUESTED", reason=reason, failsafe=failsafe)

    def _pause(self, t, state, status, **kw):
        """Enter / stay in a hover pause; logs one event per entry."""
        if self._paused != state:
            self._paused = state
            self._event(t, state, **kw)
        status.update(state=state, **kw)

    @staticmethod
    def _sanitize(cmd: Command, stats) -> Command:
        vals = (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate)
        if not all(math.isfinite(float(v)) for v in vals):
            stats["invalid_cmds"] += 1
            return Command()
        return cmd

    # ------------------------------------------------------------------ main step
    def step(self, t, pose_xyz, yaw_enu, grid, map_version, slam_ok, slam_age_s, armed, mode, confirmed_ids=(), geo_ready=True,
             fc_link_ok=True, grid_age_s=0.0, front_clear_m=None, front_age_s=99.0):
        """All arguments come from SLAM / MAVLink telemetry / the live map. Returns Step (command already sanitised)."""
        st = self._step(t, pose_xyz, yaw_enu, grid, map_version, slam_ok, slam_age_s, armed, mode, confirmed_ids, geo_ready,
                        fc_link_ok, grid_age_s, front_clear_m, front_age_s)
        st.cmd = self._sanitize(st.cmd, self.stats)
        st.status.setdefault("stats", dict(self.stats))
        if self.events:
            st.status["last_event"] = self.events[-1]
        return st

    def _step(self, t, pose_xyz, yaw_enu, grid, map_version, slam_ok, slam_age_s, armed, mode, confirmed_ids, geo_ready,
              fc_link_ok, grid_age_s, front_clear_m=None, front_age_s=99.0):
        self.tick += 1
        d, n = self.d, self.n
        cmd, actions = Command(), []
        if self.t_start is None:
            self.t_start = t
        self.mission_time = t - self.t_start
        pos = np.array(pose_xyz[:2], float) if pose_xyz is not None else None
        status = {"t": t, "state": self.state, "target": self.target()[0], "mission_time": self.mission_time}
        flying = self.state not in ("WAIT_SLAM", "ARM", "OFFBOARD", "DONE", "ABORTED")

        if self.state in ("DONE", "ABORTED"):
            status["state"] = self.state
            return Step(cmd, actions, status)

        # ---- vehicle-side safety first (from MAVLink telemetry; never from the simulator)
        if flying and self.was_airborne and not armed:
            if self.state == "LANDING":
                self.state, self.done = "DONE", True
                self._event(t, "LANDED", reason=self.end_reason)
            else:
                self.state, self.done = "ABORTED", True
                self.stats["failsafe_events"] += 1
                self._event(t, "ABORTED", reason=f"vehicle disarmed in flight (mode {mode})")
            status["state"] = self.state
            return Step(cmd, actions, status)
        if flying and armed:
            self.was_airborne = True
        if self.state == "LANDING":
            if mode != "LAND":
                actions.append("LAND")
            status.update(state="LANDING", reason=self.end_reason)
            return Step(cmd, actions, status)
        if flying and mode in ("FAILSAFE_HOLD", "LAND"):
            # the FC has taken over (setpoint-loss watchdog, low battery, external LAND): do not fight it, finish with a landing
            self._start_landing(t, f"flight controller in {mode}")
            if mode != "LAND":
                actions.append("LAND")
            status.update(state="LANDING", reason=self.end_reason)
            return Step(cmd, actions, status)
        if flying and not fc_link_ok:
            # no MAVLink telemetry: whatever we send may not arrive; command a hover (the FC's own watchdog handles a dead link)
            self.stats["link_lost_ticks"] += 1
            self._pause(t, "LINK_LOST", status)
            return Step(cmd, actions, status, self.path)
        if flying and mode == "HOLD" and self.state not in ("TAKEOFF",) and self.was_airborne:
            # FC dropped out of OFFBOARD without a failsafe (e.g. a rejected mode switch): hover and ask for OFFBOARD again
            actions.append("OFFBOARD")
            self._pause(t, "FC_NOT_OFFBOARD", status)
            return Step(cmd, actions, status, self.path)

        # ---- tracking loss / stale pose: never fall back to ground truth, hold and reacquire, land if it does not come back
        pose_bad = (pose_xyz is None) or (not slam_ok) or (slam_age_s > n["stale_pose_s"])
        if flying and pose_bad:
            self.stats["lost_ticks"] += 1
            if self.lost_since is None:
                self.lost_since = t
                self._event(t, "LOCALIZATION_LOST", slam_ok=bool(slam_ok), pose_age_s=round(float(min(slam_age_s, 99.0)), 2))
            status.update(state="LOST_HOLD", lost_for_s=t - self.lost_since)
            if t - self.lost_since > n.get("lost_land_s", 30.0):
                self._start_landing(t, f"localisation lost for {t - self.lost_since:.1f} s")
                actions.append("LAND")
                status["state"] = "LANDING"
            elif t - self.lost_since > n["lost_hold_s"] and self.last_good is not None:
                cmd.yaw_rate = 0.2 * float(np.sign(_wrap(self.last_good["yaw"] - (yaw_enu if yaw_enu is not None else 0.0))))   # slow sweep back to a familiar view
            return Step(cmd, actions, status, self.path)
        if not pose_bad:
            if self.lost_since is not None:
                self._event(t, "LOCALIZATION_RECOVERED", lost_for_s=round(t - self.lost_since, 2))
            self.lost_since = None
            self.last_good = {"pos": pos.copy(), "yaw": yaw_enu, "t": t, "z": pose_xyz[2]}

        z = pose_xyz[2] if pose_xyz is not None else 0.0

        # ---- start-up sequence
        if self.state == "WAIT_SLAM":
            if not pose_bad and geo_ready and grid is not None and grid.size > 4:
                self.state = "ARM"
            status["state"] = self.state
            return Step(cmd, actions, status)
        if self.state == "ARM":
            actions.append("ARM")
            if armed:
                self.state = "OFFBOARD"
            status["state"] = self.state
            return Step(cmd, actions, status)
        if self.state == "OFFBOARD":
            actions.append("OFFBOARD")
            if mode == "OFFBOARD":
                self.state = "TAKEOFF"
                self._event(t, "TAKEOFF")
            status["state"] = self.state
            return Step(cmd, actions, status)

        # ---- altitude hold (all flying states)
        alt = d["flight_altitude_m"]
        cmd.vz = float(np.clip(1.0 * (alt - z), -d["max_climb_mps"], d["max_climb_mps"]))

        if self.state == "TAKEOFF":
            if abs(alt - z) < d["takeoff_tolerance_m"]:
                self.state = "SEARCH"
                self.idx = 0
            status["state"] = self.state
            return Step(cmd, actions, status)

        if self.state == "HOLD_END":
            status.update(state=self.state, reason=self.end_reason)
            if self.hold_until is None:
                self.hold_until = t + d["hold_after_mission_s"]
            if t >= self.hold_until:
                if self.end_action == "land":
                    self._start_landing(t, self.end_reason or "mission complete", failsafe=False)
                    actions.append("LAND")
                    status["state"] = "LANDING"
                else:
                    self.state, self.done = "DONE", True
                    self._event(t, "DONE", reason=self.end_reason, end_action="hold")
            return Step(cmd, actions, status)

        # ---- the map must be live: a stale planning grid means the sensors / mapping stopped -> hover
        if grid_age_s > n.get("grid_stale_s", 3.0):
            self.stats["map_stale_ticks"] += 1
            self._pause(t, "MAP_STALE", status)
            status["grid_age_s"] = grid_age_s
            return Step(cmd, actions, status, self.path)
        self._paused = None

        # ---- new confirmed survivor -> short INSPECT pause (hover, so the operator sees the tag being created)
        new = [i for i in confirmed_ids if i not in self.known_confirmed]
        if new:
            self.known_confirmed.update(new)
            self.pause_until, self.pause_reason = t + 2.5, f"survivor #{new[0]:02d} confirmed"
            self._event(t, "SURVIVOR_CONFIRMED", track_ids=list(new))
        if t < self.pause_until:
            status.update(state="INSPECT", note=self.pause_reason)
            return Step(cmd, actions, status, self.path)

        # ---- waypoint sequencing
        name, goal = self.target()
        goal_a = np.array(goal)
        dist_goal = float(np.linalg.norm(goal_a - pos))
        if dist_goal < d["goal_tolerance_m"]:
            self._event(t, "WAYPOINT_REACHED", target=name)
            self.idx += 1
            self.path = []
            self.no_route_since = self.blocked_since = None
        if self.idx >= len(self.wps):
            self._finish(t, "mission complete" if not self.skipped else f"mission complete ({len(self.skipped)} waypoint(s) unreachable)")
            status["state"] = self.state
            return Step(cmd, actions, status)
        name, goal = self.target()
        goal_a = np.array(goal)
        dist_goal = float(np.linalg.norm(goal_a - pos))
        self.state = "RETURN" if name.startswith("RETURN") else "SEARCH"

        # ---- A*: (re)plan when there is no path, periodically, when the map changed and the path got blocked, or on arrival of new info
        blocked_now = self.cfg["astar"]["replan_on_map_change"] and map_version != self.last_map_version and self._path_blocked(grid)
        need = (not self.path) or (t - self.last_plan_t > self.cfg["astar"]["replan_period_s"]) or blocked_now
        if need:
            if blocked_now:
                self.stats["blocked_path_replans"] += 1
                self._event(t, "PATH_BLOCKED_REPLAN", target=name)
            res = self.planner.plan(grid, tuple(pos), tuple(goal))
            self.last_plan_t, self.last_map_version = t, map_version
            self.stats["replans"] += 1
            if res.found:
                self.path, self.path_partial = res.path, res.partial
            else:
                self.stats["astar_failures"] += 1
                if self.path and not self._path_blocked(grid):
                    pass                                            # keep following the last valid route while re-planning
                else:
                    self.path, self.path_partial = [], False
            status["astar"] = {"found": res.found, "partial": res.partial, "expanded": res.expanded, "n_pts": len(res.path), "reason": res.reason}

        # ---- remembered moving-obstacle regions (map.dynamic_memory) are blocked for A*. If they alone keep us from a complete route for a while, wait
        #      first (the obstacle may have moved on), then cross them with caution (A* cost mode; VFH+ and the front guard still react to what is sensed)
        #      instead of giving the waypoint up. Reverts to blocking after dynamic_cross_s.
        if self.planner.dynamic_mode_override and t > self.dyn_cross_until:
            self.planner.dynamic_mode_override = None
            self._event(t, "DYNAMIC_REGION_BLOCKING_RESTORED")
        if n.get("dynamic_wait_s") is not None and (not self.path or self.path_partial) and grid is not None and bool((grid == DYN).any()):
            if self.dyn_block_since is None:
                self.dyn_block_since = t
            if t - self.dyn_block_since > n.get("dynamic_wait_s", 10.0) and not self.planner.dynamic_mode_override:
                self.planner.dynamic_mode_override = "penalty"
                self.dyn_cross_until = t + n.get("dynamic_cross_s", 60.0)
                self.last_plan_t = -1e9
                self._event(t, "DYNAMIC_REGION_CROSSING", target=name, waited_s=round(t - self.dyn_block_since, 1))
                res = self.planner.plan(grid, tuple(pos), tuple(goal))
                self.last_plan_t, self.stats["replans"] = t, self.stats["replans"] + 1
                if res.found:
                    self.path, self.path_partial = res.path, res.partial
        else:
            self.dyn_block_since = None

        # ---- no A* route: hover (keep altitude), look toward the goal so the map can grow, re-plan; skip the waypoint after a timeout
        if not self.path:
            self.stats["no_route_ticks"] += 1
            if self.no_route_since is None:
                self.no_route_since = t
                self._event(t, "NO_ROUTE", target=name)
            yaw = yaw_enu if yaw_enu is not None else 0.0
            cmd.yaw_rate = float(np.clip(1.2 * _wrap(math.atan2(goal[1] - pos[1], goal[0] - pos[0]) - yaw), -d["max_yaw_rate_rps"], d["max_yaw_rate_rps"]))
            waited = t - self.no_route_since
            if waited > n.get("no_route_timeout_s", 10.0):
                self._advance_waypoint(t, "no A* route")
            status.update(state="NO_ROUTE_HOLD", target=name, dist_to_goal=dist_goal, no_route_for_s=waited)
            return Step(cmd, actions, status, self.path)
        self.no_route_since = None

        carrot = self._carrot(pos)
        frontier_wait = carrot is None or float(np.linalg.norm(np.array(carrot) - pos)) < 0.05
        if frontier_wait:
            # end of a (partial) path reached: the next step lies in unknown space -> do NOT fly into it; turn toward the goal and re-plan
            carrot = tuple(goal)
        tgt_dir = math.atan2(carrot[1] - pos[1], carrot[0] - pos[0])

        # ---- VFH+ local avoidance
        yaw = yaw_enu if yaw_enu is not None else tgt_dir
        vres = self.vfh.choose(np.where(grid == OCC, OCC, FREE).astype(np.int8), tuple(self.origin), tuple(pos), tgt_dir, yaw)
        if vres.blocked_target:
            self.stats["vfh_active_ticks"] += 1
            if abs(vres.deviation_deg) > 10:
                self.stats["vfh_deviations"] += 1
            self.stats["max_vfh_deviation_deg"] = max(self.stats["max_vfh_deviation_deg"], abs(vres.deviation_deg))
        vfh_status = {"dir_deg": math.degrees(vres.direction), "target_dir_deg": math.degrees(tgt_dir), "blocked": vres.blocked_target,
                      "deviation_deg": vres.deviation_deg, "nearest_obstacle_m": vres.nearest_obstacle_m if vres.nearest_obstacle_m != float("inf") else None,
                      "speed_scale": vres.speed_scale, "reason": vres.reason}

        # ---- obstacle inside the minimum distance: stop progressing and back away from it (direction from the map, not a fixed side)
        dmin = n.get("min_obstacle_dist_m", 0.7)
        if vres.nearest_obstacle_m < dmin and vres.nearest_bearing is not None:
            self.stats["retreat_ticks"] += 1
            if self.stats["retreat_ticks"] == 1 or (self.events and self.events[-1]["event"] != "OBSTACLE_TOO_CLOSE"):
                self._event(t, "OBSTACLE_TOO_CLOSE", distance_m=round(vres.nearest_obstacle_m, 2))
            away = vres.nearest_bearing + math.pi
            v_r = n.get("retreat_speed_mps", 0.3)
            cmd.vx, cmd.vy = v_r * math.cos(away), v_r * math.sin(away)
            self.path = []                                          # force a fresh A* plan from the new position
            status.update(state="AVOID_RETREAT", target=name, dist_to_goal=dist_goal, vfh=vfh_status)
            return Step(cmd, actions, status, self.path)

        # ---- VFH+ found no opening: hover, re-plan globally; skip the waypoint if it stays blocked
        if vres.speed_scale <= 0.0 and vres.reason:
            self.stats["vfh_no_opening_ticks"] += 1
            if self.blocked_since is None:
                self.blocked_since = t
                self._event(t, "VFH_NO_OPENING", target=name, reason=vres.reason)
            self.last_plan_t = -1e9                                 # re-plan on the next tick
            if t - self.blocked_since > n.get("blocked_timeout_s", 12.0):
                self._advance_waypoint(t, f"VFH+: {vres.reason}")
            status.update(state="VFH_BLOCKED", target=name, dist_to_goal=dist_goal, vfh=vfh_status)
            return Step(cmd, actions, status, self.path)
        self.blocked_since = None

        speed = d["cruise_speed_mps"] * vres.speed_scale
        if self.path_partial:
            speed = min(speed, 0.8 * d["cruise_speed_mps"])
        if frontier_wait:
            speed = 0.0                                             # at the edge of known-free space: rotate / observe only
        speed = min(speed, max(0.3, dist_goal * 0.6))                # decelerate into the waypoint
        # never move faster than the camera can see: slow while the heading is far from the direction of travel
        yerr = _wrap(vres.direction - yaw)
        if abs(yerr) > math.radians(30):
            speed *= max(0.0, math.cos(min(abs(yerr), math.pi / 2)))
            speed = min(speed, d["slow_speed_mps"])
        cmd.vx, cmd.vy = speed * math.cos(vres.direction), speed * math.sin(vres.direction)
        cmd.yaw_rate = float(np.clip(1.2 * yerr, -d["max_yaw_rate_rps"], d["max_yaw_rate_rps"]))

        # ---- sensor-frame guard: depth-image clearance straight ahead (no SLAM pose, no map). A moving / just-collapsed object can offset the
        #      SLAM-frame map by a metre (measured with ORB-SLAM3), so the map-based checks above cannot be the only protection.
        fg = n.get("front_guard", {})
        if front_clear_m is not None and front_age_s <= fg.get("max_age_s", 1.5) and yaw_enu is not None:
            fwd = np.array([math.cos(yaw_enu), math.sin(yaw_enu)])
            if front_clear_m < fg.get("retreat_m", 0.75):
                v_r = n.get("retreat_speed_mps", 0.3)
                cmd.vx, cmd.vy = -v_r * fwd[0], -v_r * fwd[1]       # back away from what the camera sees ahead
                self.path = []                                       # fresh A* plan once clear
                self.stats["front_guard_retreat_ticks"] = self.stats.get("front_guard_retreat_ticks", 0) + 1
                if self._front_guard_level != "retreat":
                    self._event(t, "FRONT_CLEARANCE_RETREAT", clearance_m=round(front_clear_m, 2))
                self._front_guard_level = "retreat"
                status.update(state="AVOID_RETREAT", target=name, dist_to_goal=dist_goal, front_clearance_m=front_clear_m, vfh=vfh_status)
                return Step(cmd, actions, status, self.path)
            if front_clear_m < fg.get("stop_m", 1.0):
                along = cmd.vx * fwd[0] + cmd.vy * fwd[1]
                if along > 0.0:
                    cmd.vx, cmd.vy = cmd.vx - along * fwd[0], cmd.vy - along * fwd[1]    # no forward motion; sideways motion left to VFH+/A*
                    self.stats["front_guard_hold_ticks"] = self.stats.get("front_guard_hold_ticks", 0) + 1
                    if self._front_guard_level is None:
                        self._event(t, "FRONT_CLEARANCE_HOLD", clearance_m=round(front_clear_m, 2))
                    if self._front_guard_level != "retreat":
                        self._front_guard_level = "hold"
                    status["front_guard"] = "hold"
            else:
                self._front_guard_level = None
        status["front_clearance_m"] = front_clear_m
        status.update(state=self.state, target=name, dist_to_goal=dist_goal, speed=speed, vfh=vfh_status,
                      path_partial=self.path_partial, frontier_wait=frontier_wait, stats=dict(self.stats))
        self.log.append((t, float(pos[0]), float(pos[1]), self.state, name, vres.blocked_target, vres.deviation_deg, speed))
        return Step(cmd, actions, status, self.path)
