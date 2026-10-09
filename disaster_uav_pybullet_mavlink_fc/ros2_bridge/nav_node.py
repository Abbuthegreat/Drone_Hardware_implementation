"""
Navigation node: deterministic search mission executed with A* (global, live map) + VFH+ (local) -> /uav/cmd_vel.

Subscribes : /uav/slam/pose, /uav/slam/status, /uav/map/planning_grid, /uav/map/semantic, /uav/armed, /uav/mode,
             /uav/telemetry (MAVLink link health + FC failsafe flag only; its position is NOT used), /uav/imu (header time = FC clock only)
Publishes  : /uav/cmd_vel  geometry_msgs/Twist  *** FRAME = ENU of the SLAM map: linear.x = East, linear.y = North, linear.z = Up [m/s],
                           angular.z = yaw rate counter-clockwise positive [rad/s] ***  (the bridge runs with cmd_frame=enu)
             /uav/nav/status JSON, /uav/nav/path (nav_msgs/Path, slam_map), /demo/mission_done (Bool)
Calls      : /uav/arm, /uav/offboard (std_srvs/SetBool), /uav/land (std_srvs/Trigger) on the MAVLink bridge
The vehicle is commanded ONLY through /uav/cmd_vel -> bridge -> MAVLink -> simulated FC -> PyBullet. No ground truth is read.
"""
from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from common.config import load_config
from mission.disaster_demo import MissionController
from sensor_msgs.msg import Imu
from ros2_bridge.ros_util import matrix_to_pose, pose_to_matrix, s_to_stamp, stamp_to_s


class NavNode(Node):
    def __init__(self):
        super().__init__("uav_nav")
        self.cfg = load_config()
        self.log_dir = self.declare_parameter("log_dir", "").value
        self.mc = None
        self.pose_T = None
        self.pose_t = -1.0
        self.slam_state = "INIT"
        self.grid = None
        self.grid_origin = None
        self.grid_v = None
        self.map_version = 0
        self.geo_ready = False
        self.confirmed = []
        self.armed = False
        self.mode = ""
        self.t_now = 0.0
        self.grid_t = None
        self.tel_wall = None
        self.link_ok = False
        self.fc_failsafe = False
        self.n_events = 0
        self.front_clear, self.front_t = None, None
        self.pending = {}
        self.create_subscription(PoseStamped, "/uav/slam/pose", self.on_pose, 10)
        self.create_subscription(String, "/uav/slam/status", self.on_slam, 10)
        self.create_subscription(OccupancyGrid, "/uav/map/planning_grid", self.on_grid, 2)
        self.create_subscription(String, "/uav/map/semantic", self.on_sem, 5)
        self.create_subscription(Bool, "/uav/armed", lambda m: setattr(self, "armed", m.data), 5)
        self.create_subscription(String, "/uav/mode", lambda m: setattr(self, "mode", m.data), 5)
        self.create_subscription(String, "/uav/telemetry", self.on_tel, 5)
        self.create_subscription(Imu, "/uav/imu", lambda m: setattr(self, "t_now", max(self.t_now, stamp_to_s(m.header.stamp))), 50)   # simulation/FC clock only
        self.pub_cmd = self.create_publisher(Twist, "/uav/cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "/uav/nav/status", 10)
        self.pub_path = self.create_publisher(Path, "/uav/nav/path", 5)
        self.pub_done = self.create_publisher(Bool, "/demo/mission_done", 5)
        self.cli_arm = self.create_client(SetBool, "/uav/arm")
        self.cli_off = self.create_client(SetBool, "/uav/offboard")
        self.cli_land = self.create_client(Trigger, "/uav/land")
        self.csv = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.csv = open(os.path.join(self.log_dir, "nav_log.csv"), "w", buffering=1)
            self.csv.write("t,state,target,x,y,z,yaw,vx_cmd,vy_cmd,vz_cmd,yaw_rate_cmd,vfh_blocked,vfh_dev_deg,nearest_obs_m,astar_partial,path_pts,speed\n")
            self.f_ev = open(os.path.join(self.log_dir, "nav_events.jsonl"), "w", buffering=1)
            self.f_path = open(os.path.join(self.log_dir, "astar_paths.jsonl"), "w", buffering=1)
            self._last_replans = -1
        self.create_timer(1.0 / self.cfg["nav"]["control_hz"], self.tick)
        self.last_state = None
        self.done_sent = False
        self.get_logger().info("nav node up")

    # ------------------------------------------------------------------ inputs
    def on_pose(self, m):
        self.pose_T, self.pose_t = pose_to_matrix(m), stamp_to_s(m.header.stamp)
        self.t_now = max(self.t_now, self.pose_t)

    def on_slam(self, m):
        d = json.loads(m.data)
        self.slam_state = d.get("state", "INIT")

    def on_tel(self, m):
        d = json.loads(m.data)
        link = d.get("link") or {}
        self.link_ok = bool(link.get("ok", True))
        self.fc_failsafe = bool(d.get("failsafe", False))
        self.tel_wall = d.get("fc_time_boot_ms", 0) / 1000.0          # FC (simulation) time of the newest MAVLink telemetry

    def fc_link_ok(self):
        """
        MAVLink link healthy: the bridge reports fresh telemetry AND the FC time it relays keeps up with the sensor clock seen here (camera /
        SLAM stamps, simulation time). A dead link or bridge freezes the FC time while the camera time advances; a slow simulator or a busy
        node delays both equally, so neither causes a false alarm (a wall-clock age would, since the sim runs slower than real time).
        """
        if self.tel_wall is None:
            return False
        return self.link_ok and (self.t_now - self.tel_wall) < self.cfg["nav"].get("fc_link_timeout_s", 1.5)

    def on_grid(self, m: OccupancyGrid):
        g = np.array(m.data, np.int8).reshape(m.info.height, m.info.width).T          # -> [ix, iy]
        self.grid = g
        self.grid_t = stamp_to_s(m.header.stamp)
        self.grid_v = m.info.resolution
        self.grid_origin = (m.info.origin.position.x, m.info.origin.position.y)
        self.map_version = m.info.map_load_time.sec
        if self.mc is None:
            self.mc = MissionController(self.cfg, self.grid_v, self.grid_origin)
            self.get_logger().info(f"mission controller ready: {len(self.mc.wps)} waypoints")

    def on_sem(self, m):
        d = json.loads(m.data)
        self.geo_ready = d.get("geo_anchor") is not None
        self.confirmed = d.get("confirmed_ids", [])
        self.front_clear, self.front_t = d.get("front_clearance_m"), d.get("front_clearance_t")

    # ------------------------------------------------------------------ services
    def _call(self, name, client, value=None):
        if self.pending.get(name) is not None and not self.pending[name].done():
            return
        if not client.service_is_ready():
            return
        if value is None:
            req = Trigger.Request()
        else:
            req = SetBool.Request()
            req.data = value
        self.pending[name] = client.call_async(req)

    # ------------------------------------------------------------------ control loop
    def tick(self):
        if self.mc is None:
            return
        pose = self.pose_T
        pos = None if pose is None else pose[:3, 3]
        yaw = None if pose is None else math.atan2(pose[1, 0], pose[0, 0])
        age = self.t_now - self.pose_t if pose is not None else 99.0
        slam_ok = self.slam_state == "OK"
        grid_age = self.t_now - self.grid_t if self.grid_t is not None else 99.0
        st = self.mc.step(self.t_now, pos, yaw, self.grid, self.map_version, slam_ok, age, self.armed, self.mode, self.confirmed, self.geo_ready,
                          fc_link_ok=self.fc_link_ok(), grid_age_s=grid_age,
                          front_clear_m=self.front_clear, front_age_s=(self.t_now - self.front_t) if self.front_t is not None else 99.0)
        for a in st.actions:
            if a == "ARM":
                self._call("arm", self.cli_arm, True)
            elif a == "OFFBOARD":
                self._call("off", self.cli_off, True)
            elif a == "LAND":
                self._call("land", self.cli_land)
        c = st.cmd
        m = Twist()
        m.linear.x, m.linear.y, m.linear.z, m.angular.z = float(c.vx), float(c.vy), float(c.vz), float(c.yaw_rate)
        self.pub_cmd.publish(m)                                   # ENU: x East, y North, z Up, yaw rate CCW
        if st.path:
            p = Path()
            p.header.stamp, p.header.frame_id = s_to_stamp(self.t_now), "slam_map"
            for x, y in st.path:
                ps = PoseStamped()
                ps.header = p.header
                ps.pose.position.x, ps.pose.position.y = float(x), float(y)
                p.poses.append(ps)
            self.pub_path.publish(p)
        status = dict(st.status)
        status["wp_index"] = self.mc.idx
        status["mission_done"] = self.mc.done
        status["slam"] = self.slam_state
        status["cmd_enu"] = [c.vx, c.vy, c.vz, c.yaw_rate]
        status["fc"] = {"mode": self.mode, "armed": self.armed, "link_ok": self.fc_link_ok(), "failsafe": self.fc_failsafe}
        status["grid_age_s"] = round(grid_age, 2)
        status["pose_est_enu"] = None if pos is None else [round(float(v), 3) for v in (*pos, yaw)]        # SLAM estimate: x, y, z, yaw
        status["astar_path"] = [[round(float(x), 2), round(float(y), 2)] for x, y in st.path[:60]]
        status["skipped_waypoints"] = list(self.mc.skipped)
        status["end_reason"] = self.mc.end_reason
        self.pub_status.publish(String(data=json.dumps(status, default=float)))
        if self.mc.done:
            self.pub_done.publish(Bool(data=True))
        if self.csv and pos is not None:
            v = status.get("vfh", {}) or {}
            self.csv.write(f"{self.t_now:.3f},{status.get('state')},{status.get('target')},{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f},{yaw:.3f},"
                           f"{c.vx:.3f},{c.vy:.3f},{c.vz:.3f},{c.yaw_rate:.3f},{int(bool(v.get('blocked', False)))},{v.get('deviation_deg', 0.0):.1f},"
                           f"{v.get('nearest_obstacle_m') or -1:.2f},{int(bool(status.get('path_partial', False)))},{len(st.path)},{status.get('speed', 0.0):.3f}\n")
        rp = (status.get("stats") or {}).get("replans")
        if self.csv and rp is not None and rp != self._last_replans and st.path:
            self._last_replans = rp
            self.f_path.write(json.dumps({"t": self.t_now, "target": status.get("target"), "partial": status.get("path_partial"),
                                          "path": [[round(x, 2), round(y, 2)] for x, y in st.path]}) + "\n")
        for ev in self.mc.events[self.n_events:]:
            self.get_logger().info(f"mission event: {json.dumps(ev)}")
            if self.csv:
                self.f_ev.write(json.dumps(dict(ev, state=status.get("state"), kind="event")) + "\n")
        self.n_events = len(self.mc.events)
        if status.get("state") != self.last_state:
            self.last_state = status.get("state")
            self.get_logger().info(f"mission state -> {self.last_state}  target={status.get('target')}  t={self.t_now:.1f}s")
            if self.csv:
                self.f_ev.write(json.dumps({"t": self.t_now, "state": self.last_state, "target": status.get("target")}) + "\n")


def main(args=None):
    rclpy.init(args=args)
    n = NavNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
