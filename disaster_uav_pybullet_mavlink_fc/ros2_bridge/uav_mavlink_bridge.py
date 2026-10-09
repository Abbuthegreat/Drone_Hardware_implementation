"""
ROS 2  <->  MAVLink bridge (companion side).

    ROS 2 /uav/cmd_vel  --validate-->  ENU->NED  -->  SET_POSITION_TARGET_LOCAL_NED  --UDP/serial-->  FC
    FC  --MAVLink telemetry-->  /uav/telemetry, /uav/odom, /uav/imu, /uav/gps/fix, /uav/mode, /uav/armed

The bridge only knows MAVLink (through companion.mavlink_client). It has no idea whether the FC is the
PyBullet simulator (udp:127.0.0.1:14540) or a Pixhawk (serial:/dev/serial0:921600). It must not import
PyBullet, `sim`, `worlds` or `eval`.

/uav/cmd_vel  (geometry_msgs/Twist)  -- FRAME IS EXPLICIT, chosen by parameter `cmd_frame`:
    enu      (default, REP-103 world frame)  linear.x = East [m/s], linear.y = North [m/s], linear.z = Up [m/s],
                                             angular.z = yaw rate, counter-clockwise positive seen from above [rad/s]
    ned      linear.x = North, linear.y = East, linear.z = Down, angular.z = yaw rate clockwise positive
    body_flu linear.x = forward, linear.y = left, linear.z = up (relative to the CURRENT heading),
             angular.z = yaw rate counter-clockwise positive
Whatever the input frame, the MAVLink message is SET_POSITION_TARGET_LOCAL_NED in MAV_FRAME_LOCAL_NED
with the velocity fields and yaw-rate set (yaw rate clockwise positive, as MAVLink defines it).

/uav/cmd_pos (geometry_msgs/PoseStamped, ENU): position setpoint [m] + yaw (ENU, CCW from East).

Setpoints are streamed continuously at `setpoint_rate_hz` (an Offboard-style link needs a steady stream).
If no /uav/cmd_vel arrives for `cmd_timeout_s`, the bridge keeps streaming a ZERO velocity setpoint
(so the FC brakes and holds) - use the `/uav/setpoint_stream` service (SetBool false) to stop
streaming altogether and exercise the FC's setpoint-loss failsafe.

Services (std_srvs): /uav/arm (SetBool: true=arm, false=disarm), /uav/force_disarm (Trigger),
/uav/offboard (SetBool: true=OFFBOARD, false=HOLD), /uav/land (Trigger), /uav/setpoint_stream (SetBool).

/uav/telemetry is std_msgs/String containing JSON (documented in README) so no custom message package
has to be built; typed standard topics are published alongside it.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import String, Bool
from std_srvs.srv import SetBool, Trigger

from common.frames import (R_from_euler, R_enu_flu_from_ned_frd, quat_from_R, frd_to_flu, ned_to_enu, wrap_pi,
                           ros_twist_to_ned)
from companion.mavlink_client import CompanionMavlink


def _stamp(t_s: float):
    """builtin_interfaces/Time from a flight-controller timestamp (simulation time) - all sensors share this time base."""
    from builtin_interfaces.msg import Time
    m = Time()
    m.sec, m.nanosec = int(t_s), int((t_s % 1.0) * 1e9)
    return m


class UavMavlinkBridge(Node):
    def __init__(self):
        super().__init__("uav_mavlink_bridge")
        d = self.declare_parameter
        self.connection = d("connection", "udp:127.0.0.1:14540").value
        self.cmd_frame = d("cmd_frame", "enu").value
        self.max_xy = float(d("max_speed_xy", 5.0).value)
        self.max_z = float(d("max_speed_z", 2.0).value)
        self.max_yaw_rate = float(d("max_yaw_rate", 1.5).value)
        self.cmd_timeout = float(d("cmd_timeout_s", 0.5).value)
        self.rate = float(d("setpoint_rate_hz", 20.0).value)
        self.log_path = d("log_path", "").value or None
        self.auto_start = bool(d("auto_stream", True).value)
        self.link_timeout = float(d("link_timeout_s", 1.5).value)       # HEARTBEAT / telemetry older than this -> link reported down
        if self.cmd_frame not in ("enu", "ned", "body_flu"):
            raise ValueError("cmd_frame must be enu|ned|body_flu")

        self.mav = CompanionMavlink(self.connection, log_path=self.log_path)
        self.get_logger().info(f"connecting MAVLink {self.connection} ...")
        self.mav.connect(timeout=30.0)
        self.get_logger().info(f"FC heartbeat received (sys {self.mav.target_system})")

        self.last_cmd = None            # (vn, ve, vd, yr)
        self.last_cmd_time = 0.0
        self.pos_cmd = None
        self.streaming = self.auto_start
        self.n_rejected = 0
        self.n_cmds = 0

        self.create_subscription(Twist, "/uav/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(PoseStamped, "/uav/cmd_pos", self.on_cmd_pos, 10)
        self.pub_tel = self.create_publisher(String, "/uav/telemetry", 10)
        self.pub_odom = self.create_publisher(Odometry, "/uav/odom", 10)
        self.pub_imu = self.create_publisher(Imu, "/uav/imu", 200)
        self.pub_gps = self.create_publisher(NavSatFix, "/uav/gps/fix", 10)
        self.pub_mode = self.create_publisher(String, "/uav/mode", 10)
        self.pub_armed = self.create_publisher(Bool, "/uav/armed", 10)
        self.create_service(SetBool, "/uav/arm", self.srv_arm)
        self.create_service(Trigger, "/uav/force_disarm", self.srv_force_disarm)
        self.create_service(SetBool, "/uav/offboard", self.srv_offboard)
        self.create_service(Trigger, "/uav/land", self.srv_land)
        self.create_service(SetBool, "/uav/setpoint_stream", self.srv_stream)
        self.create_timer(1.0 / self.rate, self.stream_setpoint)
        self.create_timer(1.0 / 30.0, self.publish_telemetry)
        self.create_timer(1.0 / 100.0, self.publish_imu)
        self.create_timer(1.0 / 10.0, self.publish_gps)
        self._gps_last_t = -1.0

    # ------------------------------------------------------------------ ROS -> MAVLink
    def _yaw_now(self):
        t = self.mav.telemetry()
        return 0.0 if t is None else t.yaw_rad

    def on_cmd_vel(self, msg: Twist):
        vals = (msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z)
        if not all(math.isfinite(v) for v in vals):
            self.n_rejected += 1
            self.get_logger().warn("cmd_vel rejected: non-finite value")
            return
        vn, ve, vd, yr = ros_twist_to_ned(self.cmd_frame, vals[:3], vals[3], self._yaw_now())
        vxy = math.hypot(vn, ve)
        if vxy > self.max_xy:
            vn, ve = vn * self.max_xy / vxy, ve * self.max_xy / vxy
        vd = max(-self.max_z, min(self.max_z, vd))
        yr = max(-self.max_yaw_rate, min(self.max_yaw_rate, yr))
        self.last_cmd, self.last_cmd_time, self.pos_cmd = (vn, ve, vd, yr), time.time(), None
        self.n_cmds += 1
        if self.n_cmds % 20 == 1:
            self.get_logger().info(f"cmd_vel[{self.cmd_frame}] -> MAVLink NED vel=({vn:.2f},{ve:.2f},{vd:.2f}) yaw_rate={yr:.2f}")

    def on_cmd_pos(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)):
            self.n_rejected += 1
            return
        yaw_enu = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        yaw_ned = wrap_pi(math.pi / 2 - yaw_enu)
        self.pos_cmd = (p.y, p.x, -p.z, yaw_ned)          # N, E, D, yaw
        self.last_cmd_time = time.time()
        self.last_cmd = None

    def stream_setpoint(self):
        if not self.streaming:
            return
        fresh = (time.time() - self.last_cmd_time) < self.cmd_timeout
        if self.pos_cmd is not None and fresh:
            self.mav.send_position_ned(*self.pos_cmd[:3], yaw=self.pos_cmd[3])
        elif self.last_cmd is not None and fresh:
            self.mav.send_velocity_ned(*self.last_cmd)
        else:
            self.mav.send_velocity_ned(0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ services
    @staticmethod
    def _ok(res):
        return res == 0

    def srv_arm(self, req, resp):
        r = self.mav.arm() if req.data else self.mav.disarm()
        resp.success, resp.message = self._ok(r), f"MAV_RESULT={r}"
        return resp

    def srv_force_disarm(self, req, resp):
        r = self.mav.disarm(force=True)
        resp.success, resp.message = self._ok(r), f"MAV_RESULT={r}"
        return resp

    def srv_offboard(self, req, resp):
        r = self.mav.set_mode("OFFBOARD" if req.data else "HOLD")
        resp.success, resp.message = self._ok(r), f"MAV_RESULT={r}"
        return resp

    def srv_land(self, req, resp):
        r = self.mav.land()
        resp.success, resp.message = self._ok(r), f"MAV_RESULT={r}"
        return resp

    def srv_stream(self, req, resp):
        self.streaming = bool(req.data)
        resp.success, resp.message = True, f"setpoint streaming {'ON' if self.streaming else 'OFF'}"
        self.get_logger().warn(resp.message)
        return resp

    # ------------------------------------------------------------------ MAVLink -> ROS
    def publish_telemetry(self):
        t = self.mav.telemetry()
        if t is None:
            return
        now = self.get_clock().now().to_msg()
        R_enu_flu = R_enu_flu_from_ned_frd(R_from_euler(t.roll_rad, t.pitch_rad, t.yaw_rad))
        qw, qx, qy, qz = quat_from_R(R_enu_flu)
        pos_enu, vel_enu = ned_to_enu(t.local_ned), ned_to_enu(t.velocity_ned)
        w_flu = frd_to_flu(t.rates_frd)
        # link health from the high-rate streams (ATTITUDE 50 Hz / LOCAL_POSITION_NED 30 Hz): HEARTBEAT is only 2 Hz in FC (simulation)
        # time, and the simulator may run several times slower than wall clock, so a wall-clock heartbeat age would give false alarms
        hb_age = self.mav.age("HEARTBEAT")
        tel_age = time.time() - t.t_recv
        link_ok = tel_age < self.link_timeout
        doc = {
            "stamp": now.sec + now.nanosec * 1e-9, "fc_time_boot_ms": t.time_boot_ms,
            "link": {"ok": link_ok, "heartbeat_age_s": hb_age, "telemetry_age_s": tel_age},
            "failsafe": t.mode == "FAILSAFE_HOLD" or t.system_status == 5,          # MAV_STATE_CRITICAL
            "frame": "ENU (x East, y North, z Up), body FLU, origin = simulation/map origin",
            "position_enu": pos_enu.tolist(), "velocity_enu": vel_enu.tolist(),
            "orientation_xyzw_enu_flu": [qx, qy, qz, qw],
            "position_ned": t.local_ned.tolist(), "velocity_ned": t.velocity_ned.tolist(),
            "roll_pitch_yaw_ned_frd": [t.roll_rad, t.pitch_rad, t.yaw_rad],
            "gps": {"lat": t.lat_deg, "lon": t.lon_deg, "alt": t.alt_m},
            "mode": t.mode, "armed": t.armed, "system_status": t.system_status,
            "battery_v": t.battery_v, "battery_pct": t.battery_pct,
            "motor_rpm": None if self.mav.motor_rpm() is None else self.mav.motor_rpm().tolist(),
            "setpoints_sent": self.mav.n_setpoints_sent, "cmd_rejected": self.n_rejected,
        }
        self.pub_tel.publish(String(data=json.dumps(doc)))
        self.pub_mode.publish(String(data=t.mode))
        self.pub_armed.publish(Bool(data=t.armed))
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = _stamp(t.time_boot_ms / 1000.0), "map", "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, pos_enu)
        (o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z,
         o.pose.pose.orientation.w) = map(float, (qx, qy, qz, qw))
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = map(float, vel_enu)   # world-frame velocity
        o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = map(float, w_flu)
        self.pub_odom.publish(o)

    def publish_imu(self):
        for s in self.mav.pop_imu():
            m = Imu()
            m.header.stamp = _stamp(s.t_s)
            m.header.frame_id = "flight_controller_link"
            a, g = frd_to_flu(s.accel_frd), frd_to_flu(s.gyro_frd)       # REP-103: FLU
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = map(float, a)
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = map(float, g)
            m.orientation_covariance[0] = -1.0                            # no orientation estimate in this message
            self.pub_imu.publish(m)

    def publish_gps(self):
        g = self.mav.gps()
        if g is None or g.t_s == self._gps_last_t:
            return
        self._gps_last_t = g.t_s
        m = NavSatFix()
        m.header.stamp, m.header.frame_id = _stamp(g.t_s), "gps_link"
        m.status.status = NavSatStatus.STATUS_FIX if g.fix_type >= 3 else NavSatStatus.STATUS_NO_FIX
        m.status.service = NavSatStatus.SERVICE_GPS
        m.latitude, m.longitude, m.altitude = g.lat_deg, g.lon_deg, g.alt_m
        m.position_covariance = [g.h_acc_m ** 2, 0.0, 0.0, 0.0, g.h_acc_m ** 2, 0.0, 0.0, 0.0, g.v_acc_m ** 2]
        m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        self.pub_gps.publish(m)

    def destroy_node(self):
        try:
            self.mav.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UavMavlinkBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
