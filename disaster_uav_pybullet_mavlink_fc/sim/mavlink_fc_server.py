"""
MAVLink server for the simulated flight controller.

    companion --UDP--> [udpin:0.0.0.0:14540] MavlinkFcServer --> SimulatedFlightController --> ... --> PyBullet

This is the wire boundary. It decodes real MAVLink (common dialect, MAVLink 2) and calls the FC's
command handlers; telemetry is generated from the FC's *estimate* and the simulated sensors (never
from anything the autonomy stack could not get from a real Pixhawk).

Companion -> FC : HEARTBEAT, SET_POSITION_TARGET_LOCAL_NED, SET_ATTITUDE_TARGET, COMMAND_LONG
                  (ARM_DISARM, DO_SET_MODE, SET_MESSAGE_INTERVAL), SET_MODE
FC -> companion : HEARTBEAT, LOCAL_POSITION_NED, GLOBAL_POSITION_INT, ATTITUDE, SYS_STATUS,
                  COMMAND_ACK, STATUSTEXT, HIGHRES_IMU (IMU), GPS_RAW_INT (GPS),
                  RAW_RPM (motor rpm), ACTUATOR_OUTPUT_STATUS (normalised motor commands)

The server is polled from the simulation loop (`poll()` non-blocking, then `send_telemetry()` on a
simulation-time schedule), so everything is deterministic with respect to simulation time.
"""
from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np

from common.mav import open_link, mavlink
from common.fc_modes import (mode_to_custom_mode, custom_mode_to_name, decode_custom_mode, COMMANDABLE_MODES,
                             PX4_MAIN_OFFBOARD, PX4_MAIN_AUTO, PX4_SUB_AUTO_LOITER, PX4_SUB_AUTO_LAND)
from common.frames import euler_from_R, wrap_pi
from sim.flight_controller_sim import RES_ACCEPTED, RES_DENIED, RES_FAILED

FORCE_DISARM_MAGIC = 21196.0      # MAV_CMD_COMPONENT_ARM_DISARM param2 value that forces (dis)arm

MSG_ID = {"HEARTBEAT": 0, "SYS_STATUS": 1, "ATTITUDE": 30, "LOCAL_POSITION_NED": 32, "GLOBAL_POSITION_INT": 33,
          "GPS_RAW_INT": 24, "HIGHRES_IMU": 105, "RAW_RPM": 339, "ACTUATOR_OUTPUT_STATUS": 375}


class MavlinkFcServer:
    def __init__(self, sim, bind: str | None = None, log_path: str | None = None):
        self.sim = sim
        cfg = sim.cfg
        m = cfg["mavlink"]
        self.sys_id, self.comp_id = m["sys_id"], m["comp_id"]
        self.link = open_link(bind or m["fc_bind"], source_system=self.sys_id, source_component=self.comp_id)
        self.periods = {k: 1.0 / v for k, v in m["rates_hz"].items() if v > 0}
        self.hb_period = 1.0 / m["heartbeat_hz"]
        self.next_due = {}
        self.next_hb = 0.0
        self.rx_counts, self.tx_counts = Counter(), Counter()
        self.last_companion_hb_t = None
        self.companion_seen = False
        self.log = open(log_path, "w", buffering=1) if log_path else None
        self._imu_pending = False
        self.wall_start = time.time()
        hook = self._on_step
        sim.hooks.append(hook)

    # ------------------------------------------------------------------ logging
    def _log(self, direction, name, detail=""):
        if self.log:
            self.log.write(f"{self.sim.t:9.4f} {direction} {name} {detail}\n")

    # ------------------------------------------------------------------ RX
    def poll(self, max_msgs=64):
        t = self.sim.t
        for _ in range(max_msgs):
            try:
                msg = self.link.recv_match(blocking=False)
            except Exception:
                break
            if msg is None:
                break
            name = msg.get_type()
            if name == "BAD_DATA":
                continue
            self.rx_counts[name] += 1
            self.companion_seen = True
            h = getattr(self, "_rx_" + name.lower(), None)
            if h:
                h(msg, t)

    def _for_us(self, msg):
        ts = getattr(msg, "target_system", self.sys_id)
        return ts in (0, self.sys_id)

    def _rx_heartbeat(self, msg, t):
        if msg.type == mavlink.MAV_TYPE_ONBOARD_CONTROLLER or msg.get_srcSystem() != self.sys_id:
            self.last_companion_hb_t = t

    def _rx_set_position_target_local_ned(self, msg, t):
        if not self._for_us(msg):
            return
        ok = self.sim.fc.handle_position_target(
            t, int(msg.coordinate_frame), int(msg.type_mask), (msg.x, msg.y, msg.z), (msg.vx, msg.vy, msg.vz),
            (msg.afx, msg.afy, msg.afz), msg.yaw, msg.yaw_rate)
        self._log("RX", "SET_POSITION_TARGET_LOCAL_NED",
                  f"frame={msg.coordinate_frame} mask={msg.type_mask:#06x} pos=({msg.x:.2f},{msg.y:.2f},{msg.z:.2f}) "
                  f"vel=({msg.vx:.2f},{msg.vy:.2f},{msg.vz:.2f}) yaw={msg.yaw:.2f} yr={msg.yaw_rate:.2f} accepted={ok}")

    def _rx_set_attitude_target(self, msg, t):
        if not self._for_us(msg):
            return
        ok = self.sim.fc.handle_attitude_target(t, msg.q, msg.thrust, int(msg.type_mask))
        self._log("RX", "SET_ATTITUDE_TARGET", f"thrust={msg.thrust:.2f} mask={msg.type_mask:#04x} accepted={ok}")

    def _rx_set_mode(self, msg, t):
        if not self._for_us(msg):
            return
        res = self._apply_custom_mode(int(msg.base_mode), int(msg.custom_mode), t)
        self._log("RX", "SET_MODE", f"base={msg.base_mode} custom={msg.custom_mode:#x} -> {res}")

    def _apply_custom_mode(self, base_mode, custom_mode, t):
        if not (base_mode & mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED):
            return RES_DENIED
        main, sub = decode_custom_mode(custom_mode)
        for name in COMMANDABLE_MODES:
            if (main, sub) == decode_custom_mode(mode_to_custom_mode(name)):
                return self.sim.fc.set_mode(name, t)
        return RES_DENIED

    def _ack(self, msg, cmd, result):
        self.link.mav.command_ack_send(cmd, result, 0, 0, msg.get_srcSystem(), msg.get_srcComponent())
        self.tx_counts["COMMAND_ACK"] += 1
        self._log("TX", "COMMAND_ACK", f"command={cmd} result={result}")

    def _rx_command_long(self, msg, t):
        if not self._for_us(msg):
            return
        cmd, fc, sim = int(msg.command), self.sim.fc, self.sim
        C = mavlink
        if cmd == C.MAV_CMD_COMPONENT_ARM_DISARM:
            force = abs(msg.param2 - FORCE_DISARM_MAGIC) < 0.5
            if msg.param1 >= 0.5:
                res = fc.arm(t, battery_ok=sim.battery.arming_ok())
            else:
                res = fc.disarm(t, force=force)
        elif cmd == C.MAV_CMD_DO_SET_MODE:
            res = self._apply_custom_mode(int(msg.param1), int(msg.param2) << 16 | int(msg.param3) << 24, t)
        elif cmd == C.MAV_CMD_NAV_LAND:
            res = fc.set_mode("LAND", t)
        elif cmd == C.MAV_CMD_SET_MESSAGE_INTERVAL:
            res = self._set_interval(int(msg.param1), float(msg.param2))
        else:
            res = C.MAV_RESULT_UNSUPPORTED
        self._log("RX", "COMMAND_LONG", f"command={cmd} p1={msg.param1} p2={msg.param2} p3={msg.param3} -> {res}")
        self._ack(msg, cmd, res)

    def _set_interval(self, msg_id, interval_us):
        names = {v: k for k, v in MSG_ID.items()}
        name = names.get(msg_id)
        if name is None or name == "HEARTBEAT":
            return mavlink.MAV_RESULT_UNSUPPORTED
        key = name.lower()
        if interval_us < 0:
            self.periods.pop(key, None)
        elif interval_us == 0:
            self.periods[key] = 1.0 / self.sim.cfg["mavlink"]["rates_hz"].get(key, 10)
        else:
            self.periods[key] = interval_us * 1e-6
        return RES_ACCEPTED

    # ------------------------------------------------------------------ TX
    def _has_peer(self):
        return self.companion_seen

    def _due(self, key, t):
        per = self.periods.get(key)
        if per is None:
            return False
        due = self.next_due.get(key, 0.0)
        if t + 1e-9 < due:
            return False
        nxt = due + per
        self.next_due[key] = nxt if nxt > t else t + per
        return True

    def send_telemetry(self):
        if not self._has_peer():
            return
        sim, fc, t = self.sim, self.sim.fc, self.sim.t
        mav = self.link.mav
        est, tms = sim.estimate, int(t * 1000) & 0xFFFFFFFF
        tus = int(t * 1e6)

        if t + 1e-9 >= self.next_hb:
            self.next_hb = self.next_hb + self.hb_period if self.next_hb + self.hb_period > t else t + self.hb_period
            base = mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            if fc.armed:
                base |= mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if fc.mode == "OFFBOARD":
                base |= mavlink.MAV_MODE_FLAG_GUIDED_ENABLED
            mav.heartbeat_send(mavlink.MAV_TYPE_QUADROTOR, mavlink.MAV_AUTOPILOT_GENERIC, base,
                               mode_to_custom_mode(fc.mode), fc.system_status())
            self.tx_counts["HEARTBEAT"] += 1

        pos, vel = est.pos_ned, est.vel_ned
        roll, pitch, yaw = euler_from_R(est.R_ned_frd)
        if self._due("local_position_ned", t):
            mav.local_position_ned_send(tms, *map(float, pos), *map(float, vel))
            self.tx_counts["LOCAL_POSITION_NED"] += 1
        if self._due("attitude", t):
            mav.attitude_send(tms, roll, pitch, yaw, *map(float, est.gyro_frd))
            self.tx_counts["ATTITUDE"] += 1
        if self._due("global_position_int", t):
            lat, lon, alt = sim.geo.ned_to_geodetic(pos)
            mav.global_position_int_send(tms, int(round(lat * 1e7)), int(round(lon * 1e7)), int(alt * 1000),
                                         int(-(pos[2] - fc.home_down) * 1000), int(vel[0] * 100), int(vel[1] * 100),
                                         int(vel[2] * 100), int((math.degrees(yaw) % 360.0) * 100) % 36000)
            self.tx_counts["GLOBAL_POSITION_INT"] += 1
        if self._due("sys_status", t):
            b = sim.battery
            present = enabled = health = 0x1 | 0x2 | 0x20      # gyro | accel | GPS
            mav.sys_status_send(present, enabled, health, 0, int(b.voltage_v * 1000), int(b.current_a * 100),
                                int(round(b.soc * 100)), 0, 0, 0, 0, 0, 0)
            self.tx_counts["SYS_STATUS"] += 1
        if self._due("highres_imu", t) and sim.imu_sample is not None:
            s = sim.imu_sample
            mav.highres_imu_send(int(s.t * 1e6), *map(float, s.accel), *map(float, s.gyro), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0x3F)
            self.tx_counts["HIGHRES_IMU"] += 1
        if sim.new_gps and "gps_raw_int" in self.periods:
            g = sim.gps_sample
            gs = float(np.linalg.norm(g.vel_ned[:2]))
            cog = int((math.degrees(math.atan2(g.vel_ned[1], g.vel_ned[0])) % 360) * 100) % 36000 if gs > 0.3 else 65535
            mav.gps_raw_int_send(int(g.t * 1e6), g.fix_type, int(round(g.lat_deg * 1e7)), int(round(g.lon_deg * 1e7)),
                                 int(g.alt_m * 1000), int(g.hdop * 100), int(g.hdop * 150), int(gs * 100), cog,
                                 g.satellites, 0, int(g.eph_m * 1000), int(g.epv_m * 1000), 50, 0)
            self.tx_counts["GPS_RAW_INT"] += 1
        if self._due("raw_rpm", t):
            for i, rpm in enumerate(sim.motor.rpm()):
                mav.raw_rpm_send(i, float(rpm))
            mav.actuator_output_status_send(tus, 4, list(map(float, sim.u)) + [0.0] * 28)
            self.tx_counts["RAW_RPM"] += 4
            self.tx_counts["ACTUATOR_OUTPUT_STATUS"] += 1
        while fc.events:
            sev, text = fc.events.pop(0)
            mav.statustext_send(sev, text.encode()[:50])
            self.tx_counts["STATUSTEXT"] += 1
            self._log("TX", "STATUSTEXT", text)

    def _on_step(self, sim):
        self.poll()
        self.send_telemetry()

    def close(self):
        try:
            self.link.close()
        finally:
            if self.log:
                self.log.close()
