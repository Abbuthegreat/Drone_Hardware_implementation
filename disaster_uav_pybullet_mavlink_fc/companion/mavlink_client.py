"""
Companion-computer side MAVLink client  (this is the only vehicle interface the autonomy code uses).

    simulation :  CompanionMavlink("udp:127.0.0.1:14540")            -> simulated FC (PyBullet)
    hardware   :  CompanionMavlink("serial:/dev/serial0:921600")     -> Pixhawk 6C Mini

This module must never import PyBullet or any simulator / ground-truth module (enforced by
tests/test_ground_truth_firewall.py). Conventions: all positions/velocities handed to and returned
by this class are MAVLink NED (North, East, Down) in metres, m/s; yaw is clockwise from North in
radians; yaw_rate is positive clockwise seen from above (about the Down axis).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from common.mav import open_link, mavlink
from common.fc_modes import mode_to_custom_mode, custom_mode_to_name

# SET_POSITION_TARGET_LOCAL_NED type_mask bits: a SET bit means "ignore this field"
_IGN_POS, _IGN_VEL, _IGN_ACC = 0b111, 0b111000, 0b111000000
_IGN_YAW, _IGN_YAW_RATE = 1 << 10, 1 << 11

FORCE_MAGIC = 21196.0


@dataclass
class Telemetry:
    t_recv: float                 # wall clock time of the newest of the fused messages
    time_boot_ms: int
    local_ned: np.ndarray         # [N, E, D] m
    velocity_ned: np.ndarray      # [vN, vE, vD] m/s
    lat_deg: float
    lon_deg: float
    alt_m: float                  # MSL/ellipsoid altitude from GLOBAL_POSITION_INT
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    rates_frd: np.ndarray
    mode: str
    armed: bool
    system_status: int
    battery_v: float = float("nan")
    battery_pct: int = -1


@dataclass
class ImuSampleMsg:
    t_s: float                    # FC timestamp [s]
    accel_frd: np.ndarray
    gyro_frd: np.ndarray


@dataclass
class GpsFixMsg:
    t_s: float
    fix_type: int
    lat_deg: float
    lon_deg: float
    alt_m: float
    hdop: float
    satellites: int
    h_acc_m: float
    v_acc_m: float


class CompanionMavlink:
    def __init__(self, connection="udp:127.0.0.1:14540", baud=115200, source_system=255, source_component=191,
                 log_path: Optional[str] = None, heartbeat_hz=1.0):
        self.connection, self.baud = connection, baud
        self.src_sys, self.src_comp = source_system, source_component
        self.master = None
        self.target_system, self.target_component = 1, 1
        self.last_messages = {}
        self.msg_wall_time = {}
        self.msg_counts = {}
        self.imu_queue = deque(maxlen=5000)
        self.acks = {}
        self.motor_rpm_by_index = {}
        self._ack_cv = threading.Condition()
        self._tx_lock = threading.Lock()
        self._stop = threading.Event()
        self._rx_thread = self._hb_thread = None
        self.heartbeat_hz = heartbeat_hz
        self.last_setpoint_wall = 0.0
        self.n_setpoints_sent = 0
        self.fc_autopilot = None
        self._log = open(log_path, "a", buffering=1) if log_path else None

    # ------------------------------------------------------------------ connection
    def _log_line(self, direction, name, detail=""):
        if self._log:
            self._log.write(f"{time.time():.4f} {direction} {name} {detail}\n")

    def connect(self, timeout=10.0):
        """Open the link, send our HEARTBEAT (UDP servers only learn our address from it) and wait for the FC's."""
        self.master = open_link(self.connection, self.baud, self.src_sys, self.src_comp)
        self._stop.clear()
        t_end = time.time() + timeout
        while time.time() < t_end:
            self.send_heartbeat()
            msg = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=0.25)
            if msg is not None and msg.get_srcComponent() in (1, 0, 250) and msg.type != mavlink.MAV_TYPE_ONBOARD_CONTROLLER:
                self.target_system, self.target_component = msg.get_srcSystem(), msg.get_srcComponent()
                self.fc_autopilot = msg.autopilot
                self._store(msg)
                break
        else:
            raise TimeoutError(f"no FC HEARTBEAT on {self.connection} within {timeout}s")
        self._rx_thread = threading.Thread(target=self._receive_loop, daemon=True, name="mav-rx")
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="mav-hb")
        self._rx_thread.start()
        self._hb_thread.start()
        self._log_line("--", "CONNECTED", f"sys={self.target_system} comp={self.target_component} autopilot={self.fc_autopilot}")
        return True

    def _store(self, msg):
        name = msg.get_type()
        self.last_messages[name] = msg
        self.msg_wall_time[name] = time.time()
        self.msg_counts[name] = self.msg_counts.get(name, 0) + 1

    def _receive_loop(self):
        while not self._stop.is_set() and self.master is not None:
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.2)
            except Exception:
                msg = None
                time.sleep(0.01)
            if msg is None or msg.get_type() == "BAD_DATA":
                continue
            self._store(msg)
            name = msg.get_type()
            if name == "COMMAND_ACK":
                with self._ack_cv:
                    self.acks[msg.command] = msg.result
                    self._ack_cv.notify_all()
                self._log_line("RX", "COMMAND_ACK", f"command={msg.command} result={msg.result}")
            elif name == "HIGHRES_IMU":
                self.imu_queue.append(ImuSampleMsg(msg.time_usec * 1e-6, np.array([msg.xacc, msg.yacc, msg.zacc]),
                                                   np.array([msg.xgyro, msg.ygyro, msg.zgyro])))
            elif name == "RAW_RPM":
                self.motor_rpm_by_index[int(msg.index)] = float(msg.frequency)
            elif name == "STATUSTEXT":
                self._log_line("RX", "STATUSTEXT", msg.text)

    def _heartbeat_loop(self):
        while not self._stop.is_set():
            try:
                self.send_heartbeat()
            except Exception:
                pass
            self._stop.wait(1.0 / self.heartbeat_hz)

    def send_heartbeat(self):
        with self._tx_lock:
            self.master.mav.heartbeat_send(mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavlink.MAV_AUTOPILOT_INVALID, 0, 0,
                                           mavlink.MAV_STATE_ACTIVE)

    # ------------------------------------------------------------------ setpoints
    def _send_target(self, mask, pos, vel, yaw, yaw_rate, frame=mavlink.MAV_FRAME_LOCAL_NED):
        with self._tx_lock:
            self.master.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF, self.target_system, self.target_component, frame, mask,
                float(pos[0]), float(pos[1]), float(pos[2]), float(vel[0]), float(vel[1]), float(vel[2]),
                0.0, 0.0, 0.0, float(yaw), float(yaw_rate))
        self.last_setpoint_wall = time.time()
        self.n_setpoints_sent += 1
        if self.n_setpoints_sent % 20 == 1:
            self._log_line("TX", "SET_POSITION_TARGET_LOCAL_NED", f"mask={mask:#06x} pos={tuple(np.round(pos,2))} vel={tuple(np.round(vel,2))} yaw={yaw:.2f} yr={yaw_rate:.2f} (n={self.n_setpoints_sent})")

    def send_velocity_ned(self, vn, ve, vd, yaw_rate=0.0):
        """Velocity setpoint in local NED [m/s] + yaw rate [rad/s, clockwise positive]."""
        mask = _IGN_POS | _IGN_ACC | _IGN_YAW
        self._send_target(mask, (0, 0, 0), (vn, ve, vd), 0.0, yaw_rate)

    def send_position_ned(self, north, east, down, yaw=None):
        """Position setpoint in local NED [m]; yaw [rad, 0 = North, clockwise positive] optional."""
        mask = _IGN_VEL | _IGN_ACC | _IGN_YAW_RATE | (_IGN_YAW if yaw is None else 0)
        self._send_target(mask, (north, east, down), (0, 0, 0), 0.0 if yaw is None else yaw, 0.0)

    def send_attitude_target(self, q_wxyz, thrust):
        """SET_ATTITUDE_TARGET: attitude quaternion (body FRD -> NED) and normalised thrust 0..1."""
        with self._tx_lock:
            self.master.mav.set_attitude_target_send(
                int(time.time() * 1000) & 0xFFFFFFFF, self.target_system, self.target_component,
                0b00000111, [float(x) for x in q_wxyz], 0.0, 0.0, 0.0, float(thrust))
        self.last_setpoint_wall = time.time()
        self.n_setpoints_sent += 1

    # ------------------------------------------------------------------ commands
    def command_long(self, command, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0, wait_ack=True, timeout=2.0):
        with self._ack_cv:
            self.acks.pop(command, None)
        with self._tx_lock:
            self.master.mav.command_long_send(self.target_system, self.target_component, command, 0,
                                              p1, p2, p3, p4, p5, p6, p7)
        self._log_line("TX", "COMMAND_LONG", f"command={command} p1={p1} p2={p2} p3={p3}")
        if not wait_ack:
            return None
        with self._ack_cv:
            if self._ack_cv.wait_for(lambda: command in self.acks, timeout):
                return self.acks[command]
        return None

    def arm(self, force=False, timeout=2.0):
        return self.command_long(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, FORCE_MAGIC if force else 0, timeout=timeout)

    def disarm(self, force=False, timeout=2.0):
        return self.command_long(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, FORCE_MAGIC if force else 0, timeout=timeout)

    def set_mode(self, name, timeout=2.0):
        """MAV_CMD_DO_SET_MODE with the PX4 custom-mode encoding (see common/fc_modes.py)."""
        cm = mode_to_custom_mode(name)
        return self.command_long(mavlink.MAV_CMD_DO_SET_MODE, mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                 (cm >> 16) & 0xFF, (cm >> 24) & 0xFF, timeout=timeout)

    def set_mode_message(self, name):
        """Same as set_mode() but using the SET_MODE message (no ACK is defined for it)."""
        with self._tx_lock:
            self.master.mav.set_mode_send(self.target_system, mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                          mode_to_custom_mode(name))

    def land(self, timeout=2.0):
        return self.command_long(mavlink.MAV_CMD_NAV_LAND, timeout=timeout)

    def set_message_interval(self, msg_id, hz):
        return self.command_long(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz if hz > 0 else -1)

    def start_offboard(self, warmup_s=1.0, rate_hz=20.0, timeout=2.0):
        """PX4-style: stream hold setpoints, then request OFFBOARD and arm. Returns (arm_result, mode_result)."""
        t_end = time.time() + warmup_s
        while time.time() < t_end:
            self.send_velocity_ned(0, 0, 0)
            time.sleep(1.0 / rate_hz)
        r_arm = self.arm(timeout=timeout)
        self.send_velocity_ned(0, 0, 0)
        r_mode = self.set_mode("OFFBOARD", timeout=timeout)
        return r_arm, r_mode

    # ------------------------------------------------------------------ telemetry
    def age(self, msg_name):
        t = self.msg_wall_time.get(msg_name)
        return None if t is None else time.time() - t

    def telemetry(self) -> Optional[Telemetry]:
        lp, gp, at = (self.last_messages.get(k) for k in ("LOCAL_POSITION_NED", "GLOBAL_POSITION_INT", "ATTITUDE"))
        hb, ss = self.last_messages.get("HEARTBEAT"), self.last_messages.get("SYS_STATUS")
        if lp is None or gp is None or at is None or hb is None:
            return None
        critical = hb.system_status == mavlink.MAV_STATE_CRITICAL
        return Telemetry(
            t_recv=max(self.msg_wall_time[k] for k in ("LOCAL_POSITION_NED", "GLOBAL_POSITION_INT", "ATTITUDE")),
            time_boot_ms=lp.time_boot_ms,
            local_ned=np.array([lp.x, lp.y, lp.z], float), velocity_ned=np.array([lp.vx, lp.vy, lp.vz], float),
            lat_deg=gp.lat / 1e7, lon_deg=gp.lon / 1e7, alt_m=gp.alt / 1000.0,
            roll_rad=at.roll, pitch_rad=at.pitch, yaw_rad=at.yaw,
            rates_frd=np.array([at.rollspeed, at.pitchspeed, at.yawspeed]),
            mode=custom_mode_to_name(hb.custom_mode, critical),
            armed=bool(hb.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED), system_status=hb.system_status,
            battery_v=ss.voltage_battery / 1000.0 if ss else float("nan"),
            battery_pct=ss.battery_remaining if ss else -1)

    def gps(self) -> Optional[GpsFixMsg]:
        m = self.last_messages.get("GPS_RAW_INT")
        if m is None:
            return None
        return GpsFixMsg(m.time_usec * 1e-6, m.fix_type, m.lat / 1e7, m.lon / 1e7, m.alt / 1000.0, m.eph / 100.0,
                         m.satellites_visible, m.h_acc / 1000.0, m.v_acc / 1000.0)

    def pop_imu(self):
        out = list(self.imu_queue)
        self.imu_queue.clear()
        return out

    def motor_rpm(self):
        """Latest RAW_RPM per motor as an array of 4 (None until received)."""
        if len(self.motor_rpm_by_index) < 4:
            return None
        return np.array([self.motor_rpm_by_index[i] for i in range(4)], float)

    def close(self):
        self._stop.set()
        for th in (self._rx_thread, self._hb_thread):
            if th is not None:
                th.join(timeout=1.0)
        if self.master is not None:
            self.master.close()
            self.master = None
        if self._log:
            self._log.close()


def hardware_connection(device="/dev/serial0", baud=921600):
    """Same API against a real Pixhawk over the Raspberry Pi serial port."""
    return CompanionMavlink(f"serial:{device}:{baud}", baud=baud)
