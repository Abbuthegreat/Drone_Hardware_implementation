"""
Simulated flight controller (NOT PX4, does not contain PX4 code).

Pure control logic: no PyBullet, no MAVLink sockets. Inputs are an estimate of the vehicle state
and the battery voltage; the output is one normalised command per motor. Everything is NED/FRD.

    MAVLink setpoint (via sim/mavlink_fc_server.py)
      -> setpoint interpretation (type_mask, frame)
      -> position P -> velocity PID -> acceleration -> thrust vector -> attitude setpoint
      -> attitude P -> rate PID -> torque + collective
      -> mixer (with saturation handling) -> per-motor thrust -> thrust->command map
      -> [motor + ESC dynamics live in sim/motor_model.py, driven by the simulation loop]

Flight modes: HOLD, OFFBOARD, LAND, FAILSAFE_HOLD (not directly commandable).
Watchdog: in OFFBOARD, if no valid setpoint arrives for `offboard_timeout_s` -> FAILSAFE_HOLD
(brake to a stop and hold position); after `failsafe_land_after_s` more seconds -> LAND.
This timeout is a parameter of THIS simulator, not a PX4 parameter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from common.frames import (R_from_quat, euler_from_R, wrap_pi, quat_xyzw_to_wxyz)
from sim.config import inertia_matrix
from sim.controllers import (VelocityController, AttitudeController, RateController,
                             position_to_velocity, thrust_vector_to_attitude)
from sim.mixer import QuadMixer
from sim.motor_model import MotorModel

# MAVLink constants used here (values from the common message set)
FRAME_LOCAL_NED, FRAME_LOCAL_OFFSET_NED, FRAME_BODY_NED, FRAME_BODY_OFFSET_NED = 1, 7, 8, 9
IGN_POS, IGN_VEL, IGN_ACC = 0b111, 0b111000, 0b111000000
IGN_YAW, IGN_YAW_RATE = 1 << 10, 1 << 11
ATT_IGN_ATTITUDE = 1 << 7

RES_ACCEPTED, RES_DENIED, RES_FAILED = 0, 2, 4


@dataclass
class Estimate:
    """What the FC 'knows' about the vehicle (truth + configured errors; gyro is the simulated IMU)."""
    pos_ned: np.ndarray
    vel_ned: np.ndarray
    R_ned_frd: np.ndarray
    gyro_frd: np.ndarray


@dataclass
class Setpoint:
    pos: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None
    acc: Optional[np.ndarray] = None
    yaw: Optional[float] = None
    yaw_rate: Optional[float] = None
    att_R: Optional[np.ndarray] = None
    att_thrust: float = 0.0
    t_rx: float = 0.0


class SimulatedFlightController:
    def __init__(self, cfg: dict, mixer: QuadMixer, motor: MotorModel):
        self.cfg = cfg
        fc = cfg["flight_controller"]
        lim = fc["limits"]
        self.fc = fc
        self.mass = float(cfg["vehicle"]["mass_kg"])
        self.g = float(cfg["vehicle"]["gravity_mps2"])
        self.mixer, self.motor = mixer, motor
        self.max_tilt = math.radians(lim["max_tilt_deg"])
        self.lim = lim
        J = np.diag(inertia_matrix(cfg)) * fc["model_inertia_scale"]
        v, r, a = fc["velocity"], fc["rate"], fc["attitude"]
        self.pos_kp = np.array(fc["position"]["kp"], float)
        self.vel_ctrl = VelocityController(v["kp"], v["ki"], v["kd"], v["i_limit"], lim["max_accel_xy_mps2"],
                                           0.8 * self.g, 0.9 * self.g)
        self.att_ctrl = AttitudeController(a["kp"], lim["max_rate_rps"])
        self.rate_ctrl = RateController(r["kp"], r["ki"], r["kd"], r["i_limit"], J, r["dterm_lpf_hz"])
        self.timeout = float(fc["offboard_timeout_s"])

        self.armed = False
        self.mode = "HOLD"
        self.failsafe_reason = ""
        self.failsafe_since = None
        self.sp = Setpoint()
        self.last_sp_t = -1e9
        self.n_sp_rx = 0
        self.n_sp_rejected = 0
        self.yaw_sp = 0.0
        self.hold_pos = None
        self.landed = True
        self._land_timer = 0.0
        self.est: Optional[Estimate] = None
        self.events: list = []                    # (severity, text) drained by the MAVLink server
        self.last_thrusts = np.zeros(4)
        self.last_u = np.zeros(4)
        self.last_info = {}
        self.home_down = 0.0

    # ------------------------------------------------------------ commands (called by the MAVLink layer)
    def _event(self, sev, text):
        self.events.append((sev, text))

    def arm(self, t, battery_ok=True):
        if self.armed:
            return RES_ACCEPTED
        if not battery_ok:
            self._event(4, "Arming denied: battery")
            return RES_DENIED
        if self.est is None:
            return RES_DENIED
        self.armed = True
        self.landed = float(np.linalg.norm(self.est.vel_ned)) < self.fc["landed"]["vel_threshold_mps"]
        self._land_timer = 0.0
        self.yaw_sp = euler_from_R(self.est.R_ned_frd)[2]
        self.hold_pos = None
        self._reset_loops()
        self.home_down = float(self.est.pos_ned[2])
        self._event(6, "Armed")
        return RES_ACCEPTED

    def disarm(self, t, force=False):
        if not self.armed:
            return RES_ACCEPTED
        if not self.landed and not force:
            self._event(4, "Disarm denied: in flight")
            return RES_DENIED
        self.armed = False
        self._reset_loops()
        self._event(6, "Disarmed")
        return RES_ACCEPTED

    def set_mode(self, name, t):
        if name == "OFFBOARD":
            if self.fc["offboard_entry_requires_stream"] and (t - self.last_sp_t) > self.timeout:
                self._event(4, "Offboard denied: no setpoint stream")
                return RES_DENIED
            self.mode = "OFFBOARD"
            self.failsafe_reason, self.failsafe_since = "", None
            if self.est is not None:
                self.yaw_sp = euler_from_R(self.est.R_ned_frd)[2] if self.sp.yaw is None else self.yaw_sp
        elif name in ("HOLD", "LAND"):
            self.mode = name
            self.failsafe_reason, self.failsafe_since = "", None
            self.hold_pos = None
        else:
            return RES_DENIED
        self._event(6, f"Mode {self.mode}")
        return RES_ACCEPTED

    def handle_position_target(self, t, frame, mask, pos, vel, acc, yaw, yaw_rate):
        """SET_POSITION_TARGET_LOCAL_NED. Returns True if the setpoint was accepted."""
        est = self.est
        if frame not in (FRAME_LOCAL_NED, FRAME_LOCAL_OFFSET_NED, FRAME_BODY_NED, FRAME_BODY_OFFSET_NED) or est is None:
            self.n_sp_rejected += 1
            return False
        use_pos, use_vel, use_acc = not (mask & IGN_POS), not (mask & IGN_VEL), not (mask & IGN_ACC)
        use_yaw, use_yr = not (mask & IGN_YAW), not (mask & IGN_YAW_RATE)
        vals = [np.asarray(x, float) for x in (pos, vel, acc)]
        if not all(np.all(np.isfinite(v)) for v in vals) or not (use_pos or use_vel or use_acc):
            self.n_sp_rejected += 1
            return False
        cur_yaw = euler_from_R(est.R_ned_frd)[2]
        c, s = math.cos(cur_yaw), math.sin(cur_yaw)
        Ryaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        body = frame in (FRAME_BODY_NED, FRAME_BODY_OFFSET_NED)
        if body:                                  # vectors are in a yaw-aligned frame (x fwd, y right, z down)
            vals = [Ryaw @ v for v in vals]
        p_ned = vals[0]
        if frame in (FRAME_LOCAL_OFFSET_NED, FRAME_BODY_OFFSET_NED):
            p_ned = est.pos_ned + vals[0]
        sp = Setpoint(pos=p_ned if use_pos else None, vel=vals[1] if use_vel else None,
                      acc=vals[2] if use_acc else None, t_rx=t)
        if use_yaw:
            sp.yaw = wrap_pi(float(yaw) + (cur_yaw if body else 0.0))
        if use_yr:
            sp.yaw_rate = float(np.clip(yaw_rate, -self.lim["max_yaw_rate_rps"], self.lim["max_yaw_rate_rps"]))
        self.sp, self.last_sp_t = sp, t
        self.n_sp_rx += 1
        return True

    def handle_attitude_target(self, t, q_wxyz, thrust, mask):
        q = np.asarray(q_wxyz, float)
        if mask & ATT_IGN_ATTITUDE or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
            self.n_sp_rejected += 1
            return False
        self.sp = Setpoint(att_R=R_from_quat(q), att_thrust=float(np.clip(thrust, 0.0, 1.0)), t_rx=t)
        self.last_sp_t = t
        self.n_sp_rx += 1
        return True

    # ------------------------------------------------------------ internals
    def _reset_loops(self):
        self.vel_ctrl.reset()
        self.rate_ctrl.reset()

    def _failsafe(self, reason):
        if self.mode != "FAILSAFE_HOLD":
            self.mode = "FAILSAFE_HOLD"
            self.failsafe_reason = reason
            self.failsafe_since = None            # set on next update (needs time)
            self.hold_pos = None
            self._event(3, f"Failsafe: {reason}")

    def _watchdogs(self, t, batt_low):
        if not self.armed:
            return
        if self.mode == "OFFBOARD" and (t - self.last_sp_t) > self.timeout:
            self._failsafe("OFFBOARD_SETPOINT_LOSS")
        if batt_low and self.mode != "LAND":
            self.mode = "LAND"
            self.failsafe_reason = "LOW_BATTERY"
            self.hold_pos = None
            self._event(2, "Failsafe: low battery -> LAND")
        if self.mode == "FAILSAFE_HOLD":
            if self.failsafe_since is None:
                self.failsafe_since = t
            la = self.fc["failsafe_land_after_s"]
            if la > 0 and (t - self.failsafe_since) > la:
                self.mode = "LAND"
                self.hold_pos = None
                self._event(3, "Failsafe: no setpoints -> LAND")

    def _velocity_setpoint(self, est, dt):
        """Returns (vel_sp, acc_ff, position_mode_flag) for the current mode; may set self.yaw_sp."""
        lim = self.lim
        pos, vel = est.pos_ned, est.vel_ned
        acc_ff = np.zeros(3)
        if self.mode == "OFFBOARD":
            sp = self.sp
            if sp.yaw is not None:
                self.yaw_sp = sp.yaw
            elif sp.yaw_rate is not None:
                self.yaw_sp = wrap_pi(self.yaw_sp + sp.yaw_rate * dt)
            if sp.pos is not None:
                vff = sp.vel if sp.vel is not None else np.zeros(3)
                v = position_to_velocity(sp.pos, pos, self.pos_kp, vff, lim["max_vel_xy_mps"], lim["max_climb_mps"], lim["max_descend_mps"])
                return v, acc_ff, True
            if sp.vel is not None:
                v = sp.vel.copy()
                vxy = np.linalg.norm(v[:2])
                if vxy > lim["max_vel_xy_mps"]:
                    v[:2] *= lim["max_vel_xy_mps"] / vxy
                v[2] = min(max(v[2], -lim["max_climb_mps"]), lim["max_descend_mps"])
                if sp.acc is not None:
                    acc_ff = sp.acc
                return v, acc_ff, False
            return np.zeros(3), acc_ff, False
        # HOLD / FAILSAFE_HOLD / LAND: brake to a stop, then latch and hold position.
        if self.hold_pos is None and float(np.linalg.norm(vel)) < 0.25:
            self.hold_pos = pos.copy()
        if self.mode == "LAND":
            v = np.array([0.0, 0.0, lim["land_speed_mps"]])
            if self.hold_pos is not None:
                v[:2] = position_to_velocity(self.hold_pos, pos, self.pos_kp, np.zeros(3), 1.0, 1.0, 1.0)[:2]
            return v, acc_ff, False
        if self.hold_pos is None:
            return np.zeros(3), acc_ff, False
        v = position_to_velocity(self.hold_pos, pos, self.pos_kp, np.zeros(3), lim["max_vel_xy_mps"],
                                 lim["max_climb_mps"], lim["max_descend_mps"])
        return v, acc_ff, True

    # ------------------------------------------------------------ main update
    def update(self, t, dt, est: Estimate, v_bus: float, batt_low: bool = False):
        """Returns per-motor normalised command u (0..1). Motors idle/off when disarmed."""
        self.est = est
        self._watchdogs(t, batt_low)
        if not self.armed:
            self.last_thrusts[:] = 0
            self.last_u[:] = 0
            return self.last_u

        mm = self.motor
        v_comp = v_bus if self.cfg["propulsion"]["voltage_compensation"] else mm.v_nom
        t_min, t_max = mm.thrust_min(), 0.97 * mm.thrust_max(v_comp)
        R = est.R_ned_frd
        weight = self.mass * self.g

        attitude_passthrough = self.mode == "OFFBOARD" and self.sp.att_R is not None
        if attitude_passthrough:
            R_sp, T_des = self.sp.att_R, self.sp.att_thrust * 4 * mm.thrust_max(v_comp)
            self.vel_ctrl.reset()
            yaw_sp = euler_from_R(R_sp)[2]
            F = None
        else:
            v_sp, acc_ff, _ = self._velocity_setpoint(est, dt)
            acc_sp = self.vel_ctrl.update(v_sp, est.vel_ned, acc_ff, dt, freeze_integrator=self.landed)
            if self.mode == "OFFBOARD" and self.sp.acc is not None and self.sp.vel is None and self.sp.pos is None:
                acc_sp = self.sp.acc.copy()
            if self.landed:
                acc_sp[:2] = 0.0
            R_sp, T_des, F = thrust_vector_to_attitude(acc_sp, self.yaw_sp, self.mass, self.g, self.max_tilt, t_min, t_max)
            yaw_sp = self.yaw_sp
            # thrust actually needed along the *current* body axis
            T_des = float(np.clip(np.dot(F, -R[:, 2]), 4 * t_min, 4 * t_max))

        # ---- landed detection / takeoff
        vel_small = float(np.linalg.norm(est.vel_ned)) < self.fc["landed"]["vel_threshold_mps"]
        if self.landed:
            # take-off only from an explicit OFFBOARD climb demand (never from a plain hold on the ground)
            if self.mode == "OFFBOARD" and T_des > 1.05 * weight:
                self.landed = False
                self._reset_loops()
        else:
            if vel_small and T_des < 0.7 * weight:
                self._land_timer += dt
                if self._land_timer > self.fc["landed"]["land_detect_time_s"]:
                    self.landed = True
                    self._land_timer = 0.0
            else:
                self._land_timer = 0.0
        if self.landed and self.mode == "LAND":
            self._land_timer += dt
            if self._land_timer > 2.0 * self.fc["landed"]["land_detect_time_s"]:
                self.disarm(t)
                return self.last_u * 0
        if self.landed:
            self.vel_ctrl.reset()
            self.rate_ctrl.reset()
            thrusts = np.full(4, t_min)
            info = {"landed": True}
        else:
            yr_ff = 0.0
            if self.mode == "OFFBOARD" and self.sp.yaw_rate is not None:
                yr_ff = self.sp.yaw_rate
            rate_sp = self.att_ctrl.update(R, R_sp, yaw_sp, yr_ff)
            tau = self.rate_ctrl.update(rate_sp, est.gyro_frd, dt)
            thrusts, info = self.mixer.mix_saturated(T_des, tau, t_min, t_max)
            info["landed"] = False
        self.last_thrusts = thrusts
        self.last_u = self.motor.command_for_thrust(thrusts, v_comp)
        self.last_info = info
        return self.last_u

    # ------------------------------------------------------------ status
    def system_status(self):
        """MAV_STATE: 3 STANDBY, 4 ACTIVE, 5 CRITICAL."""
        if not self.armed:
            return 3
        return 5 if self.failsafe_reason else 4
