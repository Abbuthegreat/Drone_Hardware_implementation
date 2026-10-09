"""
Cascaded controllers of the simulated FC. Everything is NED / FRD (like a real FC).

    position P  ->  velocity PID  ->  acceleration  ->  thrust vector -> attitude setpoint + collective
    attitude P (tilt + yaw)  ->  body-rate setpoint  ->  rate PID (torque = J * demand)  ->  mixer
"""
from __future__ import annotations

import math

import numpy as np

from common.frames import wrap_pi, euler_from_R


class LowPass:
    def __init__(self, cutoff_hz, n=3):
        self.fc = cutoff_hz
        self.y = None
        self.n = n

    def reset(self, x=None):
        self.y = None if x is None else np.array(x, float)

    def update(self, x, dt):
        x = np.asarray(x, float)
        if self.y is None or self.fc <= 0:
            self.y = x.copy()
            return self.y
        a = dt / (dt + 1.0 / (2 * math.pi * self.fc))
        self.y = self.y + a * (x - self.y)
        return self.y


class VelocityController:
    """PI(D) velocity loop, output = acceleration setpoint in NED [m/s^2] (gravity NOT included)."""

    def __init__(self, kp, ki, kd, i_limit, max_acc_xy, max_acc_up, max_acc_down):
        self.kp, self.ki, self.kd = map(lambda a: np.array(a, float), (kp, ki, kd))
        self.i_lim = np.array(i_limit, float)
        self.max_acc_xy, self.max_acc_up, self.max_acc_down = max_acc_xy, max_acc_up, max_acc_down
        self.integ = np.zeros(3)
        self.prev_vel = None
        self._vf = LowPass(15.0)

    def reset(self):
        self.integ[:] = 0.0
        self.prev_vel = None
        self._vf.reset()

    def update(self, vel_sp, vel, acc_ff, dt, freeze_integrator=False):
        err = vel_sp - vel
        d = np.zeros(3)
        vf = self._vf.update(vel, dt)
        if self.prev_vel is not None and dt > 0:
            d = -self.kd * (vf - self.prev_vel) / dt
        self.prev_vel = vf.copy()
        acc = self.kp * err + self.integ + d + acc_ff
        # saturate horizontally (vector) and vertically
        axy = np.linalg.norm(acc[:2])
        sat = False
        if axy > self.max_acc_xy:
            acc[:2] *= self.max_acc_xy / axy
            sat = True
        lo, hi = -self.max_acc_up, self.max_acc_down     # NED z: negative = up
        if acc[2] < lo or acc[2] > hi:
            acc[2] = min(max(acc[2], lo), hi)
            sat = True
        if not freeze_integrator and not sat:
            self.integ = np.clip(self.integ + self.ki * err * dt, -self.i_lim, self.i_lim)
        return acc


def position_to_velocity(pos_sp, pos, kp, vel_ff, max_xy, max_up, max_down):
    v = np.asarray(kp, float) * (pos_sp - pos) + vel_ff
    vxy = np.linalg.norm(v[:2])
    if vxy > max_xy:
        v[:2] *= max_xy / vxy
    v[2] = min(max(v[2], -max_up), max_down)
    return v


def thrust_vector_to_attitude(acc_sp, yaw_sp, mass, g, max_tilt_rad, t_min, t_max):
    """
    Desired specific force F = m (a - g_vec), g_vec = (0,0,+g) NED. Body -z axis points along F.
    Returns (R_sp body->NED, thrust_magnitude_N).
    """
    F = mass * (np.asarray(acc_sp, float) - np.array([0.0, 0.0, g]))
    # limit tilt while keeping the vertical component (up = -z)
    up = max(-F[2], 1e-3)
    fh = np.linalg.norm(F[:2])
    if fh > up * math.tan(max_tilt_rad):
        F[:2] *= up * math.tan(max_tilt_rad) / fh
    T = float(np.linalg.norm(F))
    if T < 1e-6:
        F = np.array([0.0, 0.0, -1.0]); T = 1e-6
    z_b = -F / np.linalg.norm(F)                       # body z (down) axis in NED
    x_c = np.array([math.cos(yaw_sp), math.sin(yaw_sp), 0.0])
    y_b = np.cross(z_b, x_c)
    ny = np.linalg.norm(y_b)
    y_b = y_b / ny if ny > 1e-6 else np.array([-math.sin(yaw_sp), math.cos(yaw_sp), 0.0])
    x_b = np.cross(y_b, z_b)
    R_sp = np.column_stack([x_b, y_b, z_b])
    return R_sp, min(max(T, t_min * 4), t_max * 4), F


class AttitudeController:
    def __init__(self, kp, max_rate):
        self.kp = np.array(kp, float)
        self.max_rate = np.array(max_rate, float)

    def update(self, R, R_sp, yaw_sp, yaw_rate_ff=0.0):
        z_b, z_sp = R[:, 2], R_sp[:, 2]
        e_tilt_body = R.T @ np.cross(z_b, z_sp)             # rotation taking body z onto the desired z, in body axes
        yaw = euler_from_R(R)[2]
        e_yaw = wrap_pi(yaw_sp - yaw)
        rate = np.array([self.kp[0] * e_tilt_body[0], self.kp[1] * e_tilt_body[1],
                         self.kp[2] * e_yaw * max(R[2, 2], 0.0) + yaw_rate_ff * R[2, 2]])
        return np.clip(rate, -self.max_rate, self.max_rate)


class RateController:
    """Body-rate PID. Output is a torque: tau = J*(angular-acceleration demand) + w x (J w)."""

    def __init__(self, kp, ki, kd, i_limit, J_diag, dterm_lpf_hz):
        self.kp, self.ki, self.kd = map(lambda a: np.array(a, float), (kp, ki, kd))
        self.i_lim = np.array(i_limit, float)
        self.J = np.array(J_diag, float)
        self.integ = np.zeros(3)
        self.prev = None
        self._d = LowPass(dterm_lpf_hz)

    def reset(self):
        self.integ[:] = 0
        self.prev = None
        self._d.reset()

    def update(self, rate_sp, rate, dt, freeze_integrator=False):
        err = rate_sp - rate
        alpha_est = np.zeros(3)
        if self.prev is not None and dt > 0:
            alpha_est = self._d.update((rate - self.prev) / dt, dt)
        self.prev = rate.copy()
        if not freeze_integrator:
            self.integ = np.clip(self.integ + self.ki * err * dt, -self.i_lim, self.i_lim)
        demand = self.kp * err + self.integ - self.kd * alpha_est
        tau = self.J * demand + np.cross(rate, self.J * rate)
        return tau
