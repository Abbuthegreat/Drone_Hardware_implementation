"""
Quad-X motor mixer (FRD body frame, PX4 numbering).

Motor 1 : front-right  CCW      Motor 3 : front-left  CW
Motor 2 : rear-left    CCW      Motor 4 : rear-right  CW          (spin direction seen from ABOVE)

Body frame is FRD (x forward, y right, z down). Rotor i sits at (x_i, y_i) relative to the CG and
produces thrust T_i along -z_body (up). Its reaction torque on the airframe is opposite to its spin:
a CCW propeller (seen from above) pushes the body clockwise, i.e. about +z_FRD.

    total thrust  T      = sum T_i
    roll  torque  tau_x  = -sum y_i T_i        (right-hand motors up -> roll left  = negative)
    pitch torque  tau_y  =  sum x_i T_i        (front motors up      -> nose up    = positive)
    yaw   torque  tau_z  =  sum s_i (kQ/kT) T_i   s_i = +1 CCW, -1 CW  (positive = clockwise from above)

so   [T, tau_x, tau_y, tau_z]^T = A [T_1..T_4]^T   and   T_i = A^-1 [T, tau_x, tau_y, tau_z].
"""
from __future__ import annotations

import numpy as np


class QuadMixer:
    def __init__(self, rotor_pos_frd, spin, km_ratio):
        pos = np.asarray(rotor_pos_frd, float)
        s = np.asarray(spin, float)
        self.A = np.vstack([np.ones(4), -pos[:, 1], pos[:, 0], s * km_ratio])
        self.A_inv = np.linalg.inv(self.A)

    def forward(self, thrusts):
        """Motor thrusts -> [T, tau_x, tau_y, tau_z]."""
        return self.A @ np.asarray(thrusts, float)

    def mix(self, thrust, tau):
        """Unsaturated allocation."""
        return self.A_inv @ np.array([thrust, tau[0], tau[1], tau[2]], float)

    def mix_saturated(self, thrust, tau, t_min, t_max):
        """
        Allocation with saturation handling. Priority: roll/pitch torque > collective > yaw torque.
        The collective is shifted (airmode-style) to make the requested torques feasible; if they still
        do not fit, yaw is reduced first, then roll/pitch.
        The collective is applied through its own allocation vector u = A^-1 [1,0,0,0] (uniform only when the CG is exactly at the
        geometric centre), so a small CG offset does not leak a parasitic torque.
        Returns (motor_thrusts, info dict).
        """
        tau = np.asarray(tau, float)
        u = self.A_inv[:, 0]                                   # thrust per unit total collective, torque-free

        def fit(scale_rp, scale_yaw):
            t = np.array([scale_rp * tau[0], scale_rp * tau[1], scale_yaw * tau[2]])
            dist = self.A_inv @ np.array([0.0, *t])            # torque-only part (zero net collective)
            lo = np.max((t_min - dist) / u)                    # feasible range of the total collective c: t_min <= c*u_i + dist_i <= t_max
            hi = np.min((t_max - dist) / u)
            if lo <= hi + 1e-9:
                c = min(max(thrust, lo), hi)
                return c * u + dist, True, c
            return None, False, None

        for sy in (1.0, 0.5, 0.25, 0.0):
            out, ok, c = fit(1.0, sy)
            if ok:
                return out, {"saturated": sy < 1.0 or abs(c - thrust) > 1e-9, "yaw_scale": sy, "rp_scale": 1.0}
        dist = self.A_inv @ np.array([0.0, tau[0], tau[1], 0.0])
        span = (dist / u).max() - (dist / u).min()
        s = min(1.0, (t_max - t_min) / max(span, 1e-9) * 0.98)
        out, _, _ = fit(s, 0.0)
        if out is None:
            out = np.clip(thrust * u, t_min, t_max)
        return out, {"saturated": True, "yaw_scale": 0.0, "rp_scale": s}
