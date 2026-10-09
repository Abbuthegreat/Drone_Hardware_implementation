"""
Motor + ESC + propeller model (all coefficients are calibration parameters, see config `propulsion`).

    T_i = kT * w_i^2                   thrust of one rotor            [N]
    Q_i = kQ * w_i^2                   aerodynamic reaction torque    [N*m]
    P_i = Q_i * w_i = kQ * w_i^3       shaft power                    [W]
    dw/dt = (w_target - w) / tau       first-order motor + ESC lag
    w_target = w_idle + (w_max(V) - w_idle) * u^expo         u in [0,1] normalised command
    w_max(V) = w_max_nominal * V / V_nominal                  loaded speed scales with bus voltage

NOTE: 800 KV * V is only the theoretical no-load speed. It is used solely as an upper-bound sanity
check in config validation; loaded propeller RPM is the separate parameter
`max_loaded_rpm_at_nominal_v`.
"""
from __future__ import annotations

import math

import numpy as np

RPM = 2 * math.pi / 60.0


class MotorModel:
    def __init__(self, cfg: dict):
        pr = cfg["propulsion"]
        self.kT = float(pr["thrust_coeff_kT"])
        self.kQ = float(pr["torque_coeff_kQ"])
        self.tau = float(pr["motor_time_constant_s"])
        self.expo = float(pr["command_expo"])
        self.eta = float(pr["esc_motor_efficiency"])
        self.v_nom = float(cfg["battery"]["nominal_v"])
        self.w_max_nom = float(pr["max_loaded_rpm_at_nominal_v"]) * RPM
        self.w_idle = float(pr["idle_rpm"]) * RPM
        self.omega = np.zeros(4)

    # ---- static mapping (also what the FC uses as its "thrust curve" knowledge)
    def omega_max(self, v_bus: float) -> float:
        return self.w_max_nom * max(v_bus, 0.0) / self.v_nom

    def thrust_max(self, v_bus: float) -> float:
        return self.kT * self.omega_max(v_bus) ** 2

    def thrust_min(self) -> float:
        return self.kT * self.w_idle ** 2

    def omega_target(self, u, v_bus: float, armed: bool):
        if not armed:
            return np.zeros(4)
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        wm = self.omega_max(v_bus)
        return self.w_idle + (wm - self.w_idle) * u ** self.expo

    def command_for_thrust(self, thrust_n, v_bus: float):
        """Inverse map: thrust [N] -> normalised command u in [0,1] at bus voltage v_bus."""
        t = np.maximum(np.asarray(thrust_n, dtype=float), 0.0)
        w = np.sqrt(t / self.kT)
        wm = self.omega_max(v_bus)
        x = np.clip((w - self.w_idle) / max(wm - self.w_idle, 1e-6), 0.0, 1.0)
        return x ** (1.0 / self.expo)

    # ---- dynamics
    def step(self, u, v_bus: float, armed: bool, dt: float):
        wt = self.omega_target(u, v_bus, armed)
        alpha = 1.0 - math.exp(-dt / max(self.tau, 1e-4))
        self.omega += alpha * (wt - self.omega)
        self.omega = np.minimum(self.omega, self.omega_max(v_bus))   # bus-voltage saturation
        return self.outputs()

    def outputs(self):
        w2 = self.omega ** 2
        return self.kT * w2, self.kQ * w2      # thrust [N], reaction torque magnitude [N*m]

    def electrical_power(self, avionics_w: float = 0.0) -> float:
        shaft = float(np.sum(self.kQ * self.omega ** 3))
        return shaft / self.eta + avionics_w

    def rpm(self):
        return self.omega / RPM
