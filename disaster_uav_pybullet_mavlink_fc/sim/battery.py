"""
Simple engineering battery model (not electrochemical).

    V_oc(SOC)  = cells * interp(SOC, ocv_soc_points, ocv_cell_v_points)
    P_elec     = V_oc*I - R*I^2   ->  I = (V_oc - sqrt(V_oc^2 - 4 R P)) / (2 R)
    V_terminal = V_oc - I*R                   (voltage sag)
    SOC       -= I*dt / capacity
The terminal voltage feeds back into the motor model (available thrust ~ V^2).
"""
from __future__ import annotations

import math

import numpy as np


class BatteryModel:
    def __init__(self, cfg: dict):
        b = cfg["battery"]
        self.cells = int(b["cells"])
        self.capacity_c = float(b["capacity_mah"]) * 3.6          # mAh -> coulomb
        self.R = float(b["internal_resistance_ohm"])
        self.soc_pts = np.array(b["ocv_soc_points"], float)
        self.v_pts = np.array(b["ocv_cell_v_points"], float)
        self.soc = float(b["initial_soc"])
        self.tau_f = float(b["voltage_filter_tau_s"])
        self.min_v, self.nom_v, self.max_v = b["min_v"], b["nominal_v"], b["max_v"]
        self.failsafe_v, self.failsafe_soc = b["failsafe_voltage_v"], b["failsafe_soc"]
        self.warn_v, self.min_safe_v = b["low_voltage_warn_v"], b["min_safe_voltage_v"]
        self.current_a = 0.0
        self.voltage_v = self.open_circuit_v()
        self.voltage_filtered_v = self.voltage_v
        self.used_mah = 0.0

    def open_circuit_v(self) -> float:
        return self.cells * float(np.interp(self.soc, self.soc_pts, self.v_pts))

    def step(self, power_w: float, dt: float):
        voc = self.open_circuit_v()
        disc = voc * voc - 4.0 * self.R * max(power_w, 0.0)
        if disc < 0:                                   # demanded power exceeds what the pack can deliver
            self.current_a = voc / (2 * self.R)
        else:
            self.current_a = (voc - math.sqrt(disc)) / (2 * self.R)
        self.voltage_v = voc - self.current_a * self.R
        dq = self.current_a * dt
        self.soc = max(0.0, self.soc - dq / self.capacity_c)
        self.used_mah += dq / 3.6
        a = 1.0 - math.exp(-dt / self.tau_f)
        self.voltage_filtered_v += a * (self.voltage_v - self.voltage_filtered_v)
        return self.voltage_v

    @property
    def remaining_mah(self) -> float:
        return self.soc * self.capacity_c / 3.6

    def low_battery_failsafe(self) -> bool:
        return self.voltage_filtered_v < self.failsafe_v or self.soc < self.failsafe_soc

    def arming_ok(self) -> bool:
        return self.open_circuit_v() >= self.min_safe_v and self.soc > self.failsafe_soc
