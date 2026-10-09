"""Test J (motor lag) + thrust/battery model sanity."""
import math

import pytest

from sim.battery import BatteryModel
from sim.motor_model import MotorModel, RPM


def test_hover_command_gives_expected_thrust(cfg):
    m = MotorModel(cfg)
    V = cfg["battery"]["nominal_v"]
    u = m.command_for_thrust([21.582 / 4] * 4, V)
    assert 0.2 < u[0] < 0.9
    for _ in range(4000):
        thr, _ = m.step(u, V, True, 0.002)
    assert thr.sum() == pytest.approx(21.582, rel=1e-3)


def test_motor_lag_is_first_order(cfg):
    m = MotorModel(cfg)
    V = cfg["battery"]["nominal_v"]
    for _ in range(3000):
        m.step([0.4] * 4, V, True, 0.002)
    t0 = m.outputs()[0].copy()
    w0, wt = m.omega.copy(), m.omega_target([0.9] * 4, V, True)
    thr1, _ = m.step([0.9] * 4, V, True, 0.002)
    t_target = m.kT * wt[0] ** 2
    assert (thr1[0] - t0[0]) / (t_target - t0[0]) < 0.1     # after one 2 ms step thrust is still >90% away from its target
    tau = cfg["propulsion"]["motor_time_constant_s"]
    for _ in range(int(round(tau / 0.002)) - 1):
        m.step([0.9] * 4, V, True, 0.002)
    frac = (m.omega[0] - w0[0]) / (wt[0] - w0[0])
    assert frac == pytest.approx(1 - math.exp(-1), abs=0.03)     # 63% after one time constant


def test_thrust_falls_when_disarmed(cfg):
    m = MotorModel(cfg)
    V = cfg["battery"]["nominal_v"]
    for _ in range(2000):
        m.step([0.5] * 4, V, True, 0.002)
    T_on = m.outputs()[0].sum()
    for _ in range(int(0.5 / 0.002)):
        m.step([0.5] * 4, V, False, 0.002)
    assert m.outputs()[0].sum() < 0.01 * T_on


def test_available_thrust_scales_with_voltage_squared(cfg):
    m = MotorModel(cfg)
    assert m.thrust_max(19.2) / m.thrust_max(25.2) == pytest.approx((19.2 / 25.2) ** 2)
    assert m.thrust_max(cfg["battery"]["nominal_v"]) * 4 > 21.582 * 1.5     # margin to hover at nominal
    assert m.omega_max(22.2) == pytest.approx(cfg["propulsion"]["max_loaded_rpm_at_nominal_v"] * RPM)


def test_saturation_at_low_voltage(cfg):
    m = MotorModel(cfg)
    for _ in range(3000):
        thr, _ = m.step([1.0] * 4, 19.2, True, 0.002)
    assert thr[0] == pytest.approx(m.thrust_max(19.2), rel=1e-3)


def test_yaw_torque_ratio(cfg):
    m = MotorModel(cfg)
    m.omega[:] = 600.0
    T, Q = m.outputs()
    assert Q[0] / T[0] == pytest.approx(cfg["propulsion"]["torque_coeff_kQ"] / cfg["propulsion"]["thrust_coeff_kT"])


def test_battery_sag_and_soc(cfg):
    b = BatteryModel(cfg)
    v_oc = b.open_circuit_v()
    assert cfg["battery"]["min_v"] <= v_oc <= cfg["battery"]["max_v"]
    b.step(300.0, 1.0)
    assert b.voltage_v < v_oc and b.current_a > 5
    soc0 = b.soc
    for _ in range(600):
        b.step(300.0, 1.0)                              # 10 minutes of hover-ish load
    assert b.soc < soc0 and b.used_mah > 500
    b2 = BatteryModel(cfg)
    b2.step(1000.0, 1.0)
    assert b2.voltage_v < BatteryModel(cfg).open_circuit_v() - 1.0     # heavy load -> visible sag


def test_battery_failsafe_and_arming(cfg):
    b = BatteryModel(cfg)
    assert b.arming_ok() and not b.low_battery_failsafe()
    b.soc = 0.05
    b.voltage_filtered_v = b.open_circuit_v()
    assert b.low_battery_failsafe() and not b.arming_ok()
