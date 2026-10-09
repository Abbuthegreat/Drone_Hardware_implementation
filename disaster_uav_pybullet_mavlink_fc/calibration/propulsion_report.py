"""
Calibration 3, 4, 5, 6, 7 on the SIMULATOR: what the current placeholder parameters imply.

  python -m calibration.propulsion_report

3 hover-thrust calibration : command / rpm needed for 5.3955 N per motor at 19.2, 22.2, 25.2 V
4 motor step response      : simulated 10-90 % rise time and fitted tau vs configured tau
5 yaw-torque calibration   : in PyBullet, differential CCW/CW thrust -> yaw angular acceleration -> kQ/kT recovered
6 maximum-thrust           : per-motor / total max thrust, thrust-to-weight ratio at each voltage
7 battery voltage/thrust   : loaded voltage + available thrust vs state of charge at hover load
These are consequences of PLACEHOLDER parameters. They are NOT measurements of the S3110 + propeller.
"""
import math

import numpy as np
import pybullet as p

from calibration._util import write_report
from calibration.fit_propulsion import fit_time_constant
from sim.battery import BatteryModel
from sim.config import load_config, derived
from sim.motor_model import MotorModel, RPM
from sim.sim_core import UavSim


def main():
    cfg = load_config()
    g, mass = cfg["vehicle"]["gravity_mps2"], cfg["vehicle"]["mass_kg"]
    rep = {"disclaimer": "placeholder propulsion parameters; no bench data yet"}
    # 3 ---------------------------------------------------------------- hover thrust
    m = MotorModel(cfg)
    T_h = mass * g / 4
    rep["3_hover"] = {}
    for V in (cfg["battery"]["min_v"], cfg["battery"]["nominal_v"], cfg["battery"]["max_v"]):
        u = float(m.command_for_thrust([T_h], V)[0])
        w = math.sqrt(T_h / m.kT)
        rep["3_hover"][f"{V}V"] = {"thrust_per_motor_n": T_h, "sum_n": 4 * T_h, "normalised_command": u, "rpm": w / RPM,
                                   "fraction_of_available_thrust": T_h / m.thrust_max(V)}
    # 4 ---------------------------------------------------------------- step response
    mm = MotorModel(cfg); V = cfg["battery"]["nominal_v"]
    t, r = [], []
    for k in range(int(1.2 / 0.001)):
        u = 0.3 if k * 0.001 < 0.2 else 0.7
        mm.step([u] * 4, V, True, 0.001)
        t.append(k * 0.001); r.append(mm.rpm()[0])
    t, r = np.array(t), np.array(r)
    y = (r - r[150]) / (r[-1] - r[150])
    t10 = t[np.argmax(y > 0.1)]; t90 = t[np.argmax(y > 0.9)]
    rep["4_step"] = {"rise_10_90_s": float(t90 - t10), "expected_2.2_tau_s": 2.2 * cfg["propulsion"]["motor_time_constant_s"],
                     "fitted_tau_s": fit_time_constant(t[180:], r[180:]).get("motor_time_constant_s"),
                     "configured_tau_s": cfg["propulsion"]["motor_time_constant_s"]}
    # 5 ---------------------------------------------------------------- yaw torque via PyBullet
    s = UavSim(world=False, pos_enu=(0, 0, 40.0))
    p.setGravity(0, 0, 0, physicsClientId=s.pb)
    km_cfg = cfg["propulsion"]["torque_coeff_kQ"] / cfg["propulsion"]["thrust_coeff_kT"]
    T = np.array([6.5, 6.5, 4.3, 4.3])          # CCW motors (1,2) higher: net clockwise yaw torque, zero net roll/pitch
    for _ in range(100):
        s.vehicle.apply_rotor_forces(T, km_cfg * T)
        s.vehicle.step()
    alpha = s.vehicle.read_state().omega_frd[2] / (100 * s.dt)
    Izz = s.cfg["mass_properties"]["inertia_about_cg_kgm2"]["Izz"]
    tau_z = Izz * alpha
    net = (T[0] + T[1]) - (T[2] + T[3])
    rep["5_yaw_torque"] = {"measured_yaw_torque_nm": tau_z, "recovered_kQ_over_kT_m": tau_z / net, "config_kQ_over_kT_m": km_cfg,
                           "direction": "clockwise seen from above (+z_FRD)" if alpha > 0 else "WRONG"}
    s.close()
    # 6 ---------------------------------------------------------------- max thrust
    rep["6_max_thrust"] = {}
    for V in (cfg["battery"]["min_v"], cfg["battery"]["nominal_v"], cfg["battery"]["max_v"]):
        tm = m.thrust_max(V)
        rep["6_max_thrust"][f"{V}V"] = {"per_motor_n": tm, "total_n": 4 * tm, "thrust_to_weight": 4 * tm / (mass * g),
                                        "rpm": m.omega_max(V) / RPM, "noload_kv_limit_rpm": cfg["propulsion"]["motor_kv_rpm_per_v"] * V}
    # 7 ---------------------------------------------------------------- battery: sag + available thrust vs SOC at hover power
    b = BatteryModel(cfg)
    P_hover = 4 * cfg["propulsion"]["torque_coeff_kQ"] * (math.sqrt(T_h / cfg["propulsion"]["thrust_coeff_kT"])) ** 3 / cfg["propulsion"]["esc_motor_efficiency"] + cfg["propulsion"]["avionics_power_w"]
    rows = []
    t_end = None
    for sec in range(0, 4000):
        b.step(P_hover, 1.0)
        if sec % 300 == 0:
            rows.append({"t_min": sec / 60, "soc": b.soc, "loaded_v": b.voltage_v, "current_a": b.current_a,
                         "available_total_thrust_n": 4 * m.thrust_max(b.voltage_v), "thrust_margin": 4 * m.thrust_max(b.voltage_v) / (mass * g)})
        if b.low_battery_failsafe() and t_end is None:
            t_end = sec
    rep["7_battery"] = {"hover_electrical_power_w": P_hover, "samples": rows, "time_to_failsafe_min_at_hover": None if t_end is None else t_end / 60.0}
    write_report("3_4_5_6_7_propulsion_sim_report", rep)
    ok = abs(rep["5_yaw_torque"]["recovered_kQ_over_kT_m"] / km_cfg - 1) < 0.05 and rep["5_yaw_torque"]["direction"].startswith("clockwise")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
