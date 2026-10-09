"""
Calibration 3, 4, 5, 6, 7: propulsion fitting tools (motor thrust, step response, yaw torque, max thrust,
battery voltage scaling). These turn bench data into the numbers for config/drone.yaml `propulsion`.

Bench CSV formats (header row required):
  thrust curve   : columns  rpm, thrust_n [, torque_nm] [, voltage_v]        -> kT, kQ (T = kT w^2, Q = kQ w^2)
  motor step     : columns  t_s, rpm                                         -> motor_time_constant_s (first-order fit)
  voltage sweep  : columns  voltage_v, thrust_n_at_full_command              -> loaded rpm scaling / thrust ~ V^2 check

  python -m calibration.fit_propulsion thrust  bench_thrust.csv
  python -m calibration.fit_propulsion step    bench_step.csv
  python -m calibration.fit_propulsion voltage bench_volt.csv
  python -m calibration.fit_propulsion selftest      # runs every fitter on synthetic data and on the simulator itself
Every command prints a ready-to-paste YAML snippet and writes logs/calibration/*.json.
"""
from __future__ import annotations

import csv
import math
import sys

import numpy as np

from calibration._util import write_report, yaml_snippet
from sim.config import load_config, derived
from sim.motor_model import MotorModel, RPM


def _load(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = list(r)
    return {k: np.array([float(x[k]) for x in rows]) for k in rows[0]}


def fit_thrust(rpm, thrust, torque=None):
    """Least squares through the origin: T = kT w^2 (and Q = kQ w^2)."""
    w2 = (np.asarray(rpm) * RPM) ** 2
    kT = float(np.sum(w2 * thrust) / np.sum(w2 * w2))
    res = {"thrust_coeff_kT": kT, "rms_thrust_residual_n": float(np.sqrt(np.mean((thrust - kT * w2) ** 2))),
           "r2": float(1 - np.sum((thrust - kT * w2) ** 2) / np.sum((thrust - thrust.mean()) ** 2))}
    if torque is not None:
        kQ = float(np.sum(w2 * torque) / np.sum(w2 * w2))
        res.update({"torque_coeff_kQ": kQ, "kQ_over_kT_m": kQ / kT})
    return res


def fit_time_constant(t, rpm):
    """Fit rpm(t) = r_inf + (r0 - r_inf) exp(-(t-t0)/tau) on a step (t0 = time of first change)."""
    t, rpm = np.asarray(t, float), np.asarray(rpm, float)
    r0, rinf = rpm[:3].mean(), rpm[-max(3, len(rpm) // 10):].mean()
    span = rinf - r0
    y = (rpm - r0) / span
    idx = np.where((y > 0.05) & (y < 0.95))[0]
    t0_idx = np.where(y > 0.02)[0][0] if np.any(y > 0.02) else 0
    if len(idx) < 3:
        return {"error": "not enough points in the 5-95 % rise"}
    A = np.vstack([t[idx] - t[t0_idx], np.ones(len(idx))]).T
    slope, _ = np.linalg.lstsq(A, np.log(np.clip(1 - y[idx], 1e-6, None)), rcond=None)[0]
    tau = -1.0 / slope
    return {"motor_time_constant_s": float(tau)}


def fit_voltage(volt, thrust):
    """thrust = c * V^p ; the simulator assumes p = 2 (loaded rpm proportional to bus voltage)."""
    p_exp, logc = np.polyfit(np.log(volt), np.log(thrust), 1)
    return {"thrust_voltage_exponent": float(p_exp), "thrust_at_nominal_22v2_n": float(math.exp(logc) * 22.2 ** p_exp),
            "model_assumes_exponent": 2.0}


def selftest():
    """Recover known parameters from synthetic noisy data, then measure the simulator's own motor model."""
    cfg = load_config()
    rng = np.random.default_rng(0)
    rpm = np.linspace(2000, 9000, 25)
    kT_true, kQ_true = 1.351e-5, 2.027e-7
    T = kT_true * (rpm * RPM) ** 2 + rng.normal(0, 0.05, rpm.size)
    Q = kQ_true * (rpm * RPM) ** 2 + rng.normal(0, 0.002, rpm.size)
    thr = fit_thrust(rpm, T, Q)
    # motor step on the simulator's own model
    m = MotorModel(cfg)
    V = cfg["battery"]["nominal_v"]
    t, r = [], []
    for k in range(int(1.0 / 0.002)):
        u = 0.3 if k * 0.002 < 0.2 else 0.7
        m.step([u] * 4, V, True, 0.002)
        t.append(k * 0.002); r.append(m.rpm()[0] + rng.normal(0, 5))
    tc = fit_time_constant(np.array(t)[100:], np.array(r)[100:])
    volts = np.linspace(19.2, 25.2, 7)
    fv = fit_voltage(volts, [MotorModel(cfg).thrust_max(v) for v in volts])
    rep = {"thrust_fit": thr, "true_kT": kT_true, "true_kQ": kQ_true, "step_fit": tc,
           "config_tau_s": cfg["propulsion"]["motor_time_constant_s"], "voltage_fit": fv,
           "ok": abs(thr["thrust_coeff_kT"] / kT_true - 1) < 0.02 and abs(thr["torque_coeff_kQ"] / kQ_true - 1) < 0.03
                 and abs(tc["motor_time_constant_s"] / cfg["propulsion"]["motor_time_constant_s"] - 1) < 0.15
                 and abs(fv["thrust_voltage_exponent"] - 2.0) < 1e-6}
    write_report("3_4_6_7_propulsion_fit_selftest", rep)
    return rep["ok"]


def main(argv):
    if not argv or argv[0] == "selftest":
        return 0 if selftest() else 1
    kind, path = argv[0], argv[1]
    d = _load(path)
    if kind == "thrust":
        res = fit_thrust(d["rpm"], d["thrust_n"], d.get("torque_nm"))
        print("# paste into config/drone.yaml -> propulsion:\n" + yaml_snippet({k: f"{v:.4e}" for k, v in res.items() if k.startswith(("thrust_c", "torque_c"))}))
    elif kind == "step":
        res = fit_time_constant(d["t_s"], d["rpm"])
        print("# paste into config/drone.yaml -> propulsion:\n" + yaml_snippet({"motor_time_constant_s": f"{res['motor_time_constant_s']:.4f}"}))
    elif kind == "voltage":
        res = fit_voltage(d["voltage_v"], d["thrust_n_at_full_command"])
    else:
        raise SystemExit(__doc__)
    write_report(f"fit_{kind}", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
