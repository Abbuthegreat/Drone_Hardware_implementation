"""
Controller tuning / verification harness (calibration item 8).

Tuning order (as required): 1 thrust/vertical  2 attitude  3 yaw  4 velocity  5 position  + hover drift.
Each test flies the closed loop in PyBullet through the FC's real setpoint interface, saves the
time series to logs/tuning/<name>.csv and computes overshoot, rise time, settling time (2 % band),
steady-state error and oscillation count.

    python -m calibration.tune_controller            # full set
    python -m calibration.tune_controller --quick    # shorter runs
Gains live in config/drone.yaml (flight_controller.*). Edit them there and re-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.frames import wrap_pi, R_from_euler  # noqa: E402
from sim.flight_controller_sim import IGN_POS, IGN_VEL, IGN_ACC, IGN_YAW, IGN_YAW_RATE  # noqa: E402
from sim.sim_core import UavSim  # noqa: E402
from tests.helpers import Flyer  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs", "tuning")


def step_metrics(t, y, y0, y_target, band=0.02):
    """Step-response metrics for a signal y(t) going from y0 to y_target."""
    t, y = np.asarray(t), np.asarray(y)
    span = y_target - y0
    if abs(span) < 1e-9:
        return {}
    yn = (y - y0) / span                                   # normalised 0 -> 1
    overshoot = max(0.0, float(yn.max() - 1.0)) * 100.0
    def first(th):
        idx = np.where(yn >= th)[0]
        return float(t[idx[0]] - t[0]) if len(idx) else float("nan")
    tr = first(0.9) - first(0.1)
    outside = np.where(np.abs(yn - 1.0) > band)[0]
    ts = float(t[outside[-1]] - t[0]) if len(outside) else 0.0
    if len(outside) and outside[-1] == len(yn) - 1:
        ts = float("nan")                                  # never settled
    tail = yn[int(0.8 * len(yn)):]
    ess = float(abs(tail.mean() - 1.0) * abs(span))
    dy = np.diff(yn)
    ext = np.where(np.sign(dy[:-1]) * np.sign(dy[1:]) < 0)[0] + 1          # local extrema
    n_osc = int(np.sum(np.abs(yn[ext] - 1.0) > 2 * band)) if len(ext) else 0   # only extrema that leave the +-4 % band
    return {"overshoot_pct": round(overshoot, 2), "rise_time_s": round(tr, 3), "settling_time_s": round(ts, 3),
            "steady_state_error": round(ess, 4), "oscillation_extrema_outside_band": n_osc}


def save_csv(name, cols, rows):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def _airborne(alt=5.0):
    s = UavSim(world=False)
    fl = Flyer(s)
    n0, e0 = fl.takeoff(alt, settle_s=10.0)
    return s, fl, n0, e0


def _record(s, fl, seconds, send, fields):
    rows = []
    period = fl.period
    for _ in range(int(seconds / period)):
        send()
        s.run(period)
        rows.append([s.t] + [f(s) for f in fields.values()])
    return rows


def test_vertical(quick):
    """1. thrust / vertical controller: altitude step +2 m and hover thrust."""
    s, fl, n0, e0 = _airborne(3.0)
    d0 = s.state.pos_ned[2]
    fields = {"z_up_m": lambda s: -s.state.pos_ned[2], "vz_up": lambda s: -s.state.vel_ned[2],
              "thrust_total_N": lambda s: float(s.motor_thrusts.sum())}
    rows = _record(s, fl, 10 if quick else 14, lambda: fl.pos(n0, e0, d0 - 2.0, 0.0), fields)
    save_csv("1_vertical_alt_step", ["t"] + list(fields), rows)
    a = np.array(rows)
    m = step_metrics(a[:, 0], a[:, 1], -d0, -d0 + 2.0)
    s.close()
    return "vertical_altitude_step_2m", m


def test_attitude(quick):
    """2. attitude controller: roll and pitch steps of 10 deg via SET_ATTITUDE_TARGET (attitude + thrust)."""
    out = {}
    for axis, name in ((0, "roll"), (1, "pitch")):
        s, fl, n0, e0 = _airborne(5.0)
        hover_thr = s.cfg["vehicle"]["mass_kg"] * s.cfg["vehicle"]["gravity_mps2"] / (4 * s.motor.thrust_max(s.battery.voltage_v))
        ang = math.radians(10.0)
        rp = [0.0, 0.0]; rp[axis] = ang
        yaw0 = s.state.euler_ned[2]
        R = R_from_euler(rp[0], rp[1], yaw0)
        from common.frames import quat_from_R
        q = quat_from_R(R)
        rows = []
        for i in range(int((1.0 if quick else 1.5) / 0.05)):        # hold level first (few samples), then step
            pass
        fields = {"angle_deg": lambda s, a=axis: math.degrees(s.state.euler_ned[a]), "rate_dps": lambda s, a=axis: math.degrees(s.state.omega_frd[a])}
        t_end = 1.2 if quick else 1.6
        for _ in range(int(t_end / 0.05)):
            s.fc.handle_attitude_target(s.t, q, hover_thr, 0)
            s.run(0.05)
            rows.append([s.t] + [f(s) for f in fields.values()])
        save_csv(f"2_attitude_{name}_step_10deg", ["t"] + list(fields), rows)
        a = np.array(rows)
        out[name] = step_metrics(a[:, 0], a[:, 1], 0.0, 10.0)
        s.close()
    return "attitude_step_10deg", out


def test_yaw(quick):
    """3. yaw controller: 45 deg yaw step."""
    s, fl, n0, e0 = _airborne(5.0)
    d = s.state.pos_ned[2]
    y0 = s.state.euler_ned[2]
    fields = {"yaw_deg": lambda s: math.degrees(wrap_pi(s.state.euler_ned[2] - y0)), "yaw_rate_dps": lambda s: math.degrees(s.state.omega_frd[2])}
    rows = _record(s, fl, 5 if quick else 7, lambda: fl.pos(n0, e0, d, y0 + math.radians(45)), fields)
    save_csv("3_yaw_step_45deg", ["t"] + list(fields), rows)
    a = np.array(rows)
    m = step_metrics(a[:, 0], a[:, 1], 0.0, 45.0)
    s.close()
    return "yaw_step_45deg", m


def test_velocity(quick):
    """4. velocity controller: 0 -> 2 m/s North step."""
    s, fl, n0, e0 = _airborne(5.0)
    fields = {"vN": lambda s: s.state.vel_ned[0], "vE": lambda s: s.state.vel_ned[1],
              "pitch_deg": lambda s: math.degrees(s.state.euler_ned[1]), "alt_err": lambda s: -s.state.pos_ned[2] - (-s.fc.home_down + 5.0)}
    rows = _record(s, fl, 6 if quick else 9, lambda: fl.vel(2.0, 0, 0), fields)
    save_csv("4_velocity_step_2mps", ["t"] + list(fields), rows)
    a = np.array(rows)
    m = step_metrics(a[:, 0], a[:, 1], 0.0, 2.0)
    m["max_cross_track_velocity"] = round(float(np.abs(a[:, 2]).max()), 3)
    m["max_altitude_deviation_m"] = round(float(np.abs(a[:, 4]).max()), 3)
    s.close()
    return "velocity_step_2mps", m


def test_position(quick):
    """5. position controller: 5 m North step."""
    s, fl, n0, e0 = _airborne(5.0)
    d = s.state.pos_ned[2]
    fields = {"N_m": lambda s: s.state.pos_ned[0] - n0, "E_m": lambda s: s.state.pos_ned[1] - e0, "vN": lambda s: s.state.vel_ned[0]}
    rows = _record(s, fl, 12 if quick else 18, lambda: fl.pos(n0 + 5.0, e0, d, 0.0), fields)
    save_csv("5_position_step_5m", ["t"] + list(fields), rows)
    a = np.array(rows)
    m = step_metrics(a[:, 0], a[:, 1], 0.0, 5.0, band=0.02)
    m["max_cross_track_m"] = round(float(np.abs(a[:, 2]).max()), 3)
    s.close()
    return "position_step_5m", m


def test_hover_drift(quick):
    """Hover stability / drift with the FC estimator noise and IMU noise switched on."""
    s, fl, n0, e0 = _airborne(4.0)
    d = s.state.pos_ned[2]
    fields = {"N_m": lambda s: s.state.pos_ned[0] - n0, "E_m": lambda s: s.state.pos_ned[1] - e0, "D_m": lambda s: s.state.pos_ned[2] - d,
              "roll_deg": lambda s: math.degrees(s.state.euler_ned[0]), "pitch_deg": lambda s: math.degrees(s.state.euler_ned[1]),
              "thrust_total_N": lambda s: float(s.motor_thrusts.sum())}
    rows = _record(s, fl, 15 if quick else 40, lambda: fl.pos(n0, e0, d, 0.0), fields)
    save_csv("6_hover_drift", ["t"] + list(fields), rows)
    a = np.array(rows)
    tail = a[len(a) // 4:]
    m = {"max_position_drift_m": round(float(np.linalg.norm(tail[:, 1:4], axis=1).max()), 4),
         "rms_tilt_deg": round(float(np.sqrt(np.mean(tail[:, 4] ** 2 + tail[:, 5] ** 2))), 4),
         "mean_thrust_N": round(float(tail[:, 6].mean()), 3), "thrust_std_N": round(float(tail[:, 6].std()), 3)}
    s.close()
    return "hover_stability", m


def test_wind_rejection(quick):
    """Disturbance rejection: constant 4 m/s wind from the South (blowing North) while holding position."""
    s, fl, n0, e0 = _airborne(4.0)
    d = s.state.pos_ned[2]
    s.vehicle.wind_enu = np.array([0.0, 4.0, 0.0])           # ENU wind vector (4 m/s toward North)
    fields = {"N_m": lambda s: s.state.pos_ned[0] - n0, "E_m": lambda s: s.state.pos_ned[1] - e0,
              "D_m": lambda s: s.state.pos_ned[2] - d, "pitch_deg": lambda s: math.degrees(s.state.euler_ned[1])}
    rows = _record(s, fl, 15 if quick else 25, lambda: fl.pos(n0, e0, d, 0.0), fields)
    save_csv("7_wind_4mps_hold", ["t"] + list(fields), rows)
    a = np.array(rows)
    m = {"peak_position_error_m": round(float(np.linalg.norm(a[:, 1:4], axis=1).max()), 3),
         "final_position_error_m": round(float(np.linalg.norm(a[-20:, 1:4].mean(axis=0))), 4),
         "steady_pitch_deg": round(float(a[-20:, 4].mean()), 2)}
    s.close()
    return "wind_4mps_hold", m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    results = {}
    for fn in (test_vertical, test_attitude, test_yaw, test_velocity, test_position, test_hover_drift, test_wind_rejection):
        name, m = fn(a.quick)
        results[name] = m
        print(f"{name}: {json.dumps(m)}", flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    return results


if __name__ == "__main__":
    main()
