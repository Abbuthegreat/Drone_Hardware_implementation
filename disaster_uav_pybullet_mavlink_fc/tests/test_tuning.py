"""Controller tuning acceptance criteria (see calibration/tune_controller.py; logs are written to logs/tuning/)."""
import math

import pytest

from calibration import tune_controller as tc


def test_vertical_step():
    _, m = tc.test_vertical(quick=True)
    assert m["overshoot_pct"] < 10 and m["settling_time_s"] < 5 and m["steady_state_error"] < 0.05


def test_attitude_steps():
    _, m = tc.test_attitude(quick=True)
    for ax in ("roll", "pitch"):
        assert m[ax]["overshoot_pct"] < 20 and m[ax]["settling_time_s"] < 1.0 and m[ax]["steady_state_error"] < 0.5   # degrees


def test_yaw_step():
    _, m = tc.test_yaw(quick=True)
    assert m["overshoot_pct"] < 15 and m["settling_time_s"] < 3.0 and m["steady_state_error"] < 1.0


def test_velocity_step():
    _, m = tc.test_velocity(quick=True)
    assert m["overshoot_pct"] < 15 and m["settling_time_s"] < 3.0 and m["steady_state_error"] < 0.1
    assert m["max_cross_track_velocity"] < 0.2 and m["max_altitude_deviation_m"] < 0.3


def test_position_step():
    _, m = tc.test_position(quick=True)
    assert m["overshoot_pct"] < 10 and m["settling_time_s"] < 10 and m["oscillation_extrema_outside_band"] <= 1
    assert m["max_cross_track_m"] < 0.3


def test_hover_drift_and_oscillation():
    _, m = tc.test_hover_drift(quick=True)
    assert m["max_position_drift_m"] < 0.1 and m["rms_tilt_deg"] < 0.5
    assert m["mean_thrust_N"] == pytest.approx(21.582, rel=0.02) and m["thrust_std_N"] < 0.5


def test_wind_rejection():
    _, m = tc.test_wind_rejection(quick=True)
    assert m["peak_position_error_m"] < 2.0 and m["final_position_error_m"] < 0.3
    assert m["steady_pitch_deg"] > 0      # leaning into the wind (nose up = positive) against the tail... see log
