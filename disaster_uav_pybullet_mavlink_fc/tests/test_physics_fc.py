"""
Physics sanity tests B-K (FC-level: commands go through the FC's setpoint interface, then the full
controller -> mixer -> motor lag -> PyBullet chain; no teleporting anywhere).
"""
import math

import numpy as np
import pytest

from common.frames import wrap_pi
from tests.helpers import Flyer


def test_B_gravity_free_fall_when_unarmed_and_unpowered(make_sim, cfg):
    g = cfg["vehicle"]["gravity_mps2"]
    s = make_sim(pos_enu=(0, 0, 20.0))
    assert not s.fc.armed
    s.run(0.1)                                            # drag is negligible at low speed -> acceleration is g
    assert s.state.vel_enu[2] == pytest.approx(-g * 0.1, rel=0.03)
    s.run(0.9)
    st = s.state
    assert 0.8 * g < -st.vel_enu[2] <= g * 1.0            # aerodynamic drag only ever slows the fall
    assert 20.0 - st.pos_enu[2] > 0.4 * g                 # it really fell (>~4 m in 1 s)
    assert s.motor_thrusts.sum() < 1e-6                   # no motor thrust while unarmed


def test_C_hover_thrust_balances_weight(make_sim, cfg):
    s = make_sim()
    fl = Flyer(s)
    n0, e0 = fl.takeoff(3.0, settle_s=9.0)
    z_samples, thr = [], []
    for _ in range(100):                                 # 5 s of hover
        fl.pos(n0, e0, s.fc.home_down - 3.0, 0.0)
        s.run(0.05)
        z_samples.append(s.state.pos_enu[2]); thr.append(s.motor_thrusts.sum())
    assert np.ptp(z_samples) < 0.15
    assert np.mean(thr) == pytest.approx(cfg["vehicle"]["mass_kg"] * cfg["vehicle"]["gravity_mps2"], rel=0.02)
    assert np.mean(thr) / 4 == pytest.approx(5.3955, rel=0.02)
    r, p, _ = s.state.euler_ned
    assert abs(r) < math.radians(1.0) and abs(p) < math.radians(1.0)


@pytest.fixture
def airborne(make_sim):
    s = make_sim(pos_enu=(0, 0, 0.15 + 0.02))
    fl = Flyer(s)
    n0, e0 = fl.takeoff(5.0, settle_s=10.0)
    return s, fl


def test_D_positive_climb_command_moves_up(airborne):
    s, fl = airborne
    z0 = s.state.pos_enu[2]
    fl.stream(3.0, lambda: fl.vel(0, 0, -1.0))          # NED: negative down = up
    assert s.state.vel_enu[2] == pytest.approx(1.0, abs=0.2)
    assert s.state.pos_enu[2] - z0 > 2.0


def test_E_negative_vertical_command_moves_down(airborne):
    s, fl = airborne
    z0 = s.state.pos_enu[2]
    fl.stream(3.0, lambda: fl.vel(0, 0, +0.8))
    assert s.state.vel_enu[2] == pytest.approx(-0.8, abs=0.2)
    assert z0 - s.state.pos_enu[2] > 1.5


def test_F_forward_velocity_command_accelerates_north(airborne):
    s, fl = airborne
    n0, e0 = s.state.pos_ned[:2]
    pitches = []
    def go():
        fl.vel(1.5, 0, 0)
        pitches.append(s.state.euler_ned[1])
    fl.stream(4.0, go)
    st = s.state
    assert st.vel_ned[0] == pytest.approx(1.5, abs=0.25) and abs(st.vel_ned[1]) < 0.3
    assert st.pos_ned[0] - n0 > 3.0 and abs(st.pos_ned[1] - e0) < 0.6
    assert min(pitches) < math.radians(-2.0)             # accelerating north requires nose DOWN (negative FRD pitch)
    assert s.state.vel_enu[1] == pytest.approx(st.vel_ned[0])       # PyBullet +Y is North


def test_G_lateral_command_moves_east(airborne):
    s, fl = airborne
    n0, e0 = s.state.pos_ned[:2]
    rolls = []
    def go():
        fl.vel(0, 1.5, 0)
        rolls.append(s.state.euler_ned[0])
    fl.stream(4.0, go)
    st = s.state
    assert st.vel_ned[1] == pytest.approx(1.5, abs=0.25) and abs(st.vel_ned[0]) < 0.3
    assert st.pos_ned[1] - e0 > 3.0
    assert max(rolls) > math.radians(2.0)                # accelerating east requires right-wing-down (positive roll)
    assert s.state.vel_enu[0] == pytest.approx(st.vel_ned[1])        # PyBullet +X is East


def test_H_positive_yaw_rate_rotates_clockwise_from_above(airborne):
    s, fl = airborne
    y0 = s.state.euler_ned[2]
    fl.stream(3.0, lambda: fl.vel(0, 0, 0, yaw_rate=0.5))
    dy = wrap_pi(s.state.euler_ned[2] - y0)
    assert dy == pytest.approx(1.5, abs=0.3)             # heading (clockwise from North) INCREASES
    assert s.state.omega_frd[2] > 0.3
    assert s.state.omega_flu_body[2] < -0.3              # i.e. clockwise about the ENU/FLU up axis
    fl.stream(3.0, lambda: fl.vel(0, 0, 0, yaw_rate=-0.5))
    assert wrap_pi(s.state.euler_ned[2] - y0 - dy) < -1.0


def test_yaw_hold_and_absolute_yaw_setpoint(airborne):
    s, fl = airborne
    n, e, d = s.state.pos_ned
    fl.stream(6.0, lambda: fl.pos(n, e, d, math.radians(90)))
    assert wrap_pi(s.state.euler_ned[2] - math.radians(90)) == pytest.approx(0, abs=math.radians(3))


def test_I_mixer_signs_in_the_physics_engine(make_sim, cfg):
    """Feed mixer outputs straight to the rotors (gravity off) and check the resulting rotation in PyBullet."""
    import pybullet as p
    s = make_sim(pos_enu=(0, 0, 30.0))
    km = cfg["propulsion"]["torque_coeff_kQ"] / cfg["propulsion"]["thrust_coeff_kT"]
    cases = {"roll": ([0.4, 0, 0], 0, +1), "pitch": ([0, 0.4, 0], 1, +1), "yaw": ([0, 0, 0.05], 2, +1)}
    for name, (tau, ax, sign) in cases.items():
        p.setGravity(0, 0, 0, physicsClientId=s.pb)
        s.vehicle.reset((0, 0, 30.0), 0.0)
        T = s.mixer.mix(0.0, tau)
        T_pos = T - T.min() + 1.0                         # keep all thrusts positive; adding a constant does not change torques
        for _ in range(60):
            s.vehicle.apply_rotor_forces(T_pos, km * T_pos)
            s.vehicle.step()
        w = s.vehicle.read_state().omega_frd
        assert sign * w[ax] > 0.05, f"{name}: expected positive FRD rate, got {w}"
        other = [i for i in range(3) if i != ax]
        assert abs(w[ax]) > 3 * max(abs(w[other[0]]), abs(w[other[1]])), f"{name}: cross-coupling too large {w}"


def test_J_motor_lag_in_the_loop(make_sim):
    s = make_sim()
    fl = Flyer(s)
    s.run(0.3)
    fl.stream(0.5, lambda: fl.vel(0, 0, 0))
    s.fc.arm(s.t); s.fc.set_mode("OFFBOARD", s.t)
    fl.stream(0.05, lambda: fl.vel(0, 0, -2.0))
    thr_early = s.motor_thrusts.sum()
    fl.stream(1.5, lambda: fl.vel(0, 0, -2.0))
    thr_later = s.motor_thrusts.sum()
    assert thr_early < 0.6 * thr_later or thr_early < 21.0     # thrust ramps, it is not a step
    assert s.fc.last_thrusts.sum() > 0


def test_K_setpoint_loss_triggers_failsafe_hold(airborne, cfg):
    s, fl = airborne
    fl.stream(3.0, lambda: fl.vel(2.0, 0, 0))
    v_before = s.state.vel_ned[0]
    assert v_before > 1.2
    t_stop = s.t
    s.run(cfg["flight_controller"]["offboard_timeout_s"] + 0.15)          # no more setpoints
    assert s.fc.mode == "FAILSAFE_HOLD" and s.fc.failsafe_reason == "OFFBOARD_SETPOINT_LOSS"
    assert s.fc.system_status() == 5
    n_at_fs = s.state.pos_ned[0]
    s.run(6.0)
    assert np.linalg.norm(s.state.vel_ned) < 0.3          # braked to a stop, did NOT keep executing the last command
    assert s.state.pos_ned[0] - n_at_fs < 4.0
    assert s.state.pos_enu[2] > 2.0                       # still airborne, holding
    hold = s.state.pos_ned.copy()
    s.run(3.0)
    assert np.linalg.norm(s.state.pos_ned - hold) < 0.5   # position hold
    # setpoints alone do not silently resume Offboard control; an explicit mode request is required
    fl.vel(1.0, 0, 0)
    s.run(0.5)
    assert s.fc.mode == "FAILSAFE_HOLD"
    assert s.fc.set_mode("OFFBOARD", s.t) == 0


def test_offboard_entry_requires_setpoint_stream(make_sim):
    s = make_sim()
    s.run(0.2)
    assert s.fc.arm(s.t) == 0
    assert s.fc.set_mode("OFFBOARD", s.t) != 0            # no stream -> denied
    assert s.fc.mode == "HOLD"


def test_disarm_in_flight_denied_unless_forced_then_thrust_falls(airborne):
    s, fl = airborne
    assert s.fc.disarm(s.t) != 0 and s.fc.armed
    hover = s.motor_thrusts.sum()
    assert s.fc.disarm(s.t, force=True) == 0
    s.run(0.6)
    assert s.motor_thrusts.sum() < 0.05 * hover


def test_land_mode_descends_lands_and_auto_disarms(airborne):
    s, fl = airborne
    assert s.fc.set_mode("LAND", s.t) == 0
    s.run(25.0)
    assert s.state.pos_enu[2] < 0.3
    assert not s.fc.armed
    assert s.motor_thrusts.sum() < 1.0
