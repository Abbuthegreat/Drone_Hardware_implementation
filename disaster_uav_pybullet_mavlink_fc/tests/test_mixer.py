"""Test I - motor mixer signs (mathematical, and against the physics: torque = r x F)."""
import numpy as np
import pytest

from sim.config import DEFAULT_URDF
from sim.mixer import QuadMixer
from sim.vehicle_model import VehicleModel


@pytest.fixture(scope="module")
def mixer(cfg):
    m = VehicleModel(DEFAULT_URDF, cfg)
    km = cfg["propulsion"]["torque_coeff_kQ"] / cfg["propulsion"]["thrust_coeff_kT"]
    return QuadMixer(m.rotor_pos_frd, m.spin, km), m


HOVER = 21.582


def test_positive_collective_raises_all_motors(mixer):
    mx, _ = mixer
    a, b = mx.mix(HOVER, [0, 0, 0]), mx.mix(HOVER + 4.0, [0, 0, 0])
    assert np.all(b > a) and np.allclose(b - a, 1.0, atol=0.03)      # CG is ~1 mm off the geometric centre (thermal camera), so motors differ ~1 %
    assert np.allclose(a, HOVER / 4, atol=0.1) and a.sum() == pytest.approx(HOVER)


def test_positive_roll_right_wing_down_needs_more_thrust_on_LEFT_motors(mixer):
    mx, _ = mixer                       # motors: 1 FR, 2 RL, 3 FL, 4 RR
    d = mx.mix(HOVER, [0.5, 0, 0]) - HOVER / 4
    assert d[1] > 0 and d[2] > 0 and d[0] < 0 and d[3] < 0     # left (2,3) up, right (1,4) down
    assert d.sum() == pytest.approx(0, abs=1e-9)


def test_positive_pitch_nose_up_needs_more_thrust_on_FRONT_motors(mixer):
    mx, _ = mixer
    d = mx.mix(HOVER, [0, 0.5, 0]) - HOVER / 4
    assert d[0] > 0 and d[2] > 0 and d[1] < 0 and d[3] < 0     # front (1,3) up, rear (2,4) down
    assert d.sum() == pytest.approx(0, abs=1e-9)


def test_positive_yaw_clockwise_needs_more_thrust_on_CCW_motors(mixer):
    mx, m = mixer                       # CCW props (1,2) push the body clockwise
    d = mx.mix(HOVER, [0, 0, 0.05]) - HOVER / 4
    assert d[0] > 0 and d[1] > 0 and d[2] < 0 and d[3] < 0
    assert d.sum() == pytest.approx(0, abs=1e-9)
    assert list(m.spin) == [1, 1, -1, -1]


def test_forward_matches_rigid_body_torque(mixer, cfg):
    """Recompute torque from r x F using FRD geometry and compare with the requested torque."""
    mx, m = mixer
    tau_req = np.array([0.3, -0.2, 0.04])
    T = mx.mix(HOVER, tau_req)
    tau = np.zeros(3)
    km = cfg["propulsion"]["torque_coeff_kQ"] / cfg["propulsion"]["thrust_coeff_kT"]
    for i in range(4):
        r = m.rotor_pos_frd[i]
        F = np.array([0, 0, -T[i]])                  # thrust along -z_frd
        tau += np.cross(r, F) + np.array([0, 0, m.spin[i] * km * T[i]])
    assert np.allclose(tau, tau_req, atol=1e-9)
    assert mx.forward(T)[0] == pytest.approx(HOVER)


def test_saturation_keeps_torque_priority_and_bounds(mixer):
    mx, _ = mixer
    t_min, t_max = 0.1, 11.0
    out, info = mx.mix_saturated(4 * t_max, [1.5, 0, 0], t_min, t_max)      # full collective + big roll
    assert out.min() >= t_min - 1e-9 and out.max() <= t_max + 1e-9
    assert info["saturated"]
    assert (out[1] + out[2]) > (out[0] + out[3])          # roll torque still has the right sign
    out2, info2 = mx.mix_saturated(HOVER, [0.1, 0.1, 0.01], t_min, t_max)
    assert not info2["saturated"]
    assert np.allclose(out2, mx.mix(HOVER, [0.1, 0.1, 0.01]))
