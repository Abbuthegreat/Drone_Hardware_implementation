"""Tests A (mass), inertia, frame geometry. Pure XML/numpy except the dynamic inertia check."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from common.frames import R_LINK_OPTICAL
from sim.config import DEFAULT_URDF, inertia_matrix, derived
from sim.vehicle_model import VehicleModel

REQUIRED_LINKS = ["base_link", "flight_controller_link", "raspberry_pi_link", "gps_link", "rgbd_camera_link", "thermal_camera_link",
                  "rgbd_camera_optical_link", "battery_link", "esc_link"] + \
                 [f"motor_{i}_link" for i in range(1, 5)] + [f"rotor_{i}_link" for i in range(1, 5)]


def test_urdf_is_valid_xml_with_required_links():
    root = ET.parse(DEFAULT_URDF).getroot()
    assert root.tag == "robot"
    names = {l.get("name") for l in root.findall("link")}
    assert set(REQUIRED_LINKS) <= names
    joints = root.findall("joint")
    assert joints and all(j.get("type") == "fixed" for j in joints)
    for j in joints:                       # every transform explicit and inspectable
        o = j.find("origin")
        assert o is not None and o.get("xyz") and o.get("rpy")


def test_total_mass_is_2_2_kg_summed_from_urdf_links():
    root = ET.parse(DEFAULT_URDF).getroot()
    total = sum(float(l.find("inertial/mass").get("value")) for l in root.findall("link") if l.find("inertial/mass") is not None)
    assert total == pytest.approx(2.2, abs=1e-6)


def test_config_mass_properties_match_urdf(cfg):
    m = VehicleModel(DEFAULT_URDF, cfg)
    assert m.total_mass == pytest.approx(cfg["vehicle"]["mass_kg"], abs=1e-6)
    assert cfg["mass_properties"]["total_mass_kg"] == pytest.approx(2.2, abs=1e-6)
    assert np.allclose(m.cg_base, cfg["mass_properties"]["cg_in_base_link_m"], atol=2e-6)
    I_cfg = inertia_matrix(cfg)
    assert np.allclose(m.inertia_cg, I_cfg, atol=2e-6), "drone.yaml inertia is stale - run python -m tools.build_urdf"


def test_inertia_is_physical(cfg):
    I = inertia_matrix(cfg)
    assert np.allclose(I, I.T)
    ev = np.linalg.eigvalsh(I)
    assert np.all(ev > 0)
    a, b, c = ev
    assert a + b >= c - 1e-12 and a + c >= b - 1e-12 and b + c >= a - 1e-12   # triangle inequality of principal moments


def _cloud_box(m, size, xyz, R, n=9):
    g = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(g * size[0], g * size[1], g * size[2], indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) @ R.T + xyz
    return pts, np.full(len(pts), m / len(pts))


def _cloud_cyl(m, radius, length, xyz, n=13):
    g = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(g * radius, g * radius, indexing="ij")
    ok = (X ** 2 + Y ** 2) <= radius ** 2
    z = (np.arange(7) + 0.5) / 7 * length - length / 2
    pts = np.array([[x, y, zz] for x, y in zip(X[ok], Y[ok]) for zz in z]) + xyz
    return pts, np.full(len(pts), m / len(pts))


def _cloud_rod(m, a, b, n=400):
    t = (np.arange(n) + 0.5) / n
    pts = np.array(a) + np.outer(t, np.array(b) - np.array(a))
    return pts, np.full(n, m / n)


def test_inertia_independent_point_cloud_integration(cfg):
    """Discretise every component of the yaml table into thousands of point masses and integrate
    m*(r^2 I - r r^T) about the CG. Independent of the URDF generator's analytic formulas."""
    from common.frames import R_from_euler
    c = cfg["components"]
    bx = cfg["mass_properties"]["battery_x_solved_m"]
    P, M = [], []
    def add(pm):
        P.append(pm[0]); M.append(pm[1])
    for pr in c["airframe_primitives"]:
        if pr["kind"] == "box":
            add(_cloud_box(pr["mass_kg"], pr["size_m"], np.array(pr["xyz"], float), np.eye(3)))
        else:
            add(_cloud_rod(pr["mass_kg"], pr["from_xyz"], pr["to_xyz"]))
    add(_cloud_box(c["base_link"]["mass_kg"], c["base_link"]["size_m"], np.zeros(3), np.eye(3), n=3))
    for name in ("flight_controller_link", "raspberry_pi_link", "esc_link", "battery_link", "rgbd_camera_link", "thermal_camera_link"):
        d = c[name]
        xyz = np.array([bx if v == "auto" else v for v in d["xyz"]], float)
        R = R_from_euler(*[math.radians(a) for a in d["rpy_deg"]]) if "rpy_deg" in d else np.eye(3)
        add(_cloud_box(d["mass_kg"], d["size_m"], xyz, R))
    g = c["gps_link"]
    add(_cloud_cyl(g["mass_kg"], g["radius_m"], g["length_m"], np.array(g["xyz"], float)))
    for mt in c["motors"]:
        xy = mt["xy"]
        add(_cloud_cyl(c["motor"]["mass_kg"], c["motor"]["radius_m"], c["motor"]["length_m"], np.array([xy[0], xy[1], 0.0])))
        add(_cloud_cyl(c["rotor"]["mass_kg"], c["rotor"]["radius_m"], c["rotor"]["thickness_m"],
                       np.array([xy[0], xy[1], c["rotor"]["z_above_motor_m"]]), n=21))
    P, M = np.vstack(P), np.concatenate(M)
    assert M.sum() == pytest.approx(2.2, abs=1e-9)
    cg = (P * M[:, None]).sum(0) / M.sum()
    r = P - cg
    I = sum(m * (ri @ ri * np.eye(3) - np.outer(ri, ri)) for ri, m in zip(r, M))
    I_cfg = inertia_matrix(cfg)
    assert np.allclose(cg, cfg["mass_properties"]["cg_in_base_link_m"], atol=1e-4)
    assert np.allclose(np.diag(I), np.diag(I_cfg), rtol=0.03)
    assert I[0, 2] == pytest.approx(I_cfg[0, 2], abs=1e-4)


def test_frame_geometry(cfg):
    m = VehicleModel(DEFAULT_URDF, cfg)
    # rotor positions: X configuration, arm length 0.30 m, PX4 numbering, symmetric about the CG
    arm = cfg["vehicle"]["arm_length_m"]
    r = m.rotor_pos_flu
    assert r.shape == (4, 3)
    assert np.allclose(np.hypot(r[:, 0], r[:, 1]), arm, atol=2e-3)
    assert np.allclose(np.abs(r[:, 0]), np.abs(r[:, 1]), atol=2e-3)
    assert r[0, 0] > 0 and r[0, 1] < 0        # 1 front-right
    assert r[1, 0] < 0 and r[1, 1] > 0        # 2 rear-left
    assert r[2, 0] > 0 and r[2, 1] > 0        # 3 front-left
    assert r[3, 0] < 0 and r[3, 1] < 0        # 4 rear-right
    assert list(m.spin) == [1, 1, -1, -1]     # CCW, CCW, CW, CW  (diagonal pairs share direction)
    assert m.spin[0] == m.spin[1] and np.sign(r[0, 0] * r[0, 1]) == np.sign(r[1, 0] * r[1, 1])   # 1,2 on same diagonal
    # CG near the physical centre
    assert abs(m.cg_base[0]) < 5e-3 and abs(m.cg_base[1]) < 5e-3 and abs(m.cg_base[2]) < 0.04
    # Pixhawk near the CG; GPS far from the Pi; camera forward of the CG with the explicit optical frame
    assert np.linalg.norm(m.imu_pos_flu) < 0.08
    pi = m.T["raspberry_pi_link"][0]
    assert np.linalg.norm(m.T["gps_link"][0] - pi) > 0.12
    assert m.cam_pos_flu[0] > 0.1
    R_link_opt = m.T["rgbd_camera_link"][1].T @ m.T["rgbd_camera_optical_link"][1]
    assert np.allclose(R_link_opt, R_LINK_OPTICAL, atol=1e-5)
    # battery placed so CG is not displaced: within the airframe footprint
    assert abs(m.T["battery_link"][0][0]) < 0.06


def test_hover_thrust_calibration_numbers(cfg):
    d = derived(cfg)
    assert d["weight_n"] == pytest.approx(21.582, abs=1e-3)
    assert d["hover_thrust_per_motor_n"] == pytest.approx(5.3955, abs=1e-4)
    # 800 KV * V is only the no-load speed and must exceed the loaded rpm
    assert d["noload_rpm"]["min_v"] == pytest.approx(15360) and d["noload_rpm"]["nominal_v"] == pytest.approx(17760)
    assert d["noload_rpm"]["max_v"] == pytest.approx(20160)
    assert cfg["propulsion"]["max_loaded_rpm_at_nominal_v"] < d["noload_rpm"]["nominal_v"]


def test_pybullet_dynamic_inertia_matches_config(make_sim, cfg):
    """Apply pure body torques in PyBullet and compare angular acceleration with I^-1 tau (from drone.yaml)."""
    import pybullet as p
    s = make_sim(pos_enu=(0, 0, 20.0))
    p.setGravity(0, 0, 0, physicsClientId=s.pb)
    I = np.diag(np.diag(inertia_matrix(cfg)))       # engine uses the diagonal (Ixz neglected, see physics_pybullet.py)
    for ax in range(3):
        s.vehicle.reset((0, 0, 20.0))
        tau = np.zeros(3); tau[ax] = 0.5
        for _ in range(50):
            R = s.vehicle.read_state().R_enu_flu
            p.applyExternalTorque(s.vehicle.body, -1, (R @ tau).tolist(), p.WORLD_FRAME, physicsClientId=s.pb)
            p.stepSimulation(physicsClientId=s.pb)
        w = s.vehicle.read_state().omega_flu_body
        alpha = w / (50 * s.dt)
        assert alpha[ax] == pytest.approx(0.5 / I[ax, ax], rel=0.03)
        p.setGravity(0, 0, 0, physicsClientId=s.pb)
    # and the mass
    s.vehicle.reset((0, 0, 20.0))
    for _ in range(100):
        p.applyExternalForce(s.vehicle.body, -1, [0, 0, 10.0], s.vehicle.read_state().pos_enu.tolist(), p.WORLD_FRAME, physicsClientId=s.pb)
        p.stepSimulation(physicsClientId=s.pb)
    a = s.vehicle.read_state().vel_enu[2] / (100 * s.dt)
    assert 10.0 / a == pytest.approx(2.2, rel=0.01)
