"""
Calibration 1 + 2: mass and inertia verification.

  python -m calibration.verify_mass_inertia

* sums the link masses in the URDF XML (does not trust drone.yaml),
* recomputes the composite CG and inertia tensor from the URDF,
* checks the PyBullet body dynamically: push with a known force (mass) and apply known torques about
  x/y/z (inertia) and compare with the drone.yaml tensor.
For the REAL airframe use a bifilar/trifilar pendulum (Izz, Ixx, Iyy) and a scale + balance rig for the
CG, then edit `components` in config/drone.yaml and re-run `python -m tools.build_urdf`.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p

from calibration._util import write_report
from sim.config import load_config, DEFAULT_URDF, inertia_matrix
from sim.sim_core import UavSim
from sim.vehicle_model import VehicleModel


def main():
    cfg = load_config()
    root = ET.parse(DEFAULT_URDF).getroot()
    urdf_mass = sum(float(l.find("inertial/mass").get("value")) for l in root.findall("link") if l.find("inertial/mass") is not None)
    m = VehicleModel(DEFAULT_URDF, cfg)
    I_cfg = inertia_matrix(cfg)
    rep = {"urdf_total_mass_kg": urdf_mass, "config_mass_kg": cfg["vehicle"]["mass_kg"],
           "mass_ok": abs(urdf_mass - 2.2) < 1e-6,
           "cg_in_base_link_m": m.cg_base.tolist(),
           "inertia_from_urdf_kgm2": {k: float(v) for k, v in zip(("Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"),
                                                                 (m.inertia_cg[0, 0], m.inertia_cg[1, 1], m.inertia_cg[2, 2],
                                                                  m.inertia_cg[0, 1], m.inertia_cg[0, 2], m.inertia_cg[1, 2]))},
           "inertia_matches_drone_yaml": bool(np.allclose(m.inertia_cg, I_cfg, atol=2e-6)),
           "principal_moments": np.linalg.eigvalsh(I_cfg).tolist(),
           "positive_definite": bool(np.all(np.linalg.eigvalsh(I_cfg) > 0))}
    s = UavSim(world=False, pos_enu=(0, 0, 30.0))
    p.setGravity(0, 0, 0, physicsClientId=s.pb)
    dt = s.dt
    s.vehicle.reset((0, 0, 30.0))
    for _ in range(200):
        p.applyExternalForce(s.vehicle.body, -1, [0, 0, 10.0], s.vehicle.read_state().pos_enu.tolist(), p.WORLD_FRAME, physicsClientId=s.pb)
        p.stepSimulation(physicsClientId=s.pb)
    a = s.vehicle.read_state().vel_enu[2] / (200 * dt)
    rep["pybullet_effective_mass_kg"] = 10.0 / a
    dyn = {}
    for ax, nm in enumerate(("Ixx", "Iyy", "Izz")):
        s.vehicle.reset((0, 0, 30.0))
        tau = np.zeros(3); tau[ax] = 0.5
        for _ in range(100):
            R = s.vehicle.read_state().R_enu_flu
            p.applyExternalTorque(s.vehicle.body, -1, (R @ tau).tolist(), p.WORLD_FRAME, physicsClientId=s.pb)
            p.stepSimulation(physicsClientId=s.pb)
        alpha = s.vehicle.read_state().omega_flu_body[ax] / (100 * dt)
        dyn[nm] = 0.5 / alpha
    rep["pybullet_effective_inertia_kgm2"] = dyn
    rep["pybullet_matches_yaml_within_3pct"] = all(abs(dyn[k] / I_cfg[i, i] - 1) < 0.03 for i, k in enumerate(("Ixx", "Iyy", "Izz")))
    s.close()
    write_report("1_2_mass_inertia", rep)
    return rep["mass_ok"] and rep["inertia_matches_drone_yaml"] and rep["pybullet_matches_yaml_within_3pct"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
