"""Config loading + validation. All calibration lives in config/drone.yaml."""
from __future__ import annotations

import math
import os

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "drone.yaml")
DEFAULT_URDF = os.path.join(ROOT, "urdf", "disaster_uav.urdf")


def load_config(path: str | None = None) -> dict:
    with open(path or os.environ.get("UAV_CONFIG") or DEFAULT_CONFIG, encoding="utf-8") as f:   # UAV_CONFIG: alternate YAML for controlled experiments
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def inertia_matrix(cfg) -> np.ndarray:
    i = cfg["mass_properties"]["inertia_about_cg_kgm2"]
    return np.array([[i["Ixx"], i["Ixy"], i["Ixz"]],
                     [i["Ixy"], i["Iyy"], i["Iyz"]],
                     [i["Ixz"], i["Iyz"], i["Izz"]]], dtype=float)


def derived(cfg) -> dict:
    """Quantities derived from calibration parameters (for reports and sanity checks)."""
    pr, bt, v = cfg["propulsion"], cfg["battery"], cfg["vehicle"]
    w_max = pr["max_loaded_rpm_at_nominal_v"] * 2 * math.pi / 60
    t_max = pr["thrust_coeff_kT"] * w_max ** 2
    weight = v["mass_kg"] * v["gravity_mps2"]
    return {
        "weight_n": weight,
        "hover_thrust_per_motor_n": weight / 4.0,
        "max_thrust_per_motor_nominal_v_n": t_max,
        "thrust_to_weight_nominal_v": 4 * t_max / weight,
        "hover_rpm_nominal_v": math.sqrt(weight / 4 / pr["thrust_coeff_kT"]) * 60 / (2 * math.pi),
        "noload_rpm": {k: pr["motor_kv_rpm_per_v"] * bt[k] for k in ("min_v", "nominal_v", "max_v")},
        "loaded_to_noload_ratio_nominal": pr["max_loaded_rpm_at_nominal_v"] / (pr["motor_kv_rpm_per_v"] * bt["nominal_v"]),
    }


def validate_config(cfg) -> None:
    pr, bt = cfg["propulsion"], cfg["battery"]
    d = derived(cfg)
    assert pr["max_loaded_rpm_at_nominal_v"] < pr["motor_kv_rpm_per_v"] * bt["nominal_v"], \
        "loaded RPM must be below the no-load KV*V limit"
    assert d["thrust_to_weight_nominal_v"] > 1.3, "vehicle cannot hover with margin"
    assert abs(bt["min_v"] - 3.2 * bt["cells"]) < 0.5 and bt["min_v"] < bt["nominal_v"] < bt["max_v"]
    hz = cfg["sim"]["physics_hz"]
    assert hz % cfg["flight_controller"]["control_hz"] == 0, "control_hz must divide physics_hz"
    assert hz % cfg["sensors"]["imu"]["rate_hz"] == 0, "imu rate_hz must divide physics_hz"
    I = inertia_matrix(cfg)
    assert np.all(np.linalg.eigvalsh(I) > 0), "inertia must be positive definite"
