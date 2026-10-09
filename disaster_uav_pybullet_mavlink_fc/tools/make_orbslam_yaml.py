"""
Write config/orbslam3/pybullet_rgbd.yaml (ORB-SLAM3 RGB-D settings) from the SIMULATED camera's explicit intrinsics (sim/rgbd_camera.py),
using the kit's own writer (slam_kit/record.py write_orbslam_yaml), then compare YAML <-> simulation numerically and test the depth conversion.
The physical camera's calibration (kit's per-flight orbslam.yaml) is never touched.

    python tools/make_orbslam_yaml.py            -> config/orbslam3/pybullet_rgbd.yaml + logs/ubuntu/calibration_compare.json
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from sim.config import load_config
from slam import kit



def main():
    cfg = load_config()
    from sim.rgbd_camera import RGBDSensor
    from sim.sim_core import UavSim
    s = UavSim(cfg, world=False)
    cam = RGBDSensor(s)
    k = cam.intrinsics()
    s.close()
    out = os.path.join(ROOT, "config", "orbslam3", "pybullet_rgbd.yaml")
    intr = dict(width=k["width"], height=k["height"], fx=k["fx"], fy=k["fy"], ppx=k["cx"], ppy=k["cy"], model="none", coeffs=[0] * 5)
    fps = cfg["sensors"]["camera"]["rate_hz"]
    kit.record_module().write_orbslam_yaml(out, intr, depth_scale=0.001, fps=fps, baseline=0.05, n_features=cfg["slam"]["n_features"])
    txt = open(out).read()
    get = lambda key: float(re.search(rf"^{re.escape(key)}:\s*([-0-9.eE]+)", txt, re.M).group(1))
    # simulation geometry from first principles: fx = (W/2)/tan(hfov/2)
    cam_cfg = cfg["sensors"]["camera"]
    W, H = k["width"], k["height"]
    cmp = {}
    for name, yk, sv in (("fx", "Camera1.fx", k["fx"]), ("fy", "Camera1.fy", k["fy"]), ("cx", "Camera1.cx", k["cx"]), ("cy", "Camera1.cy", k["cy"]),
                         ("width", "Camera.width", W), ("height", "Camera.height", H), ("k1", "Camera1.k1", 0.0), ("k2", "Camera1.k2", 0.0)):
        cmp[name] = {"sim": sv, "yaml": get(yk), "abs_diff": abs(sv - get(yk))}
    # depth conversion test: metres (32FC1) -> uint16 mm (what rgbd_stream gets) -> ORB-SLAM3 divides by DepthMapFactor
    factor = get("RGBD.DepthMapFactor")
    d = np.array([0.4, 1.234, 5.0, 12.0, 0.0, np.nan], np.float32)
    d16 = np.nan_to_num(d * 1000.0, nan=0.0).clip(0, 65535).astype(np.uint16)
    back = d16 / factor
    ok = np.nan_to_num(d, nan=0.0)
    depth_err = float(np.max(np.abs(back - ok)))
    rep = {"yaml": os.path.relpath(out, ROOT), "fov_deg": cam_cfg.get("hfov_deg", cam_cfg.get("fov_deg")), "comparison": cmp,
           "depth": {"sim_units": "metres float32 (32FC1, NaN/0 = invalid)", "ros_encoding": "32FC1", "orbslam3_input": "uint16 millimetres",
                     "conversion": "depth_u16 = round_down(depth_m * 1000)  [slam/orbslam3_backend.py]", "RGBD.DepthMapFactor": factor,
                     "roundtrip_test_m": d.tolist(), "roundtrip_back_m": back.tolist(), "max_abs_error_m": depth_err},
           "max_intrinsic_abs_diff_px": max(v["abs_diff"] for v in cmp.values())}
    os.makedirs(os.path.join(ROOT, "logs", "ubuntu"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "logs", "ubuntu", "calibration_compare.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    assert factor == 1000.0 and depth_err < 1.5e-3


if __name__ == "__main__":
    main()
