"""
Calibration 9 + 10: camera intrinsics and camera-to-body extrinsics, exercised end-to-end in the simulator.

  python -m calibration.camera_calibration

9  INTRINSICS  A checkerboard is rendered from many vehicle poses; OpenCV `calibrateCamera` recovers
   fx, fy, cx, cy which are compared with config/drone.yaml (sensors.camera). On the real RealSense use
   the same code path on saved checkerboard images (`--images DIR --board 9x6 --square 0.025`) - the
   factory intrinsics from the RealSense driver can also be used directly.
10 EXTRINSICS  A red marker at a known world position is detected in RGB, back-projected with the depth
   image and intrinsics, and transformed with the URDF body<-optical extrinsics; the result must equal
   the marker position in the body frame (from ground truth). On hardware do a hand-eye / target-based
   calibration (e.g. Kalibr camera-IMU) and update `components.rgbd_camera_link` in drone.yaml.

The teleporting of the vehicle below is an EVALUATION harness (calibration rig), not the autonomy path.
"""
import argparse
import math
import os

import cv2
import numpy as np
import pybullet as p

from calibration._util import write_report, ROOT
from common.frames import R_from_euler
from sim.config import load_config
from sim.rgbd_camera import RGBDSensor
from sim.sim_core import UavSim

TEX = os.path.join(ROOT, "data", "textures")
BOARD = (9, 6)          # inner corners
SQUARE = 0.25           # m


def _make_board_assets():
    os.makedirs(TEX, exist_ok=True)
    png, obj = os.path.join(TEX, "checkerboard.png"), os.path.join(TEX, "board_quad.obj")
    nx, ny = BOARD[0] + 1, BOARD[1] + 1
    px = 80
    img = np.zeros(((ny + 2) * px, (nx + 2) * px, 3), np.uint8) + 255          # white margin around the board
    for i in range(ny):
        for j in range(nx):
            if (i + j) % 2 == 0:
                img[(i + 1) * px:(i + 2) * px, (j + 1) * px:(j + 2) * px] = 0
    cv2.imwrite(png, img)
    W, H = (nx + 2) * SQUARE, (ny + 2) * SQUARE
    with open(obj, "w") as f:
        f.write(f"v {-W/2} 0 {-H/2}\nv {W/2} 0 {-H/2}\nv {W/2} 0 {H/2}\nv {-W/2} 0 {H/2}\n"
                "vt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\nvn 0 -1 0\nf 1/1/1 2/2/1 3/3/1 4/4/1\n")
    return png, obj


def _place_board(pb, center):
    png, obj = _make_board_assets()
    vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, meshScale=[1, 1, 1], physicsClientId=pb)
    # board plane is the x-z plane with its front face towards -y ... rotate so it faces the vehicle (looks along +y north)
    body = p.createMultiBody(0, -1, vis, center, p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=pb)
    tex = p.loadTexture(png, physicsClientId=pb)
    p.changeVisualShape(body, -1, textureUniqueId=tex, rgbaColor=[1, 1, 1, 1], physicsClientId=pb)
    return body


def intrinsics_calibration(n_views=14, seed=0):
    cfg = load_config()
    s = UavSim(cfg, world=False)
    cam = RGBDSensor(s)
    center = (0.0, 6.0, 1.6)
    _place_board(s.pb, center)
    rng = np.random.default_rng(seed)
    objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE
    obj_pts, img_pts = [], []
    for k in range(n_views):
        # vehicle pose: 2.5-4.5 m from the board, varied lateral/vertical offset and yaw/pitch/roll
        d = rng.uniform(3.0, 5.0)
        pos = np.array([rng.uniform(-1.0, 1.0), center[1] - d, rng.uniform(1.0, 2.4)])
        yaw_enu = math.pi / 2 + rng.uniform(-0.35, 0.35)
        rpy = [rng.uniform(-0.25, 0.25), rng.uniform(-0.15, 0.15), yaw_enu]
        p.resetBasePositionAndOrientation(s.vehicle.body, pos.tolist(), p.getQuaternionFromEuler(rpy), physicsClientId=s.pb)
        rgb, _ = cam.capture()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            obj_pts.append(objp); img_pts.append(corners)
    k = cam.intrinsics()
    rep = {"views_rendered": n_views, "views_with_board_detected": len(obj_pts), "config": {q: k[q] for q in ("fx", "fy", "cx", "cy")}}
    if len(obj_pts) >= 6:
        rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (cam.W, cam.H), None, None, flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
        rep.update({"estimated": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}, "distortion": dist.ravel().tolist(),
                    "reprojection_rms_px": rms,
                    "relative_error_fx_fy_pct": [abs(K[0, 0] / k["fx"] - 1) * 100, abs(K[1, 1] / k["fy"] - 1) * 100],
                    "principal_point_error_px": [K[0, 2] - k["cx"], K[1, 2] - k["cy"]]})
        rep["ok"] = bool(rep["relative_error_fx_fy_pct"][0] < 2 and rep["relative_error_fx_fy_pct"][1] < 2 and rms < 1.0)
    else:
        rep["ok"] = False
    s.close()
    return rep


def extrinsics_verification():
    cfg = load_config()
    s = UavSim(cfg, world=False)
    cam = RGBDSensor(s)
    cam.noise = 0.0
    R_bo, t_bo = cam.extrinsics_body_optical()
    marker_r = 0.15
    errs, rows = [], []
    rng = np.random.default_rng(1)
    for k in range(8):
        pos = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-1, 1), rng.uniform(2.5, 3.5)])
        rpy = [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), math.pi / 2 + rng.uniform(-0.6, 0.6)]
        p.resetBasePositionAndOrientation(s.vehicle.body, pos.tolist(), p.getQuaternionFromEuler(rpy), physicsClientId=s.pb)
        st = s.vehicle.read_state()
        # marker 3.5-4.5 m ahead of the camera, somewhat below the optical axis (camera looks 10 deg down)
        fwd_w = st.R_enu_flu @ np.array([1.0, 0.0, 0.0])
        marker_w = st.pos_enu + 4.0 * fwd_w + st.R_enu_flu @ np.array([0.0, rng.uniform(-0.8, 0.8), -1.0 - rng.uniform(0, 0.4)])
        m = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_SPHERE, radius=marker_r, rgbaColor=[1, 0, 0, 1], physicsClientId=s.pb),
                              marker_w.tolist(), physicsClientId=s.pb)
        rgb, depth = cam.capture()
        r_, g_, b_ = (rgb[:, :, i].astype(int) for i in range(3))
        mask = (r_ > 50) & (r_ > 3 * g_) & (r_ > 3 * b_)               # red-dominant incl. shaded side of the sphere
        p.removeBody(m, physicsClientId=s.pb)
        if mask.sum() < 20:
            continue
        vs, us = np.nonzero(mask)
        u, v = us.mean() + 0.5, vs.mean() + 0.5
        zc = depth[int(v), int(u)]                                     # surface depth at the marker centroid pixel
        z_surf = float(zc) if np.isfinite(zc) else float(np.nanmin(depth[mask]))
        ray = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
        P_surf = ray * z_surf
        P_o = P_surf + marker_r * ray / np.linalg.norm(ray)          # sphere centre = surface point + r along the ray
        P_b_est = R_bo @ P_o + t_bo
        P_b_true = st.R_enu_flu.T @ (marker_w - st.pos_enu)
        errs.append(float(np.linalg.norm(P_b_est - P_b_true)))
        rows.append({"est_body_flu": P_b_est.round(3).tolist(), "true_body_flu": P_b_true.round(3).tolist(), "error_m": errs[-1]})
    s.close()
    rep = {"views": rows, "mean_error_m": float(np.mean(errs)) if errs else None, "max_error_m": float(np.max(errs)) if errs else None,
           "urdf_extrinsics": {"camera_optical_position_in_body_flu_m": t_bo.tolist()}}
    rep["ok"] = bool(errs and np.max(errs) < 0.08)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", help="(hardware) directory of checkerboard images - see module docstring")
    a = ap.parse_args(argv)
    r9 = intrinsics_calibration()
    write_report("9_camera_intrinsics", r9)
    r10 = extrinsics_verification()
    write_report("10_camera_extrinsics", r10)
    return r9["ok"] and r10["ok"]


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
