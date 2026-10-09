"""
Simulated RGB-D camera, rigidly mounted on the vehicle.

    rgb, depth = rgbd.capture()

    rgb    : uint8   (H, W, 3)  RGB
    depth  : float32 (H, W)     metres along the optical axis (z-depth, RealSense convention);
                                NaN where there is no valid return (< near, >= far)

Camera model
------------
Pinhole, square pixels. PyBullet's projection is defined by the VERTICAL field of view, so the
configuration stores `vfov_deg` and everything else is derived from it:

    fy = (H/2) / tan(vfov/2),   fx = fy,   cx = W/2,   cy = H/2      (pixel-centre = +0.5 convention)
    hfov = 2 atan( (W/2) / fx )

Frames
------
rgbd_camera_link          : x forward, y left, z up  (URDF frame, mounted with the URDF extrinsics)
rgbd_camera_optical_link  : z forward, x right, y down (REP-103 optical frame; image x -> right, y -> down)
A 3-D point in the optical frame is  P = depth * [ (u-cx)/fx, (v-cy)/fy, 1 ].

The camera pose is derived from the *physically simulated* vehicle body each time capture() is
called (body pose x static URDF extrinsics). It is NOT a separate teleported camera and it does not
provide any pose to the algorithm. `extrinsics_body_optical()` returns only the static calibration.
`_pose_world_truth()` is for evaluation/tests only.
Optional stereo-like depth noise: sigma_z = depth_noise_sigma_per_m2 * z^2.
"""
from __future__ import annotations

import math

import numpy as np
import pybullet as p


class RGBDSensor:
    def __init__(self, sim_or_client, body_id=None, cfg=None, model=None, vehicle=None):
        """Preferred: RGBDSensor(uav_sim). Legacy-style args (client, body, cfg, model, vehicle) also work."""
        if hasattr(sim_or_client, "vehicle"):
            s = sim_or_client
            self.pb, self.vehicle, self.model, cfg = s.pb, s.vehicle, s.model, s.cfg
        else:
            self.pb, self.vehicle, self.model = sim_or_client, vehicle, model
        cam = cfg["sensors"]["camera"]
        self.W, self.H = int(cam["width"]), int(cam["height"])
        self.vfov = float(cam["vfov_deg"])
        self.near, self.far = float(cam["near_m"]), float(cam["far_m"])
        self.noise = float(cam.get("depth_noise_sigma_per_m2", 0.0))
        self.rate_hz = float(cam["rate_hz"])
        self.rng = np.random.default_rng(cam.get("seed", 0))
        self.fy = (self.H / 2.0) / math.tan(math.radians(self.vfov) / 2.0)
        self.fx = self.fy
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.hfov = math.degrees(2 * math.atan((self.W / 2.0) / self.fx))
        self._proj = p.computeProjectionMatrixFOV(self.vfov, self.W / self.H, self.near, self.far)

    # -------------------------------------------------------------- calibration (legitimate to expose)
    def intrinsics(self):
        return {"width": self.W, "height": self.H, "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "K": [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], "vfov_deg": self.vfov,
                "hfov_deg": self.hfov, "near_m": self.near, "far_m": self.far, "distortion": [0, 0, 0, 0, 0]}

    def extrinsics_body_optical(self):
        """Static transform optical -> body(FLU, origin at CG): (R 3x3, t 3)."""
        return self.model.cam_R_body_optical.copy(), self.model.cam_optical_pos_flu.copy()

    # -------------------------------------------------------------- evaluation only
    def _pose_world_truth(self):
        st = self.vehicle.read_state()
        R_wo = st.R_enu_flu @ self.model.cam_R_body_optical
        pos = st.pos_enu + st.R_enu_flu @ self.model.cam_optical_pos_flu
        return pos, R_wo

    # -------------------------------------------------------------- capture
    def capture(self):
        pos, R_wo = self._pose_world_truth()
        forward, down = R_wo[:, 2], R_wo[:, 1]
        view = p.computeViewMatrix(pos.tolist(), (pos + forward).tolist(), (-down).tolist(), physicsClientId=self.pb)
        img = p.getCameraImage(self.W, self.H, view, self._proj, renderer=p.ER_TINY_RENDERER,
                               flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=self.pb)
        rgb = np.asarray(img[2], dtype=np.uint8).reshape(self.H, self.W, 4)[:, :, :3].copy()
        buf = np.asarray(img[3], dtype=np.float32).reshape(self.H, self.W)
        z = (self.far * self.near) / (self.far - (self.far - self.near) * buf)
        z = z.astype(np.float32)
        invalid = (buf >= 0.9999) | (z < self.near) | (z >= self.far)
        sky = buf >= 0.9999                                            # nothing rendered here: replace PyBullet's white with a sky gradient
        if sky.any():
            g = np.linspace(0.0, 1.0, self.H, dtype=np.float32)[:, None]
            grad = np.stack([150 + 70 * g, 185 + 45 * g, 225 + 25 * g], axis=2).repeat(self.W, axis=1)
            rgb[sky] = grad[sky].astype(np.uint8)
        if self.noise > 0:
            z = z + self.rng.normal(0.0, 1.0, z.shape).astype(np.float32) * (self.noise * z * z)
            invalid |= (z < self.near) | (z >= self.far)      # noisy returns outside the sensor range are invalid too
        z[invalid] = np.nan
        return rgb, z
