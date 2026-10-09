"""
SYNTHETIC thermal camera (not radiometrically accurate).

PyBullet has no thermal radiation model. This sensor renders the scene through the *real* camera geometry of
the thermal camera link (rigidly mounted on the vehicle, its own intrinsics and extrinsics) and converts what
it sees into a temperature image using a per-object surface-temperature table that the world builder created
(skin ~34 C, clothing ~29 C, ground/rubble ~ambient +- sun/shade, burning debris hundreds of C, sky cold).
Then it applies optics blur and NETD noise.

Output: float32 (H, W) temperature in degrees Celsius (encoding 32FC1 on ROS). Humans are warmer than the
background; hot hazards saturate. The temperature table is used only inside this sensor to synthesise an image;
autonomy never sees it and must judge from the pixels.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pybullet as p


class ThermalSensor:
    def __init__(self, sim, thermal_table: dict):
        self.pb, self.vehicle, self.model = sim.pb, sim.vehicle, sim.model
        c = sim.cfg["sensors"]["thermal"]
        self.W, self.H = int(c["width"]), int(c["height"])
        self.vfov = float(c["vfov_deg"])
        self.near, self.far = float(c["near_m"]), float(c["far_m"])
        self.netd, self.blur = float(c["netd_c"]), float(c["blur_sigma_px"])
        self.ambient, self.sky = float(c["ambient_c"]), float(c["sky_c"])
        self.gvar = float(c["ground_variation_c"])
        self.rng = np.random.default_rng(c.get("seed", 0))
        self.fy = (self.H / 2.0) / math.tan(math.radians(self.vfov) / 2.0)
        self.fx = self.fy
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.hfov = math.degrees(2 * math.atan((self.W / 2.0) / self.fx))
        self._proj = p.computeProjectionMatrixFOV(self.vfov, self.W / self.H, self.near, self.far)
        n = (max(thermal_table) + 2) if thermal_table else 2
        self.lut = np.full(n + 1, self.ambient, np.float32)
        for b, t in thermal_table.items():
            self.lut[b] = t
        self.hot = set(b for b, t in thermal_table.items() if t > 100)

    def intrinsics(self):
        return {"width": self.W, "height": self.H, "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "vfov_deg": self.vfov, "hfov_deg": self.hfov, "near_m": self.near, "far_m": self.far}

    def _pose_world_truth(self):
        st = self.vehicle.read_state()
        return st.pos_enu + st.R_enu_flu @ self.model.thermal_optical_pos_flu, st.R_enu_flu @ self.model.thermal_R_body_optical

    def capture(self):
        pos, R_wo = self._pose_world_truth()
        view = p.computeViewMatrix(pos.tolist(), (pos + R_wo[:, 2]).tolist(), (-R_wo[:, 1]).tolist(), physicsClientId=self.pb)
        img = p.getCameraImage(self.W, self.H, view, self._proj, renderer=p.ER_TINY_RENDERER,
                               flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=self.pb)
        rgb = np.asarray(img[2], np.uint8).reshape(self.H, self.W, 4)[:, :, :3].astype(np.float32) / 255.0
        seg = np.asarray(img[4], np.int64).reshape(self.H, self.W)
        body = np.where(seg >= 0, seg & 0xFFFFFF, -1)
        body = np.where(body >= len(self.lut), -1, body)
        T = np.where(body >= 0, self.lut[np.clip(body, 0, len(self.lut) - 1)], self.sky).astype(np.float32)
        # sun / shade modulation on ambient-temperature surfaces, from scene brightness
        lum = rgb.mean(axis=2)
        amb = np.abs(T - self.ambient) < 0.01
        T[amb] += (lum[amb] - 0.5) * 2.0 * self.gvar
        # optics: blur, then detector noise
        if self.blur > 0:
            T = cv2.GaussianBlur(T, (0, 0), self.blur)
        T = T + self.rng.normal(0.0, self.netd, T.shape).astype(np.float32)
        return T


def to_display(T, lo=5.0, hi=45.0):
    """Celsius -> 8-bit false colour (inferno) for panels/video. Values above hi saturate."""
    g = np.clip((T - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
