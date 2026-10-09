"""
RGB + thermal fusion and thermal hazard detection.

The RGB camera is the primary sensor. The thermal image (SYNTHETIC in simulation, degrees Celsius, 32FC1) is used to
  * support or contradict an RGB person detection: the 3-D point localised from RGB + depth is projected into the thermal camera
    with the static extrinsics from config `sensor_extrinsics`; a warm blob at that spot (skin/clothing temperature, clearly warmer
    than the local background) raises the confidence, a cold spot in view lowers it. Nothing is invented when the thermal image
    shows nothing.
  * detect hot hazards (fire / hot debris): connected hot blobs above `hazard_fire_min_c`, localised by intersecting the thermal ray
    with the registered depth image.
"""
from __future__ import annotations

import cv2
import numpy as np


def _R(rows):
    return np.array(rows, float).reshape(3, 3)


class ThermalFusion:
    def __init__(self, cfg: dict, K_rgb: dict, K_th: dict):
        self.c = cfg["perception"]
        ext = cfg["sensor_extrinsics"]
        self.R_rgb, self.t_rgb = _R(ext["rgbd_optical"]["R_body_from_optical"]), np.array(ext["rgbd_optical"]["xyz_body_flu_m"], float)
        self.R_th, self.t_th = _R(ext["thermal_optical"]["R_body_from_optical"]), np.array(ext["thermal_optical"]["xyz_body_flu_m"], float)
        self.Krgb, self.Kth = K_rgb, K_th

    # ------------------------------------------------------------------ geometry
    def rgb_point_to_thermal(self, P_rgb):
        """3-D point in the RGB optical frame -> (u, v, Z) in the thermal camera (None if behind the camera)."""
        P_body = self.R_rgb @ P_rgb + self.t_rgb
        P_th = self.R_th.T @ (P_body - self.t_th)
        if P_th[2] <= 0.1:
            return None
        return (self.Kth["fx"] * P_th[0] / P_th[2] + self.Kth["cx"], self.Kth["fy"] * P_th[1] / P_th[2] + self.Kth["cy"], float(P_th[2]))

    def thermal_ray_to_rgb_point(self, u, v, depth_m):
        """Point in the RGB optical frame where the thermal ray through pixel (u,v) meets the registered depth image."""
        d_th = np.array([(u - self.Kth["cx"]) / self.Kth["fx"], (v - self.Kth["cy"]) / self.Kth["fy"], 1.0])
        H, W = depth_m.shape
        for r in np.arange(1.0, 14.0, 0.25):
            P_body = self.R_th @ (d_th * r) + self.t_th
            P_rgb = self.R_rgb.T @ (P_body - self.t_rgb)
            if P_rgb[2] <= 0.2:
                continue
            ur = self.Krgb["fx"] * P_rgb[0] / P_rgb[2] + self.Krgb["cx"]
            vr = self.Krgb["fy"] * P_rgb[1] / P_rgb[2] + self.Krgb["cy"]
            if not (0 <= ur < W and 0 <= vr < H):
                continue
            z = depth_m[int(vr), int(ur)]
            if np.isfinite(z) and abs(z - P_rgb[2]) < 0.35:
                return P_rgb
        return None

    # ------------------------------------------------------------------ person support
    def person_thermal_check(self, P_rgb, T):
        """Warm-body evidence at the projected 3-D location. Returns dict(available, in_view, tmax, bg, contrast, support)."""
        c = self.c
        pr = self.rgb_point_to_thermal(P_rgb)
        H, W = T.shape
        if pr is None:
            return {"available": True, "in_view": False, "support": False}
        u, v, Z = pr
        if not (0 <= u < W and 0 <= v < H):
            return {"available": True, "in_view": False, "support": False}
        half = int(max(c["thermal_window_px"], 0.5 * self.Kth["fx"] * 0.4 / Z))     # ~ torso width
        u0, u1, v0, v1 = int(max(u - half, 0)), int(min(u + half + 1, W)), int(max(v - half, 0)), int(min(v + half + 1, H))
        win = T[v0:v1, u0:u1]
        R2 = half * 3
        ring = T[int(max(v - R2, 0)):int(min(v + R2 + 1, H)), int(max(u - R2, 0)):int(min(u + R2 + 1, W))]
        bg = float(np.median(ring))
        tmax = float(np.percentile(win, 90))
        contrast = tmax - bg
        support = c["thermal_human_min_c"] <= tmax <= c["thermal_human_max_c"] and contrast >= c["thermal_min_contrast_c"]
        return {"available": True, "in_view": True, "tmax": tmax, "bg": bg, "contrast": contrast, "support": bool(support),
                "uv": (float(u), float(v))}

    def fuse_confidence(self, conf, chk):
        c = self.c
        if chk.get("in_view") and chk.get("support"):
            return min(1.0, conf + c["thermal_boost"])
        if chk.get("in_view"):
            return max(0.0, conf - c["thermal_penalty"])
        return conf

    # ------------------------------------------------------------------ hazards
    def hot_hazards(self, T, depth_m, bgr=None):
        c = self.c
        hot = (T > c["hazard_fire_min_c"]).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(hot, connectivity=8)
        out = []
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if bgr is not None else None
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < c["hazard_min_blob_px"]:
                continue
            u, v = cent[i]
            w_px = float(stats[i, cv2.CC_STAT_WIDTH])
            P = self.thermal_ray_to_rgb_point(u, v, depth_m)
            if P is None:
                continue
            Z = float(P[2])
            radius = max(0.6, 0.5 * w_px * Z / self.Kth["fx"] + 0.3)
            tmax = float(T[lab == i].max())
            rgb_support = False
            if hsv is not None:
                m = cv2.inRange(hsv, np.array(c["hazard_fire_rgb_hsv_lo"]), np.array(c["hazard_fire_rgb_hsv_hi"]))
                ur = int(self.Krgb["fx"] * P[0] / P[2] + self.Krgb["cx"])
                vr = int(self.Krgb["fy"] * P[1] / P[2] + self.Krgb["cy"])
                r = int(0.6 * self.Krgb["fx"] * radius / Z) + 6
                sub = m[max(vr - r, 0):vr + r, max(ur - r, 0):ur + r]
                rgb_support = bool(sub.size and (sub > 0).mean() > 0.05)
            out.append({"kind": "fire", "xyz_cam": P, "radius_m": radius, "temp_c": tmax, "thermal_support": True,
                        "rgb_support": rgb_support, "blob_px": area, "range": Z})
        return out
