"""
Detector-compatible simulated survivors.

The previous blocky proxies were not recognised by a COCO-trained YOLO. These figures use human
proportions (1.70 m), smooth ellipsoid/capsule body parts, clothing textures and a textured face/hair
head, in three postures (standing / sitting / lying). They are ordinary static PyBullet bodies:
the detector still works on RGB pixels only; nothing is injected.

Every body part is registered with a *surface temperature* (skin ~34 C, clothing ~29 C) in the returned
`parts` list. That registry is used ONLY by the simulated thermal camera (sim/thermal_camera.py) to render
an image; it is never given to the autonomy stack.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pybullet as p
import pybullet_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX_DIR = os.path.join(ROOT, "data", "textures")
SPHERE = os.path.join(pybullet_data.getDataPath(), "textured_sphere_smooth.obj")

SKIN_T, CLOTH_T = 34.0, 29.0
SHIRTS = [(0.85, 0.30, 0.10), (0.15, 0.35, 0.75), (0.75, 0.10, 0.12), (0.90, 0.80, 0.15), (0.20, 0.60, 0.30)]
PANTS = [(0.12, 0.14, 0.30), (0.25, 0.25, 0.28), (0.30, 0.22, 0.15)]
SKINS = [(0.87, 0.68, 0.55), (0.72, 0.52, 0.38), (0.55, 0.38, 0.27)]


def _face_texture(idx):
    """Equirectangular head texture: hair on top, skin, eyes, brows, nose, mouth. Face looks along +x of the head."""
    import cv2
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, f"face_{idx % 3}.png")
    if os.path.exists(path):
        return path
    W, H = 512, 256
    skin = np.array(SKINS[idx % 3][::-1]) * 255
    img = np.zeros((H, W, 3), np.uint8) + skin.astype(np.uint8)
    hair = (25, 20, 18) if idx % 2 == 0 else (40, 60, 90)
    cv2.rectangle(img, (0, 0), (W, int(H * 0.36)), hair, -1)               # top of head (v small) = hair
    cv2.rectangle(img, (0, int(H * 0.36)), (W, int(H * 0.44)), hair, -1)
    cx = W // 4                                                            # face centre column (u = 0.25 -> +x)
    for sx in (-1, 1):
        cv2.ellipse(img, (cx + sx * 26, int(H * 0.52)), (12, 7), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(img, (cx + sx * 26, int(H * 0.52)), 5, (40, 30, 20), -1)
        cv2.line(img, (cx + sx * 40, int(H * 0.45)), (cx + sx * 14, int(H * 0.46)), hair, 3)
    cv2.line(img, (cx, int(H * 0.55)), (cx, int(H * 0.66)), (90, 110, 150), 2)
    cv2.ellipse(img, (cx, int(H * 0.76)), (18, 6), 0, 0, 180, (60, 60, 150), 3)
    cv2.imwrite(path, img)
    return path


def _cloth_texture(name, rgb, seed):
    import cv2
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, f"{name}.png")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 3), np.float32) + np.array(rgb[::-1], np.float32) * 255
    img += rng.normal(0, 10, img.shape)
    for k in range(0, 64, 8):                                             # faint weave / folds
        img[k:k + 1] *= 0.85
    cv2.imwrite(path, np.clip(img, 0, 255).astype(np.uint8))
    return path


def _R(rpy):
    return np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rpy))).reshape(3, 3)


class _Builder:
    def __init__(self, pb, origin, yaw, tilt, lift):
        self.pb, self.origin = pb, np.array(origin, float)
        self.R = _R([0, tilt, yaw])
        self.lift = lift
        self.parts = []                    # (body_id, temperature_C)

    def _place(self, lp, lrpy):
        pos = self.R @ np.array(lp) + self.origin + np.array([0, 0, self.lift])
        R = self.R @ _R(lrpy)
        q = p.getQuaternionFromRotationMatrix if hasattr(p, "getQuaternionFromRotationMatrix") else None
        # rotation matrix -> quaternion (xyzw)
        from common.frames import quat_from_R
        w, x, y, z = quat_from_R(R)
        return pos.tolist(), [x, y, z, w]

    def ellipsoid(self, lp, radii, rgb, temp, tex=None, lrpy=(0, 0, 0)):
        pos, q = self._place(lp, lrpy)
        vis = p.createVisualShape(p.GEOM_MESH, fileName=SPHERE, meshScale=list(radii), rgbaColor=[*rgb, 1], physicsClientId=self.pb)
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=float(np.mean(radii)), physicsClientId=self.pb)
        b = p.createMultiBody(0, col, vis, pos, q, physicsClientId=self.pb)
        if tex:
            p.changeVisualShape(b, -1, textureUniqueId=p.loadTexture(tex, physicsClientId=self.pb), rgbaColor=[1, 1, 1, 1], physicsClientId=self.pb)
        self.parts.append((b, temp))
        return b

    def capsule(self, a, b_, radius, rgb, temp, tex=None):
        """Capsule between local points a and b_."""
        a, b_ = np.array(a, float), np.array(b_, float)
        d = b_ - a
        L = float(np.linalg.norm(d))
        mid = 0.5 * (a + b_)
        u = d / L
        z = np.array([0, 0, 1.0])
        ax = np.cross(z, u)
        n = np.linalg.norm(ax)
        if n < 1e-9:
            Rl = np.eye(3) if u[2] > 0 else np.diag([1, -1, -1.0])
        else:
            ax /= n
            ang = math.acos(max(-1, min(1, float(z @ u))))
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rl = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
        from common.frames import quat_from_R
        pos = (self.R @ mid + self.origin + np.array([0, 0, self.lift])).tolist()
        w, x, y, zq = quat_from_R(self.R @ Rl)
        h = max(L - 2 * radius, 0.01)
        vis = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=h, rgbaColor=[*rgb, 1], physicsClientId=self.pb)
        col = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=h, physicsClientId=self.pb)
        bid = p.createMultiBody(0, col, vis, pos, [x, y, zq, w], physicsClientId=self.pb)
        if tex:
            p.changeVisualShape(bid, -1, textureUniqueId=p.loadTexture(tex, physicsClientId=self.pb), rgbaColor=[1, 1, 1, 1], physicsClientId=self.pb)
        self.parts.append((bid, temp))
        return bid


def build_human(pb, xy, posture, yaw_deg, idx):
    """Create one survivor. Returns {'parts': [(body_id, temp_C)], 'xy': ...}. Feet/back rest on the ground (z=0)."""
    shirt, pants, skin = SHIRTS[idx % 5], PANTS[idx % 3], SKINS[idx % 3]
    tshirt = _cloth_texture(f"shirt_{idx % 5}", shirt, idx)
    tpants = _cloth_texture(f"pants_{idx % 3}", pants, 10 + idx)
    tface = _face_texture(idx)
    yaw = math.radians(yaw_deg)
    if posture == "lying":
        tilt, lift = math.pi / 2, 0.14        # body axis (local +z) becomes horizontal; back on the ground
    else:
        tilt, lift = 0.0, 0.0
    B = _Builder(pb, (xy[0], xy[1], 0.0), yaw, tilt, lift)
    sit = posture == "sitting"
    zoff = 0.0
    if sit:                                    # hips at z=0.16 (seated on the ground), legs forward
        hip = np.array([0, 0, 0.17])
        knee_l, knee_r = np.array([0.42, 0.10, 0.16]), np.array([0.42, -0.10, 0.16])
        foot_l, foot_r = np.array([0.80, 0.12, 0.10]), np.array([0.80, -0.12, 0.10])
    else:
        hip = np.array([0, 0, 0.92])
        knee_l, knee_r = np.array([0.02, 0.10, 0.50]), np.array([0.0, -0.10, 0.50])
        foot_l, foot_r = np.array([0.03, 0.12, 0.08]), np.array([0.0, -0.12, 0.08])
    hipL, hipR = hip + np.array([0, 0.09, 0]), hip + np.array([0, -0.09, 0])
    B.capsule(hipL, knee_l, 0.075, pants, CLOTH_T, tpants)             # thighs
    B.capsule(hipR, knee_r, 0.075, pants, CLOTH_T, tpants)
    B.capsule(knee_l, foot_l + np.array([0, 0, 0.06]), 0.062, pants, CLOTH_T, tpants)   # shins
    B.capsule(knee_r, foot_r + np.array([0, 0, 0.06]), 0.062, pants, CLOTH_T, tpants)
    B.ellipsoid(foot_l + np.array([0.06, 0, 0]), (0.13, 0.055, 0.045), (0.08, 0.08, 0.08), CLOTH_T)     # shoes
    B.ellipsoid(foot_r + np.array([0.06, 0, 0]), (0.13, 0.055, 0.045), (0.08, 0.08, 0.08), CLOTH_T)
    B.ellipsoid(hip + np.array([0, 0, 0.02]), (0.12, 0.17, 0.11), pants, CLOTH_T, tpants)                # pelvis
    torso_c = hip + np.array([0, 0, 0.32])
    B.ellipsoid(torso_c, (0.13, 0.19, 0.29), shirt, CLOTH_T, tshirt)                                     # torso
    sh_l, sh_r = torso_c + np.array([0, 0.22, 0.20]), torso_c + np.array([0, -0.22, 0.20])
    if sit:
        elbow_l, elbow_r = sh_l + np.array([0.10, 0.06, -0.22]), sh_r + np.array([0.10, -0.06, -0.22])
        hand_l, hand_r = elbow_l + np.array([0.20, 0.0, -0.10]), elbow_r + np.array([0.20, 0.0, -0.10])
    else:
        elbow_l, elbow_r = sh_l + np.array([0.0, 0.07, -0.27]), sh_r + np.array([0.0, -0.07, -0.27])
        hand_l, hand_r = elbow_l + np.array([0.03, 0.04, -0.25]), elbow_r + np.array([0.03, -0.04, -0.25])
    B.capsule(sh_l, elbow_l, 0.05, shirt, CLOTH_T, tshirt)             # upper arms (sleeves)
    B.capsule(sh_r, elbow_r, 0.05, shirt, CLOTH_T, tshirt)
    B.capsule(elbow_l, hand_l, 0.042, skin, SKIN_T)                    # forearms (bare)
    B.capsule(elbow_r, hand_r, 0.042, skin, SKIN_T)
    B.ellipsoid(hand_l, (0.045, 0.035, 0.06), skin, SKIN_T)
    B.ellipsoid(hand_r, (0.045, 0.035, 0.06), skin, SKIN_T)
    neck = torso_c + np.array([0, 0, 0.30])
    B.capsule(neck - np.array([0, 0, 0.05]), neck + np.array([0, 0, 0.06]), 0.05, skin, SKIN_T)
    # head: sphere axis is +y in the mesh -> rotate so the pole axis is body-z; face column faces +x
    B.ellipsoid(neck + np.array([0.01, 0, 0.14]), (0.095, 0.105, 0.120), skin, SKIN_T, tface, lrpy=(math.pi / 2, 0, 0))
    return {"parts": B.parts, "xy": list(xy), "posture": posture}
