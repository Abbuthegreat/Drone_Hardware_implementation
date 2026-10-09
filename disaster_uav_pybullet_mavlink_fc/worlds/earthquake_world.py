"""
Outdoor earthquake-disaster scene (PyBullet). Scenario `static` (default): static bodies only. Scenario `dynamic`
(config world.scenario or `--scenario dynamic`): adds the scripted world events of `world.dynamic_obstacles` (secondary collapse,
sliding debris; see worlds/dynamic_obstacles.py).

Contents: large ground, cracked/damaged road with offset slabs, collapsed buildings with leaning
wall fragments, rubble and broken concrete, fallen beams/slabs, utility poles with a fallen cable,
trees, boulders, wrecked vehicles, and several survivors (standing / sitting / lying).

The survivors are real PyBullet bodies, visible to the RGB-D camera. Their coordinates are returned
in `world_info["humans"]` for the SIMULATOR and the evaluation channel only - the autonomy stack is
never given them (see eval/ground_truth.py and tests/test_ground_truth_firewall.py).

All surfaces get procedural textures (data/textures/*.png, generated on first use) so that visual
odometry / ORB features exist.
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
import pybullet as p
import pybullet_data

from worlds.humans import build_human

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX_DIR = os.path.join(ROOT, "data", "textures")


# ------------------------------------------------------------------ textures
def _make_texture(name, base_rgb, seed, blotch=40, size=1024, scales=(2, 4, 8, 16, 32, 64)):
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, f"{name}.png")
    if os.path.exists(path):
        return path
    import cv2
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32) + np.array(base_rgb, np.float32)
    for scale, amp in zip(scales, (blotch * 0.9, blotch, blotch * 0.8, blotch * 0.6, blotch * 0.4, blotch * 0.3)):
        n = rng.normal(0, amp, (size // scale + 1, size // scale + 1, 1)).astype(np.float32)
        img += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC).reshape(size, size, 1)
    img += rng.normal(0, 14, (size, size, 1)).astype(np.float32)          # luminance-only grain (no colour speckle)
    for _ in range(40 * (size // 256) ** 2 // 4):                 # stains, pits and aggregate: corner-like features at many scales
        cx, cy = rng.integers(0, size, 2)
        rad = int(rng.integers(2, 10))
        tone = float(rng.uniform(0.35, 1.45))                        # stains / aggregate: brighter or darker versions of the surface tone (no random hues)
        cv2.circle(img, (int(cx), int(cy)), rad, tuple(float(np.clip(v * tone, 0, 255)) for v in base_rgb), -1)
    for _ in range(60 * (size // 256) ** 2):                  # cracks / speckles
        x0, y0 = rng.integers(0, size, 2)
        x1, y1 = np.clip([x0 + rng.integers(-40, 40), y0 + rng.integers(-40, 40)], 0, size - 1)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (30, 30, 30), 1)
    cv2.imwrite(path, np.clip(img, 0, 255).astype(np.uint8))
    return path


def _make_ground_texture(size=4096, seed=11):
    os.makedirs(TEX_DIR, exist_ok=True)
    path = os.path.join(TEX_DIR, "ground4096.png")
    if os.path.exists(path):
        return path
    import cv2
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size, 3), np.float32) + np.array((118, 140, 158), np.float32)      # BGR sandy soil
    for scale, amp in ((512, 28), (256, 24), (128, 22), (64, 20), (32, 18), (16, 16), (8, 14), (4, 12), (2, 10)):
        n = rng.normal(0, amp, (size // scale + 2, size // scale + 2, 1)).astype(np.float32)
        base += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC).reshape(size, size, 1)
    base += rng.normal(0, 10, (size, size, 1)).astype(np.float32)
    img = np.clip(base, 0, 255).astype(np.uint8)
    for rad_lo, rad_hi, count in ((1, 3, 90000), (3, 8, 30000), (8, 16, 1500)):              # pebbles and a few stones
        xs, ys = rng.integers(0, size, count), rng.integers(0, size, count)
        rs = rng.integers(rad_lo, rad_hi + 1, count)
        tones = rng.uniform(0.30, 1.35, count)
        cols = np.clip(np.array((118, 140, 158))[None, :] * tones[:, None] + rng.normal(0, 6, (count, 3)), 0, 255).astype(int)     # earth tones only
        for x, y, r, c in zip(xs, ys, rs, cols):
            cv2.circle(img, (int(x), int(y)), int(r), (int(c[0]), int(c[1]), int(c[2])), -1)
            if r > 11:
                cv2.circle(img, (int(x), int(y)), int(r), (70, 78, 86), 1)                   # soft edge, not black
    for _ in range(450):                                                                      # fine cracks / tyre marks
        x0, y0 = rng.integers(0, size, 2)
        pts = [(int(x0), int(y0))]
        for _ in range(int(rng.integers(3, 9))):
            pts.append((int(np.clip(pts[-1][0] + rng.integers(-90, 90), 0, size - 1)), int(np.clip(pts[-1][1] + rng.integers(-90, 90), 0, size - 1))))
        cv2.polylines(img, [np.array(pts, np.int32)], False, (52, 58, 66), int(rng.integers(1, 3)))
    cv2.imwrite(path, img)
    return path


class _Tex:
    def __init__(self, pb):
        self.pb = pb
        self.ids = {}
        specs = {"concrete": ((150, 148, 142), 1), "asphalt": ((70, 70, 74), 2), "dirt": ((110, 92, 70), 3),
                 "brick": ((150, 90, 70), 4), "wood": ((120, 85, 55), 5), "rust": ((120, 70, 50), 6)}
        for k, (rgb, seed) in specs.items():
            self.ids[k] = p.loadTexture(_make_texture(k, rgb[::-1], seed), physicsClientId=pb)
        # large, non-repeating ground texture (~2 cm/texel over 90 m): tonal patches, gravel, pebbles and stones so that ORB features
        # are distinctive (repetitive or featureless ground makes any visual odometry fail)
        self.ids["ground"] = p.loadTexture(_make_ground_texture(), physicsClientId=pb)

    def apply(self, body, name):
        p.changeVisualShape(body, -1, textureUniqueId=self.ids[name], rgbaColor=[1, 1, 1, 1], physicsClientId=self.pb)


# ------------------------------------------------------------------ primitives
def _box(pb, pos, half, orn=(0, 0, 0, 1), rgba=(0.4, 0.4, 0.4, 1)):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=pb)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=pb)
    return p.createMultiBody(0, col, vis, pos, orn, physicsClientId=pb)


def _cyl(pb, pos, radius, height, orn=(0, 0, 0, 1), rgba=(0.5, 0.5, 0.5, 1)):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=pb)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba, physicsClientId=pb)
    return p.createMultiBody(0, col, vis, pos, orn, physicsClientId=pb)


def _sphere(pb, pos, radius, rgba):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=pb)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=pb)
    return p.createMultiBody(0, col, vis, pos, physicsClientId=pb)


def _rot(rpy):
    return np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rpy))).reshape(3, 3)


# ------------------------------------------------------------------ survivors
_SHIRTS = [(0.90, 0.35, 0.10, 1), (0.15, 0.40, 0.80, 1), (0.80, 0.10, 0.12, 1), (0.95, 0.85, 0.15, 1), (0.20, 0.65, 0.30, 1)]
_SKIN = (0.85, 0.66, 0.50, 1)
_PANTS = (0.12, 0.14, 0.26, 1)


def _human(pb, xy, posture, yaw_deg, idx):
    """Blocky but human-proportioned figure (1.66 m standing). Parts are (kind, local pos, half-size/radius, rgba)."""
    shirt = _SHIRTS[idx % len(_SHIRTS)]
    if posture == "standing":
        parts = [("box", (0, 0.09, 0.43), (0.055, 0.055, 0.43), _PANTS), ("box", (0, -0.09, 0.43), (0.055, 0.055, 0.43), _PANTS),
                 ("box", (0, 0, 1.14), (0.10, 0.19, 0.28), shirt), ("sph", (0, 0, 1.55), 0.11, _SKIN),
                 ("box", (0, 0.25, 1.13), (0.045, 0.045, 0.27), shirt), ("box", (0, -0.25, 1.13), (0.045, 0.045, 0.27), shirt)]
        lift, tilt = 0.0, 0.0
    elif posture == "sitting":
        parts = [("box", (0.30, 0.09, 0.055), (0.30, 0.055, 0.055), _PANTS), ("box", (0.30, -0.09, 0.055), (0.30, 0.055, 0.055), _PANTS),
                 ("box", (0, 0, 0.39), (0.10, 0.19, 0.28), shirt), ("sph", (0, 0, 0.80), 0.11, _SKIN),
                 ("box", (0.05, 0.25, 0.30), (0.20, 0.045, 0.045), shirt), ("box", (0.05, -0.25, 0.30), (0.20, 0.045, 0.045), shirt)]
        lift, tilt = 0.0, 0.0
    else:  # lying: standing figure rotated so that local +z (head) points along +x, resting on the ground
        parts = [("box", (0, 0.09, 0.43), (0.055, 0.055, 0.43), _PANTS), ("box", (0, -0.09, 0.43), (0.055, 0.055, 0.43), _PANTS),
                 ("box", (0, 0, 1.14), (0.10, 0.19, 0.28), shirt), ("sph", (0, 0, 1.55), 0.11, _SKIN),
                 ("box", (0, 0.25, 1.13), (0.045, 0.045, 0.27), shirt), ("box", (0, -0.25, 1.13), (0.045, 0.045, 0.27), shirt)]
        lift, tilt = 0.11, math.pi / 2
    Rp = _rot([0, tilt, math.radians(yaw_deg)])
    Rq = np.array(p.getQuaternionFromEuler([0, tilt, math.radians(yaw_deg)]))
    ids = []
    for kind, lp, size, rgba in parts:
        w = Rp @ np.array(lp) + np.array([xy[0], xy[1], lift])
        if kind == "box":
            ids.append(_box(pb, w.tolist(), size, Rq.tolist(), rgba))
        else:
            ids.append(_sphere(pb, w.tolist(), size, rgba))
    return ids


# ------------------------------------------------------------------ scene
def build_earthquake_world(pb, cfg, seed=None):
    wcfg = cfg["world"]
    seed = wcfg.get("seed", 26) if seed is None else seed
    rnd = random.Random(seed)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=pb)
    tex = _Tex(pb)
    half = 0.5 * max(wcfg["size_m"]) + 5.0
    ground = _box(pb, (0, 0, -0.05), (half, half, 0.05), rgba=(1, 1, 1, 1))     # top surface at z = 0
    tex.apply(ground, "ground")
    p.changeDynamics(ground, -1, lateralFriction=0.9, physicsClientId=pb)

    humans = wcfg["humans"]
    spawn = cfg["spawn"]["xy_enu_m"]
    keep_out = [(h["xy"][0], h["xy"][1], 3.0) for h in humans] + [(spawn[0], spawn[1], 3.5)]
    keep_out += [(o["xyz"][0], o["xyz"][1], 3.0) for o in wcfg.get("demo_obstacles", [])]
    keep_out += [(h["xy"][0], h["xy"][1], 3.5) for h in wcfg.get("hazards", [])]
    thermal = {}                                   # body_id -> surface temperature [C], used ONLY by the simulated thermal camera

    def free(x, y, r=0.0):
        return all(math.hypot(x - kx, y - ky) > kr + r for kx, ky, kr in keep_out)

    def add(body, tname):
        tex.apply(body, tname)
        return body

    # --- main road (asphalt) with cracks / offset slabs
    add(_box(pb, (0, 0, -0.01), (40, 6, 0.03), rgba=(1, 1, 1, 1)), "asphalt")          # road surface at z = +0.02
    for x0, dz, dy in [(-18, 0.10, 0.0), (-3, -0.03, 0.5), (10, 0.18, -1.0), (24, 0.07, 1.2)]:
        q = p.getQuaternionFromEuler([0, 0, math.radians(rnd.uniform(-6, 6))])
        add(_box(pb, (x0, dy, 0.02 + dz * 0.5 - 0.03), (2.0, 5.0, 0.03 + dz * 0.5), q, (1, 1, 1, 1)), "asphalt")
    _box(pb, (0, 0, 0.022), (40, 0.12, 0.004), rgba=(0.85, 0.85, 0.3, 1))        # centre line

    # --- collapsed structures: pancaked floor slabs + leaning walls
    for (cx, cy), yaw in [((-14, 8), 15), ((12, 12), -7), ((-18, -10), 22), ((17, -13), -12)]:
        R = math.radians(yaw)
        add(_box(pb, (cx, cy, 0.5), (3.6, 2.6, 0.18), p.getQuaternionFromEuler([0.06, -0.05, R]), (1, 1, 1, 1)), "concrete")
        add(_box(pb, (cx + 0.5, cy, 1.05), (3.2, 2.3, 0.15), p.getQuaternionFromEuler([-0.12, 0.09, R + 0.1]), (1, 1, 1, 1)), "concrete")
        for k in range(3):                                             # broken wall segments
            ox, oy = rnd.uniform(-3, 3), rnd.uniform(-2.5, 2.5)
            add(_box(pb, (cx + ox, cy + oy, 1.0), (rnd.uniform(0.9, 1.6), 0.12, rnd.uniform(0.7, 1.3)),
                     p.getQuaternionFromEuler([rnd.uniform(-0.6, 0.6), rnd.uniform(-0.4, 0.4), rnd.uniform(-3.1, 3.1)]), (1, 1, 1, 1)),
                "brick")
    # one tall standing wall remnant / column (structural obstacle)
    add(_box(pb, (-6, -16, 2.0), (0.15, 2.0, 2.0), rgba=(1, 1, 1, 1)), "brick")
    add(_box(pb, (23, 8, 1.7), (0.4, 0.4, 1.7), rgba=(1, 1, 1, 1)), "concrete")

    # --- rubble / broken concrete
    n_placed = 0
    while n_placed < wcfg["debris_count"]:
        x, y = rnd.uniform(-32, 32), rnd.uniform(-30, 30)
        if abs(y) < 2.2 and abs(x) < 25:
            y += rnd.choice([-1, 1]) * rnd.uniform(2.5, 5.0)
        if not free(x, y, 0.8):
            continue
        sx, sy, sz = rnd.uniform(0.12, 0.65), rnd.uniform(0.12, 0.75), rnd.uniform(0.08, 0.45)
        orn = p.getQuaternionFromEuler([rnd.uniform(-0.7, 0.7), rnd.uniform(-0.7, 0.7), rnd.uniform(-math.pi, math.pi)])
        add(_box(pb, (x, y, sz), (sx, sy, sz), orn, (1, 1, 1, 1)), rnd.choice(["concrete", "concrete", "brick", "rust"]))
        n_placed += 1

    # --- fallen beams / slabs
    for pos, dims, eul in [((-5, 13, 0.55), (0.35, 3.8, 0.30), (0.2, 0.5, 0.2)), ((8, -7, 0.65), (0.40, 4.2, 0.35), (0.0, -0.6, 0.7)),
                           ((-12, -20, 0.6), (0.45, 3.5, 0.35), (0.4, 0.1, -0.8)), ((20, 5, 0.8), (3.5, 0.4, 0.40), (0.15, 0.5, 0.1))]:
        if free(pos[0], pos[1], 2.0):
            add(_box(pb, pos, dims, p.getQuaternionFromEuler(eul), (1, 1, 1, 1)), "concrete")

    # --- utility poles (+ one fallen pole with cable)
    for x, y in [(-7, 4), (5, 18), (-22, 4), (21, 18), (22, -20), (-24, -17)]:
        add(_cyl(pb, (x, y, 2.6), 0.16, 5.2, rgba=(1, 1, 1, 1)), "wood")
        add(_box(pb, (x, y, 5.0), (0.05, 0.9, 0.06), rgba=(1, 1, 1, 1)), "wood")
    add(_cyl(pb, (-3, -12, 0.35), 0.16, 5.5, p.getQuaternionFromEuler([0, math.pi / 2 - 0.12, 0.6]), (1, 1, 1, 1)), "wood")
    _cyl(pb, (0.5, -10.2, 0.05), 0.01, 7.0, p.getQuaternionFromEuler([0, math.pi / 2, 0.5]), rgba=(0.05, 0.05, 0.05, 1))

    # --- trees and boulders (natural features)
    for _ in range(14):
        x, y = rnd.uniform(-35, 35), rnd.uniform(-35, 35)
        if abs(y) < 7 and abs(x) < 40 or not free(x, y, 1.5):
            continue
        h = rnd.uniform(3.0, 5.5)
        add(_cyl(pb, (x, y, h / 2), 0.18, h, rgba=(1, 1, 1, 1)), "wood")
        _sphere(pb, (x, y, h + 0.6), rnd.uniform(1.2, 1.9), (rnd.uniform(0.1, 0.25), rnd.uniform(0.35, 0.55), 0.12, 1))
    for _ in range(8):
        x, y = rnd.uniform(-33, 33), rnd.uniform(-33, 33)
        if abs(y) < 7 or not free(x, y, 1.5):
            continue
        r = rnd.uniform(0.4, 0.9)
        add(_sphere(pb, (x, y, r * 0.6), r, (1, 1, 1, 1)), "concrete")

    # --- wrecked vehicles
    for x, y, yaw in [(-10, 5.5, 0.3), (14, -4.5, -0.8)]:
        q = p.getQuaternionFromEuler([0.1, 0.05, yaw])
        add(_box(pb, (x, y, 0.55), (2.2, 0.95, 0.4), q, (1, 1, 1, 1)), "rust")
        add(_box(pb, (x - 0.2, y, 1.1), (1.1, 0.85, 0.35), q, (1, 1, 1, 1)), "rust")

    obstacle_ids = {}                              # evaluation only: body id -> name (collision reports)
    # --- deterministic demo obstacles on the mission route (tall enough to intersect the flight altitude band)
    for o in wcfg.get("demo_obstacles", []):
        q = p.getQuaternionFromEuler([math.radians(a) for a in o["rpy_deg"]])
        if o["kind"] == "box":
            body = _box(pb, o["xyz"], o["half"], q, (1, 1, 1, 1))
        else:
            body = _cyl(pb, o["xyz"], o["radius"], o["height"], q, (1, 1, 1, 1))
        add(body, o.get("tex", "concrete"))
        obstacle_ids[body] = o["name"]

    # --- hazards: burning debris (flames are visual only; the hot-spot temperature is registered for the thermal camera)
    hazards = []
    for hz in wcfg.get("hazards", []):
        x, y = hz["xy"]
        r = hz["radius_m"]
        pile = []
        for k in range(6):
            ang = k * math.pi / 3
            b = _box(pb, (x + 0.6 * r * math.cos(ang), y + 0.6 * r * math.sin(ang), 0.25), (0.35, 0.25, 0.22),
                     p.getQuaternionFromEuler([0.3 * k, 0.2, ang]), (0.12, 0.10, 0.08, 1))
            pile.append(b)
            thermal[b] = 180.0
        flames = []
        for (dx, dy, h, rad, col) in [(0, 0, 0.9, 0.55, (1.0, 0.35, 0.02, 1)), (0.3, 0.2, 1.2, 0.35, (1.0, 0.6, 0.05, 1)),
                                      (-0.3, -0.1, 1.0, 0.4, (1.0, 0.45, 0.03, 1)), (0.0, 0.05, 1.5, 0.25, (1.0, 0.85, 0.2, 1))]:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=rad * r / 1.3, rgbaColor=list(col), physicsClientId=pb)
            fb = p.createMultiBody(0, -1, vis, (x + dx, y + dy, h * r / 1.3), physicsClientId=pb)
            flames.append(fb)
            thermal[fb] = float(hz["temp_c"])
        hazards.append({"id": hz["id"], "kind": hz["kind"], "xy_enu": [x, y], "radius_m": r, "temp_c": hz["temp_c"]})

    # --- dynamic scenario: world events (secondary collapse, sliding debris). Simulation side only.
    dynamic = None
    scenario = wcfg.get("scenario", "static")
    if scenario == "dynamic" and wcfg.get("dynamic_obstacles"):
        from worlds.dynamic_obstacles import DynamicObstacles
        dynamic = DynamicObstacles(pb, wcfg["dynamic_obstacles"],
                                   lambda pos, half, q, tname: add(_box(pb, pos, half, q, (1, 1, 1, 1)), tname))
        obstacle_ids.update(dynamic.body_names)
    elif scenario not in ("static", "dynamic"):
        raise ValueError(f"world.scenario must be static|dynamic, not {scenario}")

    # --- survivors (simulation-world only). Their temperature registry feeds the thermal camera only.
    human_ids = {}
    for i, h in enumerate(humans):
        hb = build_human(pb, h["xy"], h.get("posture", "standing"), h.get("yaw_deg", 0.0), i)
        human_ids[h["id"]] = [b for b, _ in hb["parts"]]
        for b, t in hb["parts"]:
            thermal[b] = t

    return {
        "humans": [{"id": h["id"], "xy_enu": list(h["xy"]), "posture": h.get("posture", "standing")} for h in humans],
        "hazards": hazards,
        "human_body_ids": human_ids,
        "thermal_c": thermal,
        "ground_body": ground,
        "obstacle_ids": obstacle_ids,
        "scenario": scenario,
        "dynamic": dynamic,
        "dynamic_obstacles": dynamic.describe() if dynamic else [],
        "description": "Outdoor earthquake-disaster scene: cracked road, collapsed structures, rubble, fallen beams, utility "
                       "poles, trees, wrecked vehicles, burning debris and survivors.",
    }
