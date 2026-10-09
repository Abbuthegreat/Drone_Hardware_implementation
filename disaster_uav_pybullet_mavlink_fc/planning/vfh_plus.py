"""
VFH+ local obstacle avoidance (Ulrich & Borenstein, 1998), operating on the OCCUPIED cells of the live metric map around the UAV.

Pipeline: active window -> polar obstacle density histogram (magnitude m = c^2 (a - b d)) with obstacle enlargement by the vehicle radius
-> smoothing -> binary histogram with hysteresis (low/high thresholds) -> candidate openings -> candidate directions ->
cost = w_target * |dir - target| + w_current * |dir - current heading| + w_prev * |dir - previous selection| -> best direction.
The output is a desired horizontal direction and a speed scale (slower when obstacles are close). It does NOT talk to the vehicle: the
navigation node turns it into /uav/cmd_vel, which goes through ROS -> MAVLink -> FC -> PyBullet.
Angles: mathematical, radians, counter-clockwise from +x (East), i.e. the ENU convention of the map frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

OCC = 100


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class VfhResult:
    direction: float                 # selected direction (rad, ENU math convention)
    speed_scale: float               # 0..1
    blocked_target: bool             # was the target direction blocked?
    deviation_deg: float             # selected - target direction
    nearest_obstacle_m: float
    free_openings: int
    reason: str = ""
    nearest_bearing: Optional[float] = None    # direction (rad, ENU) of the nearest occupied cell inside the active window


class VfhPlus:
    def __init__(self, cfg_vfh: dict, voxel_m: float, vehicle_radius_m: float):
        self.c = cfg_vfh
        self.v = voxel_m
        self.r = vehicle_radius_m
        self.n = int(cfg_vfh["n_sectors"])
        self.alpha = 2 * math.pi / self.n
        self.prev_dir: Optional[float] = None
        self.binary = np.zeros(self.n, bool)
        self.nearest_bearing: Optional[float] = None

    def histogram(self, grid, origin_xy, pos_xy):
        c = self.c
        R = c["active_radius_m"]
        h = np.zeros(self.n)
        nx, ny = grid.shape
        ci = int((pos_xy[0] - origin_xy[0]) / self.v)
        cj = int((pos_xy[1] - origin_xy[1]) / self.v)
        w = int(math.ceil(R / self.v)) + 1
        i0, i1, j0, j1 = max(ci - w, 0), min(ci + w + 1, nx), max(cj - w, 0), min(cj + w + 1, ny)
        sub = np.argwhere(grid[i0:i1, j0:j1] == OCC)
        nearest = float("inf")
        self.nearest_bearing = None
        for di, dj in sub:
            x = origin_xy[0] + (i0 + di + 0.5) * self.v
            y = origin_xy[1] + (j0 + dj + 0.5) * self.v
            dx, dy = x - pos_xy[0], y - pos_xy[1]
            d = math.hypot(dx, dy)
            if d > R or d < 1e-3:
                continue
            if d < nearest:
                nearest, self.nearest_bearing = d, math.atan2(dy, dx)
            m = 1.0 ** 2 * max(c["a"] - c["b"] * d, 0.0)
            beta = math.atan2(dy, dx)
            eps = math.asin(min(1.0, (self.r + 0.15) / max(d, self.r + 0.16)))     # obstacle enlargement
            k0 = int(math.floor(_wrap(beta - eps) / self.alpha))
            k1 = int(math.ceil(_wrap(beta - eps) / self.alpha + 2 * eps / self.alpha))
            for k in range(k0, k1 + 1):
                h[k % self.n] += m
        # smoothing (moving average over +-l sectors)
        l = int(c["smoothing"])
        if l > 0:
            ker = np.concatenate([np.arange(1, l + 1), [l + 1], np.arange(l, 0, -1)]).astype(float)
            ker /= ker.sum()
            h = np.convolve(np.concatenate([h[-l:], h, h[:l]]), ker, mode="valid")
        return h, nearest

    def choose(self, grid, origin_xy, pos_xy, target_dir, current_heading) -> VfhResult:
        c = self.c
        h, nearest = self.histogram(grid, origin_xy, pos_xy)
        b = self.binary
        hi, lo = c["threshold_high"], c["threshold_low"]
        b = np.where(h > hi, True, np.where(h < lo, False, b))                # hysteresis
        self.binary = b
        free = ~b
        target_k = int(round(_wrap(target_dir) / self.alpha)) % self.n
        blocked = b[target_k]
        if not free.any():
            return VfhResult(target_dir, 0.0, True, 0.0, nearest, 0, "all sectors blocked", self.nearest_bearing)
        # candidate openings (circular runs of free sectors)
        runs, k = [], 0
        start = None
        # rotate so that index 0 is a blocked sector to avoid wrap issues
        if free.all():
            runs = [(0, self.n - 1, True)]
        else:
            s0 = int(np.argmax(b))
            idx = [(s0 + t) % self.n for t in range(self.n)]
            cur = None
            for t, kk in enumerate(idx):
                if free[kk]:
                    cur = kk if cur is None else cur
                    last = kk
                    if t == self.n - 1 or not free[idx[t + 1]]:
                        runs.append((cur, last, False))
                        cur = None
        cands = []
        gap_min = int(c["min_gap_sectors"])
        for a, z, whole in runs:
            width = self.n if whole else ((z - a) % self.n) + 1
            if width < gap_min:
                continue
            if whole:
                cands.append(_wrap(target_dir))
                continue
            wide = width > 2 * gap_min
            ang_a, ang_z = a * self.alpha, ((a + width - 1) % self.n) * self.alpha
            if wide:
                cands.append(_wrap(ang_a + gap_min * self.alpha / 2 + self.alpha * 0.5))            # right border (safe offset)
                cands.append(_wrap(ang_z - gap_min * self.alpha / 2 - self.alpha * 0.5))            # left border
                if self._in_run(target_k, a, width):
                    cands.append(_wrap(target_dir))
            else:
                cands.append(_wrap(ang_a + (width - 1) * self.alpha / 2))
        if not cands:
            return VfhResult(target_dir, 0.0, True, 0.0, nearest, len(runs), "no opening wide enough", self.nearest_bearing)
        prev = self.prev_dir if self.prev_dir is not None else current_heading
        cost = [c["w_target"] * abs(_wrap(d - target_dir)) + c["w_current"] * abs(_wrap(d - current_heading)) + c["w_prev"] * abs(_wrap(d - prev))
                for d in cands]
        best = cands[int(np.argmin(cost))]
        self.prev_dir = best
        blockR = c["block_radius_m"]
        scale = 1.0 if nearest > 2 * blockR else max(0.3, (nearest - 0.6) / (2 * blockR - 0.6))
        if abs(_wrap(best - target_dir)) > math.radians(60):
            scale = min(scale, 0.5)
        return VfhResult(best, float(np.clip(scale, 0.0, 1.0)), bool(blocked), math.degrees(_wrap(best - target_dir)), nearest, len(runs),
                         "", self.nearest_bearing)

    def _in_run(self, k, a, width):
        return ((k - a) % self.n) < width
