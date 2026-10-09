"""
Live metric map built ONLY from RGB-D + the SLAM pose (no simulator data).

3-D log-odds voxel grid in the SLAM map frame (x East, y North, z Up, origin = take-off body position).
Ray casting: every sub-sampled depth pixel casts a ray from the camera; voxels along the ray are updated as FREE, the hit
voxel as OCCUPIED. Pixels with no return (sky / beyond range) only clear free space up to `free_ray_m` (conservative).
UNKNOWN is never assumed free: a cell is FREE only if enough voxels in the planning band were actually observed free.

Derived layers
  planning_grid(z_center)  2-D int8: 0 FREE, 100 OCCUPIED, -1 UNKNOWN for the altitude band the UAV flies in
  surface_classes()        2-D uint8 semantic surface classes from the highest occupied voxel: 0 unknown, 1 ground, 2 debris, 3 structure
"""
from __future__ import annotations

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 100
DYNAMIC = 90        # planning layer only: region remembered as containing a moving obstacle (A* avoids it; VFH+ reacts only to sensed OCCUPIED cells)
CLS_UNKNOWN, CLS_GROUND, CLS_DEBRIS, CLS_STRUCTURE = 0, 1, 2, 3


class VoxelMap3D:
    def __init__(self, cfg_map: dict):
        c = cfg_map
        self.c = c
        self.v = float(c["voxel_m"])
        x0, x1, y0, y1 = c["extent_local_m"]
        z0, z1 = c["z_range_m"]
        self.origin = np.array([x0, y0, z0], float)
        self.shape = (int(round((x1 - x0) / self.v)), int(round((y1 - y0) / self.v)), int(round((z1 - z0) / self.v)))
        self.L = np.zeros(self.shape, np.float32)
        self.seen = np.zeros(self.shape[:2], bool)
        self.n_frames = 0
        self.version = 0
        # moving-obstacle memory: a cell that was confidently OCCUPIED in the flight band and is later observed FREE contained something that moved
        # (collapse, sliding debris). Such regions stay blocked for planning for `hold_s`, because the forward camera cannot watch a sweeping obstacle
        # that leaves its field of view while the stored map still shows its old position.
        self.dm = c.get("dynamic_memory", {}) or {}
        self.dyn_state = np.zeros(self.shape[:2], bool)                    # hysteresis: True = confidently occupied
        self.dyn_until = np.full(self.shape[:2], -1.0, np.float32)
        self.t_now = 0.0
        self.z_band = None                                                 # flight altitude, set by planning_grid()
        self.n_dynamic_events = 0

    # ------------------------------------------------------------------ indexing
    def to_idx(self, pts):
        return np.floor((np.asarray(pts, float) - self.origin) / self.v).astype(np.int64)

    def in_bounds(self, idx):
        return (idx[:, 0] >= 0) & (idx[:, 0] < self.shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < self.shape[1]) & \
               (idx[:, 2] >= 0) & (idx[:, 2] < self.shape[2])

    def cell_center(self, ix, iy):
        return self.origin[0] + (ix + 0.5) * self.v, self.origin[1] + (iy + 0.5) * self.v

    # ------------------------------------------------------------------ integration
    def integrate(self, depth_m, K, T_map_cam, t=None):
        c = self.c
        stride = int(c["ray_stride_px"])
        H, W = depth_m.shape
        vs, us = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
        us, vs = us.ravel(), vs.ravel()
        z = depth_m[vs, us]
        valid_hit = np.isfinite(z) & (z > 0.3) & (z < c["max_ray_m"])
        dirs_c = np.stack([(us - K["cx"]) / K["fx"], (vs - K["cy"]) / K["fy"], np.ones(len(us))], 1)
        R, o = T_map_cam[:3, :3], T_map_cam[:3, 3]
        dirs_w = dirs_c @ R.T                                  # not normalised: |dir| * z_depth = range along the ray for z-depth
        norm = np.linalg.norm(dirs_w, axis=1)
        rng = np.where(valid_hit, z * norm, c["free_ray_m"])                # range along the (unit-normalised) ray
        u = dirs_w / norm[:, None]
        step = self.v * 0.5
        smax = int(np.ceil(c["max_ray_m"] / step))
        s = (np.arange(1, smax + 1) * step)[None, :]                        # (1, S)
        keep = s < (rng[:, None] - self.v * 0.75)                          # stop before the surface
        pts = o[None, None, :] + u[:, None, :] * s[:, :, None]              # (N, S, 3)
        free_pts = pts[keep]
        free_idx = self.to_idx(free_pts)
        free_idx = free_idx[self.in_bounds(free_idx)]
        hit_pts = o[None, :] + u[valid_hit] * rng[valid_hit][:, None]
        hit_idx = self.to_idx(hit_pts)
        hit_idx = hit_idx[self.in_bounds(hit_idx)]
        nx, ny, nz = self.shape
        lin = lambda i: (i[:, 0] * ny + i[:, 1]) * nz + i[:, 2]
        flat = self.L.reshape(-1)
        fu = np.unique(lin(free_idx))
        hu = np.unique(lin(hit_idx)) if len(hit_idx) else np.zeros(0, np.int64)
        flat[fu] = np.clip(flat[fu] + c["l_free"], c["l_min"], c["l_max"])
        flat[hu] = np.clip(flat[hu] + c["l_occ"], c["l_min"], c["l_max"])
        if len(hit_idx):
            self.seen[hit_idx[:, 0], hit_idx[:, 1]] = True
        if len(free_idx):
            self.seen[free_idx[:, 0], free_idx[:, 1]] = True
        self.n_frames += 1
        self.version += 1
        if t is not None:
            self.t_now = float(t)
        if self.dm.get("enabled") and self.z_band is not None:
            self._update_dynamic()

    def _update_dynamic(self):
        c, dm = self.c, self.dm
        k0 = max(int(np.floor((self.z_band - c["band_half_m"] - self.origin[2]) / self.v)), 0)
        k1 = min(int(np.ceil((self.z_band + c["band_half_m"] - self.origin[2]) / self.v)), self.shape[2])
        band_max = self.L[:, :, k0:k1].max(axis=2)
        to_occ = band_max > dm.get("occ_l", 1.6)
        to_free = self.seen & (band_max < c["free_threshold"])
        cleared = self.dyn_state & to_free
        if int(cleared.sum()) >= dm.get("min_cells", 3):                    # ignore isolated flicker
            self.dyn_until[cleared] = self.t_now + dm.get("hold_s", 90.0)
            self.n_dynamic_events += 1
            self.version += 1
        self.dyn_state = (self.dyn_state | to_occ) & ~to_free

    # ------------------------------------------------------------------ layers
    def planning_grid(self, z_center: float):
        """int8 (nx, ny): FREE / OCCUPIED / UNKNOWN for the vehicle's altitude band."""
        c = self.c
        k0 = int(np.floor((z_center - c["band_half_m"] - self.origin[2]) / self.v))
        k1 = int(np.ceil((z_center + c["band_half_m"] - self.origin[2]) / self.v))
        k0, k1 = max(k0, 0), min(k1, self.shape[2])
        band = self.L[:, :, k0:k1]
        occ = (band > c["occ_threshold"]).any(axis=2)
        nfree = (band < c["free_threshold"]).sum(axis=2)
        free = (~occ) & (nfree >= max(1, (k1 - k0) // 2))
        g = np.full(self.shape[:2], UNKNOWN, np.int8)
        g[free] = FREE
        g[occ] = OCCUPIED
        self.z_band = z_center
        if self.dm.get("enabled"):
            dyn = (self.dyn_until > self.t_now).astype(np.uint8)
            if dyn.any():
                import cv2
                r = max(int(round(self.dm.get("radius_m", 0.6) / self.v)), 1)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                g[(cv2.dilate(dyn, k) > 0) & (g != OCCUPIED)] = DYNAMIC
        return g

    def surface_classes(self):
        """Highest occupied voxel per column -> semantic surface class."""
        c = self.c
        occ = self.L > c["occ_threshold"]
        any_occ = occ.any(axis=2)
        top = self.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=2)
        h = self.origin[2] + (top + 0.5) * self.v                          # height above the take-off body origin
        ground_z = -0.17                                                    # body starts ~0.17 m above the take-off surface
        hh = h - ground_z
        lo, hi = c["surface_height_classes_m"]
        cls = np.full(self.shape[:2], CLS_UNKNOWN, np.uint8)
        cls[any_occ & (hh < lo)] = CLS_GROUND
        cls[any_occ & (hh >= lo) & (hh < hi)] = CLS_DEBRIS
        cls[any_occ & (hh >= hi)] = CLS_STRUCTURE
        return cls

    def max_height_map(self):
        occ = self.L > self.c["occ_threshold"]
        any_occ = occ.any(axis=2)
        top = self.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=2)
        h = np.where(any_occ, self.origin[2] + (top + 0.5) * self.v + 0.17, np.nan)
        return h
