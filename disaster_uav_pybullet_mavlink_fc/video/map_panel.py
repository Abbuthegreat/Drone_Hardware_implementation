"""
Render the live geo-semantic map as a top-down panel image (for the demo video / operator view).

Layers (drawn from the SAME map object the planner uses - nothing from the simulator):
  unknown space (dark), observed free air in the flight band, occupied cells in the flight band, surface classes from the highest occupied
  voxel (ground / low debris / structure), hazard regions, survivor nodes (yellow = candidate, red = confirmed, with a label and geo-tag),
  A* path, SLAM trail, UAV pose, search waypoints.
Coordinates: local map frame, x East (right), y North (up).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from mapping.metric_map import CLS_GROUND, CLS_DEBRIS, CLS_STRUCTURE
from video.textutil import put_text

C_UNKNOWN = (46, 44, 42)
C_FREE = (92, 90, 86)
C_GROUND = (112, 128, 108)
C_DEBRIS = (52, 128, 200)
C_STRUCT = (60, 60, 196)
C_OCC_BAND = (30, 30, 235)


class MapPanel:
    def __init__(self, size=(520, 520), min_extent_m=(26.0, 26.0)):
        self.W, self.H = size
        self.min_ext = np.array(min_extent_m, float)
        self.view = None                                   # (x0, y0, x1, y1) eased

    def _view(self, gmap, pose_xy, extra_pts):
        m = gmap.metric
        seen = np.argwhere(m.seen)
        pts = [np.array(pose_xy)] + [np.array(p) for p in extra_pts]
        if len(seen):
            lo = m.origin[:2] + seen.min(axis=0) * m.v
            hi = m.origin[:2] + (seen.max(axis=0) + 1) * m.v
            pts += [lo, hi]
        P = np.array(pts)
        lo, hi = P.min(axis=0) - 2.5, P.max(axis=0) + 2.5
        ext = np.maximum(hi - lo, self.min_ext)
        c = (lo + hi) / 2
        aspect = self.W / self.H
        if ext[0] / ext[1] > aspect:
            ext[1] = ext[0] / aspect
        else:
            ext[0] = ext[1] * aspect
        v = np.array([c[0] - ext[0] / 2, c[1] - ext[1] / 2, c[0] + ext[0] / 2, c[1] + ext[1] / 2])
        self.view = v if self.view is None else 0.9 * self.view + 0.1 * v          # ease so the panel does not jump
        return self.view

    def render(self, gmap, z_plan, pose_xy, yaw, trail, path, waypoints, wp_index, t, status_text=""):
        m = gmap.metric
        img = np.zeros((self.H, self.W, 3), np.uint8)
        img[:] = C_UNKNOWN
        x0, y0, x1, y1 = self._view(gmap, pose_xy, list(waypoints) + [(s.local_x, s.local_y) for s in gmap.survivors.tracks.values()])
        sx, sy = self.W / (x1 - x0), self.H / (y1 - y0)

        def px(x, y):
            return int(round((x - x0) * sx)), int(round(self.H - (y - y0) * sy))

        # rasterise the map cells in view
        xs = x0 + (np.arange(self.W) + 0.5) / sx
        ys = y0 + (self.H - 1 - np.arange(self.H) + 0.5) / sy
        ix = np.floor((xs - m.origin[0]) / m.v).astype(int)
        iy = np.floor((ys - m.origin[1]) / m.v).astype(int)
        vx = (ix >= 0) & (ix < m.shape[0])
        vy = (iy >= 0) & (iy < m.shape[1])
        grid = gmap.planning_grid(z_plan)
        cls = m.surface_classes()
        II, JJ = np.meshgrid(np.clip(ix, 0, m.shape[0] - 1), np.clip(iy, 0, m.shape[1] - 1), indexing="ij")     # (W,H)
        inb = np.outer(vx, vy)
        g = grid[II, JJ].T
        c = cls[II, JJ].T
        ok = inb.T
        img[ok & (g == 0)] = C_FREE
        img[ok & (c == CLS_GROUND)] = C_GROUND
        img[ok & (c == CLS_DEBRIS)] = C_DEBRIS
        img[ok & (c == CLS_STRUCTURE)] = C_STRUCT
        img[ok & (g == 100)] = C_OCC_BAND
        # 5 m grid
        for gx in np.arange(math.ceil(x0 / 5) * 5, x1, 5):
            cv2.line(img, px(gx, y0), px(gx, y1), (70, 70, 70), 1)
        for gy in np.arange(math.ceil(y0 / 5) * 5, y1, 5):
            cv2.line(img, px(x0, gy), px(x1, gy), (70, 70, 70), 1)
        # hazards
        for h in gmap.hazards.items.values():
            col = (0, 140, 255) if h.status == "CONFIRMED" else (0, 200, 255)
            cv2.circle(img, px(h.x, h.y), max(6, int(h.radius_m * sx)), col, 2, cv2.LINE_AA)
            cv2.circle(img, px(h.x, h.y), max(3, int(h.radius_m * sx * 0.45)), col, -1, cv2.LINE_AA)
            self._label(img, f"HAZARD {h.kind.upper()} {h.max_temp_c:.0f}C", px(h.x, h.y), (0, 165, 255), dy=-max(10, int(h.radius_m * sx)) - 6)
        # trail, path, waypoints
        if len(trail) > 1:
            cv2.polylines(img, [np.array([px(*p) for p in trail], np.int32).reshape(-1, 1, 2)], False, (0, 220, 255), 1, cv2.LINE_AA)
        if path and len(path) > 1:
            cv2.polylines(img, [np.array([px(*p) for p in path], np.int32).reshape(-1, 1, 2)], False, (255, 220, 40), 2, cv2.LINE_AA)
        for k, w in enumerate(waypoints):
            col = (255, 255, 255) if k >= wp_index else (140, 140, 140)
            cv2.circle(img, px(*w), 5, col, 1, cv2.LINE_AA)
            cv2.putText(img, f"WP{k+1}", (px(*w)[0] + 7, px(*w)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        # survivors
        for s in sorted(gmap.survivors.tracks.values(), key=lambda s: s.track_id):
            conf = s.status == "CONFIRMED"
            col = (40, 40, 235) if conf else (0, 230, 255)
            p = px(s.local_x, s.local_y)
            cv2.circle(img, p, 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, p, 9, col, -1, cv2.LINE_AA)
            cv2.putText(img, str(s.track_id), (p[0] - 4, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255) if conf else (0, 0, 0), 1, cv2.LINE_AA)
            if conf:
                lines = [s.label(), f"LAT {s.latitude:.6f}", f"LON {s.longitude:.6f}", f"ALT {s.altitude:.1f} m", f"CONF {s.confidence:.2f}  +/-{s.position_uncertainty:.2f} m"]
                bx = min(max(p[0] + 14, 4), self.W - 170)
                by = min(max(p[1] - 30, 4), self.H - 80)
                cv2.rectangle(img, (bx - 3, by - 12), (bx + 160, by + 6 * 13 - 6), (20, 20, 20), -1)
                cv2.rectangle(img, (bx - 3, by - 12), (bx + 160, by + 6 * 13 - 6), col, 1)
                for i, ln in enumerate(lines):
                    cv2.putText(img, ln, (bx, by + i * 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255) if i else (120, 120, 255), 1, cv2.LINE_AA)
            else:
                self._label(img, "candidate", p, (0, 230, 255), dy=22)
        # UAV
        if pose_xy is not None:
            p = px(*pose_xy)
            L = 11
            tip = (int(p[0] + L * 1.4 * math.cos(yaw)), int(p[1] - L * 1.4 * math.sin(yaw)))
            l = (int(p[0] + L * math.cos(yaw + 2.5)), int(p[1] - L * math.sin(yaw + 2.5)))
            r = (int(p[0] + L * math.cos(yaw - 2.5)), int(p[1] - L * math.sin(yaw - 2.5)))
            cv2.fillPoly(img, [np.array([tip, l, r], np.int32)], (0, 255, 255))
            cv2.polylines(img, [np.array([tip, l, r], np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)
        # scale bar, compass, legend
        bar = 5.0
        cv2.rectangle(img, (10, self.H - 22), (10 + int(bar * sx), self.H - 16), (255, 255, 255), -1)
        cv2.putText(img, "5 m", (14 + int(bar * sx), self.H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.arrowedLine(img, (self.W - 24, 42), (self.W - 24, 14), (255, 255, 255), 2, tipLength=0.35)
        cv2.putText(img, "N", (self.W - 30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "LIVE GEO-SEMANTIC MAP", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        leg = [("unknown", C_UNKNOWN), ("free", C_FREE), ("ground", C_GROUND), ("debris", C_DEBRIS), ("structure", C_STRUCT), ("A* path", (255, 220, 40))]
        for i, (n, col) in enumerate(leg):
            y = 32 + i * 13
            cv2.rectangle(img, (8, y - 8), (18, y + 1), col, -1)
            cv2.putText(img, n, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1, cv2.LINE_AA)
        n_conf = len(gmap.survivors.confirmed())
        cv2.putText(img, f"survivors confirmed: {n_conf}   candidates: {len(gmap.survivors.tracks) - n_conf}", (8, self.H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _label(img, text, p, col, dy=0):
        put_text(img, text, (p[0] + 8, p[1] + dy), 0.38, col, 1)
