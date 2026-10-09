"""
A* global planner on the LIVE metric map (2-D planning layer of the geo-semantic map).

Grid semantics: FREE = 0, OCCUPIED = 100, UNKNOWN = -1 (mapping.metric_map). Only FREE cells that keep `clearance_m` from
OCCUPIED cells are traversable; UNKNOWN is never treated as free (a small standoff from unknown is kept too). If the goal is not
reachable through known-free space (typical while exploring: the goal lies in unknown territory), the planner returns the path to the
reachable cell closest to the goal (`partial=True`) - the UAV then progresses along it and re-plans as the map grows.
No simulator information is used: the grid comes from RGB-D + SLAM pose.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

FREE, OCC, UNK = 0, 100, -1
DYN = 90                                                  # remembered moving-obstacle region (mapping.metric_map.DYNAMIC): blocked for A* (astar.dynamic_mode: block) or extra cost (penalty); ignored by VFH+
_N8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


@dataclass
class PlanResult:
    path: List[Tuple[float, float]] = field(default_factory=list)
    found: bool = False            # a path exists (complete or partial)
    partial: bool = False          # goal not reachable through known-free space; path leads to the closest reachable cell
    cost: float = 0.0
    expanded: int = 0
    reason: str = ""


class AStarPlanner:
    def __init__(self, cfg_astar: dict, voxel_m: float, origin_xy):
        self.c = cfg_astar
        self.v = voxel_m
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        self.inflated: Optional[np.ndarray] = None
        self.dynamic_mode_override: Optional[str] = None      # set by the mission: temporarily cross remembered moving-obstacle regions at a cost

    # ------------------------------------------------------------------ helpers
    def to_cell(self, x, y):
        return int(math.floor((x - self.ox) / self.v)), int(math.floor((y - self.oy) / self.v))

    def to_xy(self, i, j):
        return self.ox + (i + 0.5) * self.v, self.oy + (j + 0.5) * self.v

    def traversable(self, grid, clearance_m=None):
        """bool mask of cells the UAV centre may occupy."""
        clr = self.c["clearance_m"] if clearance_m is None else clearance_m
        block = (self.dynamic_mode_override or self.c.get("dynamic_mode", "block")) == "block"
        occ = ((grid == OCC) | ((grid == DYN) & block)).astype(np.uint8)
        unk = (grid == UNK).astype(np.uint8)
        # distance (in metres) from every cell to the nearest occupied / unknown cell
        d_occ = cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3) * self.v if occ.any() else np.full(grid.shape, 1e6, np.float32)
        d_unk = cv2.distanceTransform(1 - unk, cv2.DIST_L2, 3) * self.v if unk.any() else np.full(grid.shape, 1e6, np.float32)
        free = ((grid == FREE) | ((grid == DYN) & (not block))) & (d_occ >= clr) & (d_unk >= 0.25)      # penalty mode: remembered-moving regions stay traversable at a cost
        return free

    # ------------------------------------------------------------------ planning
    def plan(self, grid, start_xy, goal_xy) -> PlanResult:
        free = self.traversable(grid)
        self.inflated = free
        dyn_pen = np.where(grid == DYN, float(self.c.get("dynamic_penalty", 6.0)), 0.0).astype(np.float32)   # prefer a route that avoids remembered moving obstacles
        nx, ny = free.shape
        si, sj = self.to_cell(*start_xy)
        gi, gj = self.to_cell(*goal_xy)
        if not (0 <= si < nx and 0 <= sj < ny):
            return PlanResult(reason="start outside the map")
        # let the vehicle leave a tight spot: cells within 0.6 m of the start are passable if they are not occupied
        r = int(math.ceil(0.6 / self.v))
        i0, i1, j0, j1 = max(si - r, 0), min(si + r + 1, nx), max(sj - r, 0), min(sj + r + 1, ny)
        free = free.copy()
        free[i0:i1, j0:j1] |= (grid[i0:i1, j0:j1] != OCC)
        gi, gj = min(max(gi, 0), nx - 1), min(max(gj, 0), ny - 1)

        def h(i, j):
            dx, dy = abs(i - gi), abs(j - gj)
            return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

        g = {(si, sj): 0.0}
        parent = {}
        heap = [(h(si, sj), 0.0, si, sj, -1)]
        best, best_h = (si, sj), h(si, sj)
        closed = np.zeros(free.shape, bool)
        expanded = 0
        goal_tol = max(1, int(round(self.c.get("goal_reach_cells", 1))))
        reached = None
        while heap and expanded < self.c["max_nodes"]:
            f, gc, i, j, di = heapq.heappop(heap)
            if closed[i, j]:
                continue
            closed[i, j] = True
            expanded += 1
            hh = h(i, j)
            if hh < best_h:
                best, best_h = (i, j), hh
            if abs(i - gi) <= goal_tol - 1 and abs(j - gj) <= goal_tol - 1:
                reached = (i, j)
                break
            for k, (dx, dy, w) in enumerate(_N8):
                ni, nj = i + dx, j + dy
                if not (0 <= ni < nx and 0 <= nj < ny) or closed[ni, nj] or not free[ni, nj]:
                    continue
                if dx and dy and not (free[i + dx, j] and free[i, j + dy]):          # no corner cutting
                    continue
                ng = gc + w + dyn_pen[ni, nj] + (self.c["turn_penalty"] if di >= 0 and di != k else 0.0)
                if ng < g.get((ni, nj), 1e18):
                    g[(ni, nj)] = ng
                    parent[(ni, nj)] = (i, j)
                    heapq.heappush(heap, (ng + h(ni, nj), ng, ni, nj, k))
        end = reached or best
        partial = reached is None
        if partial and end == (si, sj):
            return PlanResult(found=False, partial=True, expanded=expanded, reason="no free progress toward the goal")
        cells = [end]
        while cells[-1] in parent:
            cells.append(parent[cells[-1]])
        cells.reverse()
        pts = [self.to_xy(i, j) for i, j in cells]
        if self.c.get("simplify", True) and len(pts) > 2:
            pts = self._simplify(pts, free)
        return PlanResult(pts, True, partial, g.get(end, 0.0) * self.v, expanded, "goal not reachable through known-free space" if partial else "")

    def _los(self, a, b, free):
        (ax, ay), (bx, by) = a, b
        n = int(max(abs(bx - ax), abs(by - ay)) / (self.v * 0.5)) + 1
        for t in np.linspace(0.0, 1.0, n + 1):
            i, j = self.to_cell(ax + (bx - ax) * t, ay + (by - ay) * t)
            if not (0 <= i < free.shape[0] and 0 <= j < free.shape[1]) or not free[i, j]:
                return False
        return True

    def _simplify(self, pts, free):
        out = [pts[0]]
        k = 0
        while k < len(pts) - 1:
            m = len(pts) - 1
            while m > k + 1 and not self._los(pts[k], pts[m], free):
                m -= 1
            out.append(pts[m])
            k = m
        return out
