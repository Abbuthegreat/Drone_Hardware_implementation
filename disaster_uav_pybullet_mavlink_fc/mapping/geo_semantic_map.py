"""
Unified geo-semantic disaster map = metric voxel map + persistent survivor nodes + hazard regions + geographic reference.

Survivor nodes are tracks with the kit's association idea (detections closer than `assoc_radius_m` are the same person,
`confirm_min_obs` sightings needed - kit postprocess.py --radius / --min-obs) but applied ONLINE with a running filter, a confidence score,
CANDIDATE -> CONFIRMED status, and a geographic tag computed from the filtered 3-D position:

    detection box -> depth -> camera point -> T_map_cam (SLAM) -> local map point -> GPS anchor -> lat/lon/alt

The survivor's geo-tag is NOT the UAV's GPS position: the UAV may be metres away. The only geographic information used is the GPS anchor
(local origin <-> lat/lon, averaged while the UAV is still on the ground) and the map-frame offset of the survivor from that origin.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from common.geo import GeoFrame
from mapping.metric_map import VoxelMap3D, FREE, OCCUPIED, UNKNOWN


@dataclass
class SurvivorNode:
    track_id: int
    cls: str
    confidence: float
    local_x: float
    local_y: float
    local_z: float
    latitude: float
    longitude: float
    altitude: float
    position_uncertainty: float
    first_seen: float
    last_seen: float
    observation_count: int
    status: str                                   # CANDIDATE | CONFIRMED
    evidence_frame: str = ""                      # file path of the best evidence image
    thermal_support: int = 0                      # observations with supporting thermal evidence
    score: float = 0.0
    confirmed_at: float = -1.0

    def label(self):
        return f"SURVIVOR #{self.track_id:02d}"


@dataclass
class HazardNode:
    hazard_id: int
    kind: str
    x: float
    y: float
    z: float
    radius_m: float
    max_temp_c: float
    latitude: float
    longitude: float
    first_seen: float
    last_seen: float
    observation_count: int
    status: str
    thermal_support: int = 0


class GeoReference:
    """Local map frame (x East, y North, z Up) <-> geodetic, anchored at a GPS fix."""

    def __init__(self):
        self.frame: Optional[GeoFrame] = None
        self.anchor = None

    def set_anchor(self, lat, lon, alt):
        self.frame = GeoFrame(lat, lon, alt)
        self.anchor = (lat, lon, alt)

    @property
    def ready(self):
        return self.frame is not None

    def local_to_geo(self, xyz):
        if self.frame is None:
            return float("nan"), float("nan"), float("nan")
        return self.frame.enu_to_geodetic(np.asarray(xyz, float))

    def geo_to_local(self, lat, lon, alt):
        ned = self.frame.geodetic_to_ned(lat, lon, alt)
        return np.array([ned[1], ned[0], -ned[2]])


class SurvivorTracker:
    def __init__(self, cfg_surv: dict, geo: GeoReference):
        self.c = cfg_surv
        self.geo = geo
        self.tracks: Dict[int, SurvivorNode] = {}
        self._pos: Dict[int, np.ndarray] = {}
        self._w: Dict[int, float] = {}
        self._next = 1
        self.alias: Dict[int, int] = {}            # merged track id -> surviving track id
        self.events: List[dict] = []               # confirmation events (for the logs / UI)

    def _sigma(self, rng):
        return self.c["base_sigma_m"] + self.c["sigma_range_k"] * rng * rng

    def update(self, t, obs_list):
        """obs: dict(xyz_map, conf, range, thermal_support(bool|None), evidence(str)). Returns the track id of each observation (None if rejected)."""
        tids = []
        for o in obs_list:
            if o["conf"] < self.c["min_conf"]:
                tids.append(None)
                continue
            p = np.asarray(o["xyz_map"], float)
            best, bd = None, self.c["assoc_radius_m"]
            for tid, tr in self.tracks.items():
                d = float(np.linalg.norm(p[:2] - self._pos[tid][:2]))
                if d < bd:
                    best, bd = tid, d
            sig = self._sigma(o["range"])
            if best is None:
                tid = self._next
                self._next += 1
                self._pos[tid], self._w[tid] = p.copy(), 1.0 / sig ** 2
                self.tracks[tid] = SurvivorNode(tid, "person", o["conf"], *p, 0, 0, 0, sig, t, t, 1, "CANDIDATE", o.get("evidence", ""),
                                                int(bool(o.get("thermal_support"))), float(o["conf"]))
                best = tid
            else:
                w_new = 1.0 / sig ** 2
                w_old = self._w[best]
                self._pos[best] = (self._pos[best] * w_old + p * w_new) / (w_old + w_new)
                self._w[best] = min(w_old + w_new, 1.0 / 0.10 ** 2)
                tr = self.tracks[best]
                tr.observation_count += 1
                tr.last_seen = t
                tr.score += float(o["conf"])
                tr.thermal_support += int(bool(o.get("thermal_support")))
                if o["conf"] >= tr.confidence:
                    tr.confidence = float(o["conf"])
                    tr.evidence_frame = o.get("evidence", tr.evidence_frame)
                tr.position_uncertainty = float(min(tr.position_uncertainty, math.sqrt(1.0 / self._w[best])))
            tr = self.tracks[best]
            tr.local_x, tr.local_y, tr.local_z = map(float, self._pos[best])
            tr.latitude, tr.longitude, tr.altitude = self.geo.local_to_geo(self._pos[best])
            if tr.status == "CANDIDATE" and tr.observation_count >= self.c["confirm_min_obs"] and tr.score >= self.c["confirm_min_score"]:
                tr.status, tr.confirmed_at = "CONFIRMED", t
                self.events.append({"t": t, "event": "survivor_confirmed", "id": tr.track_id, "lat": tr.latitude, "lon": tr.longitude})
            tids.append(best)
        self._merge_close(t)
        tids = [self.alias.get(x, x) if x is not None else None for x in tids]
        # expire stale weak candidates
        for tid in [k for k, tr in self.tracks.items() if tr.status == "CANDIDATE" and t - tr.last_seen > self.c["candidate_max_age_s"]]:
            del self.tracks[tid], self._pos[tid], self._w[tid]
        return tids

    def _merge_close(self, t):
        """Merge tracks closer than merge_radius_m (duplicates from partially occluded / background-biased depth)."""
        R = self.c.get("merge_radius_m", 0.0)
        if R <= 0:
            return
        changed = True
        while changed:
            changed = False
            ids = sorted(self.tracks)
            for a in ids:
                for b in ids:
                    if b <= a or a not in self.tracks or b not in self.tracks:
                        continue
                    if float(np.linalg.norm(self._pos[a][:2] - self._pos[b][:2])) < R:
                        keep, drop = (a, b) if self.tracks[a].observation_count >= self.tracks[b].observation_count else (b, a)
                        ka, kd = self.tracks[keep], self.tracks[drop]
                        wk, wd = self._w[keep], self._w[drop]
                        self._pos[keep] = (self._pos[keep] * wk + self._pos[drop] * wd) / (wk + wd)
                        self._w[keep] = min(wk + wd, 1.0 / 0.10 ** 2)
                        ka.observation_count += kd.observation_count
                        ka.score += kd.score
                        ka.thermal_support += kd.thermal_support
                        ka.first_seen, ka.last_seen = min(ka.first_seen, kd.first_seen), max(ka.last_seen, kd.last_seen)
                        if kd.confidence > ka.confidence:
                            ka.confidence, ka.evidence_frame = kd.confidence, kd.evidence_frame
                        if kd.status == "CONFIRMED" and ka.status != "CONFIRMED":
                            ka.status, ka.confirmed_at = "CONFIRMED", kd.confirmed_at
                        ka.local_x, ka.local_y, ka.local_z = map(float, self._pos[keep])
                        ka.latitude, ka.longitude, ka.altitude = self.geo.local_to_geo(self._pos[keep])
                        self.alias[drop] = keep
                        del self.tracks[drop], self._pos[drop], self._w[drop]
                        self.events.append({"t": t, "event": "tracks_merged", "kept": keep, "dropped": drop})
                        changed = True
                        break
                if changed:
                    break

    def confirmed(self):
        return [t for t in self.tracks.values() if t.status == "CONFIRMED"]


class HazardTracker:
    def __init__(self, geo: GeoReference, assoc_radius=2.5, confirm_obs=2):
        self.geo, self.r, self.n = geo, assoc_radius, confirm_obs
        self.items: Dict[int, HazardNode] = {}
        self._pos: Dict[int, np.ndarray] = {}
        self._next = 1
        self.events: List[dict] = []

    def update(self, t, obs_list):
        for o in obs_list:
            p = np.asarray(o["xyz_map"], float)
            best, bd = None, self.r
            for hid, h in self.items.items():
                d = float(np.linalg.norm(p[:2] - self._pos[hid][:2]))
                if d < bd:
                    best, bd = hid, d
            if best is None:
                hid = self._next
                self._next += 1
                self._pos[hid] = p.copy()
                lat, lon, _ = self.geo.local_to_geo(p)
                self.items[hid] = HazardNode(hid, o["kind"], *p, o.get("radius_m", 1.0), o.get("temp_c", 0.0), lat, lon, t, t, 1,
                                             "CANDIDATE", int(bool(o.get("thermal_support"))))
            else:
                h = self.items[best]
                self._pos[best] = 0.8 * self._pos[best] + 0.2 * p
                h.x, h.y, h.z = map(float, self._pos[best])
                h.observation_count += 1
                h.last_seen = t
                h.radius_m = max(h.radius_m * 0.9, o.get("radius_m", h.radius_m))
                h.max_temp_c = max(h.max_temp_c, o.get("temp_c", 0.0))
                h.thermal_support += int(bool(o.get("thermal_support")))
                h.latitude, h.longitude, _ = self.geo.local_to_geo(self._pos[best])
                if h.status == "CANDIDATE" and h.observation_count >= self.n and h.thermal_support >= 1:
                    h.status = "CONFIRMED"
                    self.events.append({"t": t, "event": "hazard_confirmed", "id": h.hazard_id, "kind": h.kind})

    def confirmed(self):
        return [h for h in self.items.values() if h.status == "CONFIRMED"]


class GeoSemanticMap:
    """One map: metric occupancy + survivor nodes + hazard regions + geographic metadata."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.metric = VoxelMap3D(cfg["map"])
        self.geo = GeoReference()
        self.survivors = SurvivorTracker(cfg["survivors"], self.geo)
        self.hazards = HazardTracker(self.geo)

    def planning_grid(self, z_center):
        """Planning layer including hazard regions (blocked) - UNKNOWN stays UNKNOWN."""
        g = self.metric.planning_grid(z_center)
        extra = self.cfg["astar"]["hazard_block_extra_m"]
        for h in self.hazards.confirmed() + [h for h in self.hazards.items.values() if h.status == "CANDIDATE" and h.max_temp_c > 100]:
            r = h.radius_m + extra
            ix, iy = self.metric.to_idx([[h.x, h.y, 0.0]])[0, :2]
            n = int(math.ceil(r / self.metric.v))
            xs = np.arange(max(ix - n, 0), min(ix + n + 1, g.shape[0]))
            ys = np.arange(max(iy - n, 0), min(iy + n + 1, g.shape[1]))
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            cx, cy = self.metric.cell_center(X, Y)
            m = (cx - h.x) ** 2 + (cy - h.y) ** 2 <= r * r
            g[X[m], Y[m]] = OCCUPIED
        return g

    def snapshot(self, t):
        return {
            "t": t,
            "geo_anchor": None if not self.geo.ready else {"lat": self.geo.anchor[0], "lon": self.geo.anchor[1], "alt": self.geo.anchor[2]},
            "map": {"frames_integrated": self.metric.n_frames, "voxel_m": self.metric.v},
            "survivors": [asdict(s) for s in self.survivors.tracks.values()],
            "hazards": [asdict(h) for h in self.hazards.items.values()],
        }

    def to_json(self, t):
        return json.dumps(self.snapshot(t))
