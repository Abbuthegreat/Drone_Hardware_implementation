"""
Perception -> mapping pipeline (ROS-independent, so it is unit-testable and reusable by the ROS node).

    frame(t): RGB + depth + thermal   and   SLAM pose T_map_body(t)
      -> integrate depth into the metric voxel map (ray casting)
      -> person detection (kit YOLO) -> depth 3-D point (kit localize) -> thermal support/contradiction -> fused confidence
      -> camera point -> map point through T_map_cam = T_map_body * T_body_cam
      -> survivor tracker (association, filtering, confirmation) -> lat/lon from the GPS anchor
      -> hot-hazard detection (thermal) -> hazard nodes
Everything used here is available on the real vehicle: pixels, SLAM pose, static calibration, GPS anchor.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from mapping.geo_semantic_map import GeoSemanticMap
from perception.front_clearance import front_clearance
from perception.person_detector import PersonDetector
from perception.thermal_fusion import ThermalFusion
from video.textutil import put_text


def _R(rows):
    return np.array(rows, float).reshape(3, 3)


class PerceptionMappingPipeline:
    def __init__(self, cfg, K_rgb, K_th, evidence_dir=None, detector=None):
        self.cfg = cfg
        self.K = K_rgb
        self.map = GeoSemanticMap(cfg)
        self.det = detector or PersonDetector(cfg["perception"], K_rgb)
        self.fusion = ThermalFusion(cfg, K_rgb, K_th)
        ext = cfg["sensor_extrinsics"]["rgbd_optical"]
        self.T_body_cam = np.eye(4)
        self.T_body_cam[:3, :3] = _R(ext["R_body_from_optical"])
        self.T_body_cam[:3, 3] = ext["xyz_body_flu_m"]
        self.evidence_dir = evidence_dir
        if evidence_dir:
            os.makedirs(evidence_dir, exist_ok=True)
        self.frame_idx = 0
        self.log = []

    def T_map_cam(self, T_map_body):
        return T_map_body @ self.T_body_cam

    def process(self, t, bgr, depth_m, thermal_c, T_map_body):
        t0 = time.perf_counter()
        c = self.cfg["perception"]
        Tmc = self.T_map_cam(T_map_body)
        self.map.metric.integrate(depth_m, self.K, Tmc, t)
        persons, obs, boxes = [], [], []
        for d in self.det.detect(bgr):
            P = self.det.localize(d["box"], depth_m)
            rec = {"box": d["box"], "conf_rgb": d["conf"], "xyz_cam": None, "range": None, "thermal": None, "conf": d["conf"]}
            spread = self.det.depth_spread(d["box"], depth_m)
            rec["depth_spread"] = spread
            reliable = spread <= c["max_depth_spread_m"]
            if P is not None and not reliable:
                rec["rejected"] = "depth patch straddles several surfaces"
                P = None
            if P is not None:
                chk = self.fusion.person_thermal_check(P, thermal_c)
                rec.update(xyz_cam=P.tolist(), range=float(np.linalg.norm(P)), thermal=chk,
                           conf=self.fusion.fuse_confidence(d["conf"], chk))
                Pm = (Tmc @ np.append(P, 1.0))[:3]
                rec["xyz_map"] = Pm.tolist()
                obs.append({"xyz_map": Pm, "conf": rec["conf"], "range": rec["range"], "thermal_support": chk.get("support") if chk.get("in_view") else None,
                            "evidence": "", "_rec": rec, "_box": d["box"]})
            persons.append(rec)
        # evidence image = frame with the box (saved when it improves a track)
        tids = []
        if obs:
            if self.evidence_dir:
                for o in obs:
                    x1, y1, x2, y2 = map(int, o["_box"])
                    img = bgr.copy()
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    o["evidence"] = os.path.join(self.evidence_dir, f"f{self.frame_idx:05d}_{x1}_{y1}.jpg")
                    cv2.imwrite(o["evidence"], img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            tids = self.map.survivors.update(t, obs)
            for o, tid in zip(obs, tids):
                o["_rec"]["track_id"] = tid
        # hazards
        haz_obs = []
        for h in self.fusion.hot_hazards(thermal_c, depth_m, bgr):
            Pm = (Tmc @ np.append(h["xyz_cam"], 1.0))[:3]
            haz_obs.append({"xyz_map": Pm, "kind": h["kind"], "radius_m": h["radius_m"], "temp_c": h["temp_c"],
                            "thermal_support": h["thermal_support"], "rgb_support": h["rgb_support"]})
        self.map.hazards.update(t, haz_obs)
        self.frame_idx += 1
        g = self.cfg["nav"].get("front_guard", {})
        fc = front_clearance(depth_m, self.K, g.get("half_width_m", 0.6), g.get("half_height_m", 0.6), percentile=g.get("percentile", 5.0))
        out = {"t": t, "persons": persons, "hazards": haz_obs, "front_clearance_m": fc, "ms": (time.perf_counter() - t0) * 1e3}
        return out

    def annotate(self, bgr, result):
        """Detection overlay for the video (yellow = candidate, red = confirmed, grey = unlocalised)."""
        img = bgr.copy()
        for r in result["persons"]:
            x1, y1, x2, y2 = map(int, r["box"])
            tid = r.get("track_id")
            tr = self.map.survivors.tracks.get(tid) if tid else None
            col = (0, 0, 255) if tr and tr.status == "CONFIRMED" else (0, 220, 255) if tr else (170, 170, 170)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            th = r.get("thermal") or {}
            lab = f"person {r['conf']:.2f}" + (f" {r['range']:.1f}m" if r.get("range") else "") + (" +thermal" if th.get("support") else "")
            if tr:
                lab = f"#{tr.track_id:02d} " + lab
            put_text(img, lab, (x1, max(14, y1 - 5)), 0.45, col, 1)
        return img
