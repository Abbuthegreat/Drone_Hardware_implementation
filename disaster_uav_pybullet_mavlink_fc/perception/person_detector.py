"""
RGB person detection + depth-based 3-D localisation, using the attached kit's code unchanged:
    slam_kit/detector.py     Detector (YOLOv8n, COCO class 0 = person; onnxruntime / ultralytics / OpenCV backends)
    slam_kit/postprocess.py  localize(): median depth of the torso region of the box -> 3-D point in the camera frame

Input is RGB pixels only. No simulator IDs or coordinates are involved.
"""
from __future__ import annotations

import os

import numpy as np

from slam import kit


class PersonDetector:
    def __init__(self, cfg_perception: dict, K: dict):
        self.c = cfg_perception
        self.K = (K["fx"], K["fy"], K["cx"], K["cy"])
        self.pp = kit.postprocess_module()
        det_mod = kit.detector_module()
        model = os.path.join(kit.ROOT, self.c["yolo_model"])
        self.det = det_mod.Detector(model, conf=float(self.c["yolo_conf"]), iou=float(self.c["yolo_iou"]),
                                    backend=self.c["yolo_backend"])

    def detect(self, bgr):
        """-> list of dict(box=(x1,y1,x2,y2), conf)"""
        return [{"box": tuple(map(float, d[:4])), "conf": float(d[4])} for d in self.det(bgr)]

    def depth_spread(self, box, depth_m):
        """Spread (p90 - p10, metres) of valid depths in the SAME torso patch the kit's localize() uses. A large spread means the patch straddles
        several surfaces (person + background / ground), so the median depth - and the 3-D point - is unreliable."""
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        u0, u1 = int(x1 + 0.3 * w), int(x2 - 0.3 * w) + 1
        v0, v1 = int(y1 + 0.2 * h), int(y1 + 0.6 * h) + 1
        patch = depth_m[max(v0, 0):max(v1, 1), max(u0, 0):max(u1, 1)]
        vals = patch[np.isfinite(patch) & (patch > 0.2) & (patch < float(self.c["max_depth_m"]))]
        if vals.size < 5:
            return float("inf")
        return float(np.percentile(vals, 90) - np.percentile(vals, 10))

    def localize(self, box, depth_m):
        """3-D point (camera optical frame, metres) of a person box; None if depth is invalid (kit rule: >= min_valid_depth_px pixels)."""
        p = self.pp.localize(box, depth_m.astype(np.float32), self.K, 1.0, float(self.c["max_depth_m"]))
        return None if p is None else np.asarray(p, float)
