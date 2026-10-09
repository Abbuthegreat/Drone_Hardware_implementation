"""
Forward clearance straight from the depth image (camera frame, no SLAM pose, no map).

Last-line collision guard for the navigation layer: a moving or newly collapsed object can corrupt the SLAM pose by a metre, so the SLAM-frame map
may place an obstacle too far away. The depth image is a direct measurement in the body frame, immune to that error.

front_clearance(): 5th-percentile depth [m, along the optical axis] of the pixels that lie inside the vehicle's swept corridor in front of the camera
(|X| < half_width and |Y| < half_height in the camera frame, near < z < far). None when too few pixels are valid.
"""
from __future__ import annotations

import numpy as np


def front_clearance(depth_m, K, half_width_m=0.6, half_height_m=0.6, near_m=0.25, far_m=6.0, percentile=5.0, min_pixels=60):
    h, w = depth_m.shape
    z = np.nan_to_num(depth_m.astype(np.float32), nan=0.0)
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    X = (u - K["cx"]) * z / K["fx"]
    Y = (v - K["cy"]) * z / K["fy"]
    m = (z > near_m) & (z < far_m) & (np.abs(X) < half_width_m) & (np.abs(Y) < half_height_m)
    if int(m.sum()) < min_pixels:
        return None
    return float(np.percentile(z[m], percentile))
