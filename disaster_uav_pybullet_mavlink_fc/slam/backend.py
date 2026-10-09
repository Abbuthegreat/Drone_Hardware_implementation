"""SLAM backend interface. Any backend consumes RGB-D (+ optional IMU) and returns the camera pose in ITS OWN world frame."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SlamResult:
    t: float
    state: str                                   # INIT | OK | LOST
    T_w_c: Optional[np.ndarray] = None           # 4x4, camera optical frame -> backend world frame (None when not tracking)
    info: dict = field(default_factory=dict)     # inliers, matches, keyframes, timing ...


class SlamBackend:
    name = "abstract"

    def feed_imu(self, t: float, gyro_cam: np.ndarray, accel_cam: np.ndarray) -> None:
        """IMU sample already expressed in the camera OPTICAL frame (rad/s, m/s^2)."""

    def process(self, t: float, bgr: np.ndarray, depth_m: np.ndarray) -> SlamResult:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
