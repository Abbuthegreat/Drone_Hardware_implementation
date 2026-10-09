"""
Data contracts between the vehicle/sensor layer and the (future) navigation stack.

NOTHING in this module, or anywhere under algo/, may import PyBullet, `sim`, `worlds` or `eval`
(tests/test_ground_truth_firewall.py enforces this). Ground truth simply has no field here.

Frames: map/NED as MAVLink defines it (x North, y East, z Down), body FRD for IMU, camera OPTICAL frame
(z forward, x right, y down) for image / depth geometry. Time is the sensor/FC timestamp in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CameraInfo:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_m: float
    far_m: float
    frame_id: str = "rgbd_camera_optical_link"


@dataclass
class RgbdFrame:
    t: float
    rgb: np.ndarray            # uint8 (H, W, 3)
    depth_m: np.ndarray        # float32 (H, W), metres along the optical axis, NaN = invalid
    info: CameraInfo


@dataclass
class ImuMeasurement:
    t: float
    gyro_frd: np.ndarray       # rad/s
    accel_frd: np.ndarray      # m/s^2 (specific force)


@dataclass
class GpsMeasurement:
    t: float
    fix_type: int              # 0 = no fix, 3 = 3D
    lat_deg: float
    lon_deg: float
    alt_m: float
    horizontal_accuracy_m: float
    satellites: int


@dataclass
class VehicleState:
    """What the flight controller reports over MAVLink (its own estimate, NOT ground truth)."""
    t: float
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    yaw_rad: float
    mode: str
    armed: bool
    battery_v: float = float("nan")


@dataclass
class SensorBundle:
    rgbd: Optional[RgbdFrame]
    imu: List[ImuMeasurement] = field(default_factory=list)   # all IMU samples since the previous call
    gps: Optional[GpsMeasurement] = None
    state: Optional[VehicleState] = None


@dataclass
class SurvivorNode:
    """Persistent geo-semantic survivor node (to be produced by the future detector + mapper)."""
    track_id: int
    cls: str
    confidence: float
    local_x: float
    local_y: float
    local_z: float
    latitude: float
    longitude: float
    altitude: float
    position_uncertainty_m: float
    first_seen: float
    last_seen: float
    observation_count: int
    status: str
    evidence_image: Optional[np.ndarray] = None
