"""
Navigation algorithm entry point (INTERFACE ONLY at this stage - no navigation is implemented).

The algorithm knows NOTHING about PyBullet and has no access to ground truth. Its whole world is:

    inputs   : SensorBundle (RGB-D frame, IMU samples, GPS fix, FC state) from algo/interfaces.py
    outputs  : ONE setpoint per step through the vehicle interface (CompanionMavlink / the ROS 2 bridge)

        vehicle.send_velocity_ned(vn, ve, vd, yaw_rate)
        vehicle.send_position_ned(north, east, down, yaw)

Simulation: the vehicle is CompanionMavlink("udp:127.0.0.1:14540") (or the ROS 2 topic /uav/cmd_vel).
Hardware  : the same class with "serial:/dev/serial0:921600". Nothing here changes.

The planned pipeline is documented as comments only:
  RGB-D -> ORB-SLAM3 RGB-D / VIWO (+IMU, GPS) -> estimated pose -> live metric map
        -> person detector -> depth 3-D point -> camera->body->map->lat/lon -> persistent SurvivorNode
        -> A* (global) + VFH+ (local) -> safety check -> setpoint.
"""
from __future__ import annotations

from typing import Optional

from algo.interfaces import SensorBundle


class DisasterNavigationAlgorithm:
    def __init__(self, vehicle):
        self.vehicle = vehicle          # exposes send_velocity_ned / send_position_ned only
        self.survivors = {}             # track_id -> SurvivorNode (future)
        self.metric_map = None          # future live metric map

    def step(self, sensors: SensorBundle) -> Optional[tuple]:
        """Called at the sensor rate. Returns the setpoint it sent (or None). NOT IMPLEMENTED YET."""
        # 1) state estimation: RGB-D SLAM / VIWO fed with sensors.rgbd, sensors.imu, sensors.gps
        # 2) update the metric map from depth + estimated pose
        # 3) person detection on sensors.rgbd.rgb
        # 4) 3-D localisation: depth + intrinsics -> camera point -> body (URDF extrinsics) -> map -> lat/lon (common.geo)
        # 5) create / update persistent survivor nodes
        # 6) A* global path, VFH+ local avoidance, safety checks
        # 7) self.vehicle.send_velocity_ned(...)   # keep streaming >= 2 Hz or the FC watchdog will trigger failsafe
        return None
