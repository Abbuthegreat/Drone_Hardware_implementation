"""
SLAM adapter: sensor conventions in, UAV pose out.

RGB-D + IMU  ->  [backend: ORB+PnP VO here / ORB-SLAM3 on Linux]  ->  UAV BODY pose in a gravity- and heading-aligned map frame

Conventions (explicit, nothing hidden)
--------------------------------------
* Backend world frame  : the first camera OPTICAL frame (z fwd, x right, y down) - what ORB-SLAM3 / the kit also use.
* IMU in                : FRD body frame, rad/s and m/s^2 (MAVLink HIGHRES_IMU). Converted to FLU (REP-103) and then to the camera optical
                          frame with the static extrinsics from config `sensor_extrinsics` (rotation only; lever arm ignored).
* Gravity alignment     : same idea as the kit (record.py measures 'up' from the accelerometer while still, postprocess.py rotates the map
                          with rot_a_to_b). Here the accelerometer mean over `init_still_s` gives 'up' in the camera frame.
* Heading alignment     : the FC heading (MAVLink ATTITUDE yaw, clockwise from North) at start rotates the map so x = East, y = North.
* Output map frame      : `slam_map` = ENU-like local frame, origin = UAV body (CG) at the first frame, z up.
* Output pose           : T_map_body (body FLU -> slam_map). The camera is a rigid child of the body (URDF extrinsics), so
                          T_map_body = T_map_w * T_w_c * T_c_body.
Nothing here reads simulator state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from slam import kit
from slam.backend import SlamBackend, SlamResult


def _R_from(rows):
    return np.array(rows, float).reshape(3, 3)


@dataclass
class UavPose:
    t: float
    state: str
    T_map_body: Optional[np.ndarray]
    T_map_cam: Optional[np.ndarray]
    info: dict


class SlamFrontEnd:
    def __init__(self, cfg: dict, backend: SlamBackend):
        self.cfg, self.c = cfg, cfg["slam"]
        self.backend = backend
        ext = cfg["sensor_extrinsics"]["rgbd_optical"]
        self.R_bo = _R_from(ext["R_body_from_optical"])        # optical -> body FLU
        self.t_bo = np.array(ext["xyz_body_flu_m"], float)     # optical origin in body (FLU, from CG)
        # camera -> body transform T_c_body (body point expressed in the camera frame)
        self.T_c_body = np.eye(4)
        self.T_c_body[:3, :3] = self.R_bo.T
        self.T_c_body[:3, 3] = -self.R_bo.T @ self.t_bo
        self.pp = kit.postprocess_module()
        self.accel_sum, self.accel_n = np.zeros(3), 0
        self.M = None                                          # map <- backend-world
        self.heading_enu = None
        self.first_frame_pending = True
        self.up_cam = None
        self.maps = {}                                         # backend map id -> alignment M (one frame per ORB-SLAM3 Atlas map)
        self.last_good = None                                  # (t, T_map_cam, T_w_c, map_id) of the last accepted pose
        self.n_map_resets = 0

    # ------------------------------------------------------------------ IMU
    def _to_cam(self, v_frd):
        v_flu = np.array([v_frd[0], -v_frd[1], -v_frd[2]], float)
        return self.R_bo.T @ v_flu

    def feed_imu(self, t, gyro_frd, accel_frd):
        gyro_c, acc_c = self._to_cam(gyro_frd), self._to_cam(accel_frd)
        self.backend.feed_imu(t, gyro_c, acc_c)
        if self.M is None:
            self.accel_sum += acc_c
            self.accel_n += 1

    def set_initial_heading(self, yaw_ned_rad: float):
        """FC heading (clockwise from North) at start -> ENU heading of the body x axis."""
        self.heading_enu = math.pi / 2 - yaw_ned_rad

    def ready_to_start(self, min_samples: int = 20) -> bool:
        return self.accel_n >= min_samples and self.heading_enu is not None

    # ------------------------------------------------------------------ frames
    def _make_alignment(self, T_w_c0):
        up_c = self.accel_sum / max(np.linalg.norm(self.accel_sum), 1e-9)      # 'up' in the camera frame of frame 0
        up_w = T_w_c0[:3, :3] @ up_c
        R_al = self.pp.rot_a_to_b(up_w, np.array([0, 0, 1.0]))                # kit function: rotate 'up' onto +z
        f = R_al @ (T_w_c0[:3, :3] @ np.array([0, 0, 1.0]))                   # camera forward in the aligned frame
        h_cam = math.atan2(f[1], f[0])
        d = self.heading_enu - h_cam
        Rz = np.array([[math.cos(d), -math.sin(d), 0], [math.sin(d), math.cos(d), 0], [0, 0, 1.0]])
        R = Rz @ R_al
        T_w_body0 = T_w_c0 @ self.T_c_body
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = -R @ T_w_body0[:3, 3]
        self.up_cam = up_c
        return M

    def process(self, t, bgr, depth_m) -> UavPose:
        if self.M is None and not self.ready_to_start():        # gravity + heading must be known BEFORE the first backend frame,
            return UavPose(t, "INIT", None, None, {"waiting": "imu/heading"})   # because frame 0 defines the backend world frame
        r: SlamResult = self.backend.process(t, bgr, depth_m)
        if r.T_w_c is None:
            return UavPose(t, r.state, None, None, r.info)
        map_id = r.info.get("map_id", 0)
        if map_id < 0:                                          # no tracked point to name the map: assume the current one
            map_id = self.last_good[3] if self.last_good else 0
        if self.M is None:
            self.M = self._make_alignment(r.T_w_c)
            self.maps[map_id] = self.M
            r.info["aligned"] = True
        else:
            new_map = map_id not in self.maps
            if not new_map and self.last_good is not None and map_id == self.last_good[3]:
                # same map id but a physically impossible jump (ORB-SLAM3 reset the active map, or a corrupted pose): treat as a new frame as well
                tl, _, Twc_l, _ = self.last_good
                dt = min(max(t - tl, 0.0), 1.0)
                lim = self.c.get("max_speed_mps", 2.0) * dt + self.c.get("jump_margin_m", 0.15)
                new_map = float(np.linalg.norm(r.T_w_c[:3, 3] - Twc_l[:3, 3])) > lim and (t - tl) > 0
            if new_map and self.last_good is not None:
                # ORB-SLAM3 started a new map (tracking was lost; the mission hovers meanwhile, so the last accepted pose is still where the vehicle is):
                # anchor the new map's frame there. If the maps merge later, the old map id comes back and its own alignment is used again.
                self.M = self.last_good[1] @ np.linalg.inv(r.T_w_c)
                self.maps[map_id] = self.M
                self.n_map_resets += 1
                r.info.update(new_map=True, map_id=map_id, n_map_resets=self.n_map_resets)
            elif new_map:
                self.M = self._make_alignment(r.T_w_c)
                self.maps[map_id] = self.M
            else:
                self.M = self.maps.get(map_id, self.M)
        T_map_c = self.M @ r.T_w_c
        T_map_body = T_map_c @ self.T_c_body
        self.last_good = (t, T_map_c, r.T_w_c, map_id)
        r.info["n_map_resets"] = self.n_map_resets
        return UavPose(t, "OK" if r.state != "INIT" else "OK", T_map_body, T_map_c, r.info)

    def reset(self):
        self.backend.reset()
        self.M = None
        self.maps, self.last_good = {}, None
        self.accel_sum[:] = 0
        self.accel_n = 0
