"""
Coordinate frames and conversions (pure numpy, shared by simulator and companion).

WORLD / MAP frames
------------------
MAVLink / flight-controller world frame : NED  (x = North, y = East, z = Down)
PyBullet / ROS (REP-103) world frame    : ENU  (x = East,  y = North, z = Up)

    v_enu = C @ v_ned      and      v_ned = C @ v_enu        (C is its own inverse)

BODY frames
-----------
Flight controller / MAVLink body frame : FRD (x = Forward, y = Right, z = Down)
URDF / PyBullet / ROS body frame       : FLU (x = Forward, y = Left,  z = Up)

    v_flu = F @ v_frd      and      v_frd = F @ v_flu         (F is its own inverse)

ATTITUDE
--------
R_ned_frd  rotates FRD body vectors into NED.
R_enu_flu  rotates FLU body vectors into ENU.
    R_ned_frd = C @ R_enu_flu @ F
Euler angles (MAVLink ATTITUDE) are the aerospace ZYX sequence of R_ned_frd:
    yaw   = heading, 0 = North, positive clockwise seen from above (towards East)
    pitch = positive nose-up
    roll  = positive right-wing-down
Yaw-rate in MAVLink (SET_POSITION_TARGET_LOCAL_NED.yaw_rate) is about the body Down axis, so
positive = clockwise from above = -(ENU z angular velocity).

Quaternions are (w, x, y, z), Hamilton convention, and rotate body -> world.
"""
from __future__ import annotations

import math
import numpy as np

# NED <-> ENU permutation (swap first two components, negate third).
C_NED_ENU = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]])
# FRD <-> FLU (negate y and z).
F_FRD_FLU = np.diag([1.0, -1.0, -1.0])


def ned_to_enu(v):
    return C_NED_ENU @ np.asarray(v, dtype=float)


def enu_to_ned(v):
    return C_NED_ENU @ np.asarray(v, dtype=float)


def frd_to_flu(v):
    return F_FRD_FLU @ np.asarray(v, dtype=float)


def flu_to_frd(v):
    return F_FRD_FLU @ np.asarray(v, dtype=float)


def R_ned_frd_from_enu_flu(R_enu_flu):
    return C_NED_ENU @ np.asarray(R_enu_flu, dtype=float) @ F_FRD_FLU


def R_enu_flu_from_ned_frd(R_ned_frd):
    return C_NED_ENU @ np.asarray(R_ned_frd, dtype=float) @ F_FRD_FLU


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------ Euler / rotation matrix
def R_from_euler(roll, pitch, yaw):
    """Body->world rotation matrix for the aerospace ZYX (yaw, pitch, roll) sequence."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def euler_from_R(R):
    """(roll, pitch, yaw) of a body->world rotation matrix, aerospace ZYX."""
    R = np.asarray(R, dtype=float)
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    if abs(R[2, 0]) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw


# ------------------------------------------------------------------ quaternions (w, x, y, z)
def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_from_R(R):
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = quat_normalize(q)
    return q if q[0] >= 0 else -q


def R_from_quat(q):
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


# ------------------------------------------------------------------ camera optical frame
# Optical frame convention (REP-103 / OpenCV): z forward, x right, y down.
# Camera *link* frame is FLU-like: x forward, y left, z up.
# Columns are the optical axes expressed in the link frame:
#   x_opt = (0,-1, 0)_link ; y_opt = (0, 0,-1)_link ; z_opt = (1, 0, 0)_link
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])


def ros_twist_to_ned(frame: str, lin, ang_z: float, yaw_ned: float):
    """
    Convert a ROS Twist (linear xyz, angular z) given in `frame` to MAVLink NED:
    returns (vn, ve, vd, yaw_rate) with yaw_rate positive CLOCKWISE seen from above.

      enu      linear = (East, North, Up),      angular.z CCW positive   (REP-103)
      ned      linear = (North, East, Down),    angular.z CW  positive
      body_flu linear = (forward, left, up) relative to the current heading `yaw_ned`, angular.z CCW positive
    """
    x, y, z = lin
    if frame == "enu":
        return y, x, -z, -ang_z
    if frame == "ned":
        return x, y, z, ang_z
    if frame == "body_flu":
        c, s = math.cos(yaw_ned), math.sin(yaw_ned)
        return x * c + y * s, x * s - y * c, -z, -ang_z
    raise ValueError(f"unknown frame {frame!r}")
