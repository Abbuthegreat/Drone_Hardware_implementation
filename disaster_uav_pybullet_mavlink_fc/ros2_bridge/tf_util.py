"""TF helpers: static transforms parsed from the URDF (legitimate calibration) and the SLAM pose -> TF.

Frames (REP-103/105): base_link (FLU, URDF origin) -> rgbd_camera_link (x fwd, 10 deg pitched) -> rgbd_camera_optical_link (z fwd, x right, y down)
                                                     -> thermal_camera_link -> thermal_camera_optical_link
                      slam_map (ENU, origin = take-off point) -> base_link  (dynamic, from /uav/slam/pose)
Exactly ONE node (the SLAM node) broadcasts these, so there are no duplicate / conflicting broadcasters.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
from geometry_msgs.msg import TransformStamped

from common.frames import quat_from_R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def urdf_fixed_joints(path=None, names=("rgbd_camera_joint", "rgbd_camera_optical_joint", "thermal_camera_joint")):
    """-> list of (parent, child, R, xyz) for the named fixed joints of the URDF."""
    root = ET.parse(path or os.path.join(ROOT, "urdf", "disaster_uav.urdf")).getroot()
    out = []
    for j in root.findall("joint"):
        if j.get("name") in names:
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
            out.append((j.find("parent").get("link"), j.find("child").get("link"), _rpy_R(*rpy), np.array(xyz)))
    return out


def make_tf(parent, child, R, t, stamp) -> TransformStamped:
    m = TransformStamped()
    m.header.stamp = stamp
    m.header.frame_id, m.child_frame_id = parent, child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in t)
    q = quat_from_R(R)                                        # (w, x, y, z)
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = (float(q[1]), float(q[2]), float(q[3]), float(q[0]))
    return m


def static_transforms(stamp):
    out = [make_tf(p, c, R, t, stamp) for p, c, R, t in urdf_fixed_joints()]
    # The URDF has no thermal optical joint; the thermal sensor uses the same REP-103 optical rotation as the RGB-D one.
    opt = [j for j in urdf_fixed_joints() if j[1] == "rgbd_camera_optical_link"][0]
    out.append(make_tf("thermal_camera_link", "thermal_camera_optical_link", opt[2], opt[3], stamp))
    return out
