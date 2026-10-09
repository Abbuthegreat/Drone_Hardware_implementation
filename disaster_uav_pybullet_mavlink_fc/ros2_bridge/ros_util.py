"""ROS 2 helpers shared by the autonomy nodes (no cv_bridge needed). Autonomy side: must not import the simulator."""
from __future__ import annotations

import array

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image


def stamp_to_s(st) -> float:
    return st.sec + st.nanosec * 1e-9


def s_to_stamp(t: float) -> Time:
    m = Time()
    m.sec, m.nanosec = int(t), int(round((t % 1.0) * 1e9))
    if m.nanosec >= 1_000_000_000:
        m.sec, m.nanosec = m.sec + 1, 0
    return m


def img_to_np(m: Image) -> np.ndarray:
    if m.encoding in ("rgb8", "bgr8"):
        return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
    if m.encoding == "32FC1":
        return np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
    if m.encoding == "mono8":
        return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
    raise ValueError(f"unsupported encoding {m.encoding}")


def np_to_img(arr: np.ndarray, encoding: str, t: float, frame_id: str) -> Image:
    m = Image()
    m.header.stamp, m.header.frame_id = s_to_stamp(t), frame_id
    m.height, m.width, m.encoding = arr.shape[0], arr.shape[1], encoding
    m.step = int(arr.strides[0])
    ar = array.array("B")                      # fast path (bytes would be validated element by element)
    ar.frombytes(np.ascontiguousarray(arr).tobytes())
    m.data = ar
    return m


def matrix_to_pose(T: np.ndarray, t: float, frame_id: str) -> PoseStamped:
    from common.frames import quat_from_R
    p = PoseStamped()
    p.header.stamp, p.header.frame_id = s_to_stamp(t), frame_id
    p.pose.position.x, p.pose.position.y, p.pose.position.z = map(float, T[:3, 3])
    w, x, y, z = quat_from_R(T[:3, :3])
    p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = float(x), float(y), float(z), float(w)
    return p


def pose_to_matrix(p: PoseStamped) -> np.ndarray:
    from common.frames import R_from_quat
    q = p.pose.orientation
    T = np.eye(4)
    T[:3, :3] = R_from_quat([q.w, q.x, q.y, q.z])
    T[:3, 3] = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
    return T
