"""
Perception + mapping node: builds the live geo-semantic map from RGB-D + thermal + the SLAM pose.

Subscribes : /uav/camera/{color,depth,thermal}/image_raw + camera_info, /uav/slam/pose, /uav/slam/status, /uav/gps/fix,
             /uav/nav/path, /uav/slam/path
Publishes  : /uav/map/planning_grid   nav_msgs/OccupancyGrid  (0 FREE, 100 OCCUPIED, -1 UNKNOWN; header.stamp = sim time,
                                      info.map_load_time.sec = map version) - the layer A* and VFH+ use
             /uav/map/semantic        std_msgs/String JSON: survivor nodes (track_id, class, confidence, local xyz, lat/lon/alt, uncertainty,
                                      first/last seen, observation count, status, evidence frame), hazard nodes, geo anchor
             /uav/perception/detections  std_msgs/String JSON per frame
             /demo/rgb_annotated, /demo/map_panel  images for the video / operator (visual only)
Frames are processed on a worker thread (latest frame wins) so a slow detector never delays anything else.
No simulator / ground-truth input.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, NavSatFix
from std_msgs.msg import String

from common.config import load_config
from mapping.geo_anchor import GpsAnchor
from mapping.pipeline import PerceptionMappingPipeline
from ros2_bridge.ros_util import img_to_np, np_to_img, pose_to_matrix, s_to_stamp, stamp_to_s
from video.map_panel import MapPanel


class PerceptionMappingNode(Node):
    def __init__(self):
        super().__init__("uav_perception_mapping")
        self.cfg = load_config()
        self.log_dir = self.declare_parameter("log_dir", "").value
        self.K_rgb = self.K_th = None
        self.pipe = None
        self.frames = {}
        self.q = queue.Queue(maxsize=1)
        self.anchor = GpsAnchor(self.cfg["sensor_extrinsics"]["gps"]["xyz_body_flu_m"])
        self.heading_enu = None
        self.nav_path, self.slam_trail = [], []
        self.pose_xy, self.yaw = None, 0.0
        self.panel = MapPanel()
        self.n_processed = self.n_dropped = 0
        self.last_sem_t = -1.0
        self.front = (-1.0, None)             # (camera time, forward corridor clearance from depth [m])
        self.wps = [tuple(w["xy"]) for w in self.cfg["demo"]["search_waypoints_local"]]
        self.wp_index = 0
        c = self.create_subscription
        c(CameraInfo, "/uav/camera/color/camera_info", self.on_info_rgb, 2)
        c(CameraInfo, "/uav/camera/thermal/camera_info", self.on_info_th, 2)
        c(Image, "/uav/camera/color/image_raw", lambda m: self.on_img(m, "rgb"), 5)
        c(Image, "/uav/camera/depth/image_raw", lambda m: self.on_img(m, "depth"), 5)
        c(Image, "/uav/camera/thermal/image_raw", lambda m: self.on_img(m, "thermal"), 5)
        c(PoseStamped, "/uav/slam/pose", self.on_pose, 10)
        c(String, "/uav/slam/status", self.on_slam_status, 10)
        c(NavSatFix, "/uav/gps/fix", self.on_gps, 10)
        c(Path, "/uav/nav/path", lambda m: setattr(self, "nav_path", [(p.pose.position.x, p.pose.position.y) for p in m.poses]), 2)
        c(String, "/uav/nav/status", self.on_nav_status, 5)
        p = self.create_publisher
        self.p_grid = p(OccupancyGrid, "/uav/map/planning_grid", 2)
        self.p_sem = p(String, "/uav/map/semantic", 5)
        self.p_det = p(String, "/uav/perception/detections", 5)
        self.p_rgb = p(Image, "/demo/rgb_annotated", 3)
        self.p_panel = p(Image, "/demo/map_panel", 3)
        self.f_det = self.f_sem = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.f_det = open(os.path.join(self.log_dir, "detections.jsonl"), "w", buffering=1)
            self.f_sem = open(os.path.join(self.log_dir, "survivor_map.jsonl"), "w", buffering=1)
        self.stop = False
        self.worker = threading.Thread(target=self.work, daemon=True)
        self.worker.start()
        self.get_logger().info("perception+mapping node up")

    # ------------------------------------------------------------------ inputs
    def on_info_rgb(self, m):
        if self.K_rgb is None:
            self.K_rgb = {"fx": m.k[0], "fy": m.k[4], "cx": m.k[2], "cy": m.k[5], "width": m.width, "height": m.height}
        self._init()

    def on_info_th(self, m):
        if self.K_th is None:
            self.K_th = {"fx": m.k[0], "fy": m.k[4], "cx": m.k[2], "cy": m.k[5], "width": m.width, "height": m.height}
        self._init()

    def _init(self):
        if self.pipe is None and self.K_rgb and self.K_th:
            ev = os.path.join(self.log_dir, "evidence") if self.log_dir else None
            self.pipe = PerceptionMappingPipeline(self.cfg, self.K_rgb, self.K_th, evidence_dir=ev)
            self.get_logger().info("pipeline ready (detector loaded)")

    def on_img(self, m, kind):
        t = round(stamp_to_s(m.header.stamp), 4)
        self.frames.setdefault(t, {})[kind] = m
        for k in [k for k in self.frames if k < t - 4.0]:
            self.frames.pop(k)

    def on_slam_status(self, m):
        d = json.loads(m.data)
        if d.get("heading_enu") is not None:
            self.heading_enu = d["heading_enu"]

    def on_gps(self, m: NavSatFix):
        if self.pipe is not None and not self.pipe.map.geo.ready:
            self.anchor.add(m.latitude, m.longitude, m.altitude, 3 if m.status.status >= 0 else 0)

    def on_nav_status(self, m):
        d = json.loads(m.data)
        self.wp_index = d.get("wp_index", self.wp_index)

    def on_pose(self, m: PoseStamped):
        if self.pipe is None:
            return
        t = round(stamp_to_s(m.header.stamp), 4)
        T = pose_to_matrix(m)
        self.pose_xy = (T[0, 3], T[1, 3])
        self.yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        if not self.slam_trail or np.hypot(self.pose_xy[0] - self.slam_trail[-1][0], self.pose_xy[1] - self.slam_trail[-1][1]) > 0.2:
            self.slam_trail.append(self.pose_xy)
        if not self.pipe.map.geo.ready and self.heading_enu is not None:
            r = self.anchor.try_finish(self.heading_enu)
            if r:
                self.pipe.map.geo.set_anchor(*r)
                self.get_logger().info(f"GPS anchor set: {r}")
        fr = self.frames.pop(t, None)
        if fr is None or not all(k in fr for k in ("rgb", "depth", "thermal")) or not self.pipe.map.geo.ready:
            return
        try:
            self.q.put_nowait((t, fr, T))
        except queue.Full:
            try:
                self.q.get_nowait()
                self.n_dropped += 1
            except queue.Empty:
                pass
            self.q.put_nowait((t, fr, T))

    # ------------------------------------------------------------------ worker
    def work(self):
        while not self.stop:
            try:
                t, fr, T = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process(t, fr, T)
            except Exception as e:                                # noqa: BLE001
                import traceback
                self.get_logger().error(traceback.format_exc())

    def process(self, t, fr, T):
        rgb = img_to_np(fr["rgb"])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth = img_to_np(fr["depth"])
        thermal = img_to_np(fr["thermal"])
        n_conf0 = len(self.pipe.map.survivors.confirmed())
        res = self.pipe.process(t, bgr, depth, thermal, T)
        self.front = (t, res.get("front_clearance_m"))
        self.n_processed += 1
        ann = self.pipe.annotate(bgr, res)
        self.p_rgb.publish(np_to_img(ann, "bgr8", t, "rgbd_camera_optical_link"))
        det = {"t": t, "frame": self.pipe.frame_idx, "ms": round(res["ms"], 1),
               "persons": [{"box": [round(v, 1) for v in p["box"]], "conf_rgb": round(p["conf_rgb"], 3), "conf": round(p["conf"], 3),
                            "range": p.get("range"), "track_id": p.get("track_id"),
                            "depth_spread": None if p.get("depth_spread") is None else round(float(p["depth_spread"]), 2), "rejected": p.get("rejected"),
                            "thermal": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in (p.get("thermal") or {}).items() if k != "uv"}}
                           for p in res["persons"]],
               "hazards": [{"kind": h["kind"], "xy": [round(float(h["xyz_map"][0]), 2), round(float(h["xyz_map"][1]), 2)], "temp_c": round(h["temp_c"], 0)}
                           for h in res["hazards"]]}
        self.p_det.publish(String(data=json.dumps(det)))
        if self.f_det:
            self.f_det.write(json.dumps(det) + "\n")
        # map products
        z_plan = self.cfg["demo"]["flight_altitude_m"]
        panel = self.panel.render(self.pipe.map, z_plan, self.pose_xy, self.yaw, self.slam_trail, self.nav_path, self.wps, self.wp_index, t)
        self.p_panel.publish(np_to_img(panel, "bgr8", t, "map"))
        if t - self.last_sem_t >= 0.5 or len(self.pipe.map.survivors.confirmed()) != n_conf0 or self.front_changed():
            self.last_sem_t = t
            self.publish_map(t, z_plan)

    def front_changed(self):
        """Forward depth clearance is a safety signal: republish at the camera rate when it is small, otherwise with the 0.5 s map update."""
        f = self.front[1]
        return f is not None and f < self.cfg["nav"].get("front_guard", {}).get("stop_m", 1.0) + 0.5

    def publish_map(self, t, z_plan):
        m = self.pipe.map
        g = m.planning_grid(z_plan)
        og = OccupancyGrid()
        og.header.stamp, og.header.frame_id = s_to_stamp(t), "slam_map"
        og.info.resolution = float(m.metric.v)
        og.info.width, og.info.height = int(g.shape[0]), int(g.shape[1])
        og.info.origin.position.x, og.info.origin.position.y = float(m.metric.origin[0]), float(m.metric.origin[1])
        og.info.map_load_time.sec = int(m.metric.version)
        og.data = g.T.reshape(-1).astype(np.int8).tolist()
        self.p_grid.publish(og)
        snap = m.snapshot(t)
        snap["confirmed_ids"] = [s.track_id for s in m.survivors.confirmed()]
        snap["front_clearance_m"], snap["front_clearance_t"] = self.front[1], self.front[0]      # depth-image corridor clearance (camera frame), SLAM-independent
        snap["events"] = m.survivors.events[-5:] + m.hazards.events[-5:]
        mm = m.metric
        snap["map"]["dynamic_memory_events"] = int(getattr(mm, "n_dynamic_events", 0))                 # analysis: moving-obstacle memory activity
        snap["map"]["dynamic_memory_cells"] = int((mm.dyn_until > mm.t_now).sum()) if hasattr(mm, "dyn_until") else 0
        self.p_sem.publish(String(data=json.dumps(snap)))
        if self.f_sem:
            self.f_sem.write(json.dumps(snap) + "\n")

    def destroy_node(self):
        self.stop = True
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    n = PerceptionMappingNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
