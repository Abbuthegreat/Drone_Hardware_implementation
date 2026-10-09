"""
Demo video compositor (ROS 2 node). One video frame is written per third-person "main view" frame (simulation time, config video.fps).

Layout (1280 x 720):
    +-------------------------------+--------------------+
    |  MAIN VIEW  (PyBullet chase)  |  LIVE GEO-SEMANTIC |
    |  + HUD / mission / VFH+ tags  |  MAP (A*, markers) |
    +---------------+---------------+---------+----------+
    | RGB + detect  | DEPTH (metres)| THERMAL | STATUS   |
    +---------------+---------------+---------+----------+
Only autonomy-side topics and the visual-only chase view are used.
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from common.config import load_config
from ros2_bridge.ros_util import img_to_np, stamp_to_s
from video.textutil import dim_rect, put_text

W, H = 1280, 720


def _fit(img, w, h, fill=(24, 24, 24)):
    if img is None:
        out = np.zeros((h, w, 3), np.uint8)
        out[:] = fill
        return out
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.zeros((h, w, 3), np.uint8)
    out[:] = fill
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def _text(img, s, org, scale=0.55, col=(255, 255, 255), thick=1, bg=True):
    put_text(img, s, org, scale, col, thick, outline=bg)


class Compositor(Node):
    def __init__(self):
        super().__init__("demo_compositor")
        self.cfg = load_config()
        v = self.cfg["video"]
        self.out_path = self.declare_parameter("output", os.path.join(ROOT, v["output"])).value
        self.log_dir = self.declare_parameter("log_dir", "").value
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.vw = None
        for codec in ("avc1", "mp4v"):                     # H.264 when this OpenCV build has it (smaller, plays everywhere), else MPEG-4 part 2
            w = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*codec), float(v["fps"]), (W, H))
            if w.isOpened():
                self.vw, self.codec = w, codec
                break
        if self.vw is None:
            raise RuntimeError("no usable video codec in this OpenCV build")
        self.latest = {"panel": None, "rgb": None, "depth": None, "thermal": None}
        self.tel, self.nav, self.slam, self.sem = {}, {}, {}, {}
        self.banner_until, self.banner_text = -1.0, ""
        self.known_confirmed = set()
        self.frames_written = 0
        self.t_sim = 0.0
        self.t_first = None
        self.done = False
        c = self.create_subscription
        c(Image, "/demo/main_view/image_raw", self.on_main, 5)
        c(Image, "/demo/map_panel", lambda m: self._store("panel", m), 3)
        c(Image, "/demo/rgb_annotated", lambda m: self._store("rgb", m), 3)
        c(Image, "/uav/camera/depth/image_raw", lambda m: self._store("depth", m), 3)
        c(Image, "/uav/camera/thermal/image_raw", lambda m: self._store("thermal", m), 3)
        c(String, "/uav/telemetry", lambda m: setattr(self, "tel", json.loads(m.data)), 5)
        c(String, "/uav/nav/status", lambda m: setattr(self, "nav", json.loads(m.data)), 5)
        c(String, "/uav/slam/status", lambda m: setattr(self, "slam", json.loads(m.data)), 5)
        c(String, "/uav/map/semantic", self.on_sem, 5)
        c(Bool, "/demo/mission_done", lambda m: setattr(self, "done", bool(m.data)), 2)
        c(Bool, "/demo/stop", self.on_stop, 2)
        self.get_logger().info(f"compositor writing {self.out_path} ({self.codec})")

    def on_stop(self, m):
        if m.data:
            self.get_logger().info("stop requested - closing the video")
            self.close()
            raise SystemExit(0)

    def _store(self, k, m):
        try:
            self.latest[k] = (img_to_np(m), m)
        except Exception:
            pass

    def on_sem(self, m):
        d = json.loads(m.data)
        self.sem = d
        for s in d.get("survivors", []):
            if s["status"] == "CONFIRMED" and s["track_id"] not in self.known_confirmed:
                self.known_confirmed.add(s["track_id"])
                self.banner_text = (f"SURVIVOR #{s['track_id']:02d} CONFIRMED   LAT {s['latitude']:.6f}  LON {s['longitude']:.6f}  "
                                    f"ALT {s['altitude']:.1f} m  CONF {s['confidence']:.2f}")
                self.banner_until = self.t_sim + 5.0
                if self.log_dir:
                    os.makedirs(os.path.join(self.log_dir, "frames"), exist_ok=True)
                    self._snap_pending = f"survivor_{s['track_id']:02d}"

    def _get(self, k):
        v = self.latest[k]
        return None if v is None else v[0]

    # ------------------------------------------------------------------ compose
    def on_main(self, m):
        self.t_sim = stamp_to_s(m.header.stamp)
        if self.t_first is None:
            self.t_first = self.t_sim
        main = _fit(img_to_np(m)[:, :, ::-1].copy() if m.encoding == "rgb8" else img_to_np(m).copy(), 800, 450)
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[0:450, 0:800] = main
        panel = self._get("panel")
        canvas[0:450, 800:1280] = _fit(panel, 480, 450)
        rgb = self._get("rgb")
        canvas[450:720, 0:360] = _fit(rgb[:, :, ::-1] if rgb is not None and self.latest["rgb"][1].encoding == "rgb8" else rgb, 360, 270)
        d = self._get("depth")
        if d is not None:
            dv = np.nan_to_num(d, nan=0.0)
            cm = cv2.applyColorMap(np.clip(dv / 10.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cm[~np.isfinite(d)] = 0
            canvas[450:720, 360:720] = _fit(cm, 360, 270)
        th = self._get("thermal")
        if th is not None:
            g = np.clip((th - 5.0) / 40.0, 0, 1)
            canvas[450:720, 720:1080] = _fit(cv2.applyColorMap((g * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), 360, 270)
        cv2.rectangle(canvas, (0, 450), (360, 720), (90, 90, 90), 1)
        cv2.rectangle(canvas, (360, 450), (720, 720), (90, 90, 90), 1)
        cv2.rectangle(canvas, (720, 450), (1080, 720), (90, 90, 90), 1)
        _text(canvas, "RGB + person detection (kit YOLOv8n)", (6, 470), 0.45, (255, 255, 255))
        _text(canvas, "DEPTH (metres, 0-10 m)", (366, 470), 0.45, (255, 255, 255))
        _text(canvas, "THERMAL (synthetic, 5-45 C)", (726, 470), 0.45, (255, 255, 255))
        self._hud(canvas)
        self._status(canvas)
        self.vw.write(canvas)
        self.frames_written += 1
        if getattr(self, "_snap_pending", None) and self.log_dir:
            cv2.imwrite(os.path.join(self.log_dir, "frames", self._snap_pending + ".png"), canvas)
            self._snap_pending = None

    def _hud(self, c):
        n, s, tl = self.nav, self.slam, self.tel
        t_rel = self.t_sim - (self.t_first or 0.0)
        dim_rect(c, 0, 0, 800, 124, 0.5)
        _text(c, "AUTONOMOUS DISASTER-RESPONSE UAV  |  PyBullet + MAVLink FC  |  SLAM + A* + VFH+", (8, 20), 0.5, (255, 255, 255))
        _text(c, f"t = {t_rel:6.1f} s (sim)", (8, 42), 0.55, (0, 255, 255))
        state = n.get("state", "-")
        _text(c, f"MISSION: {state}   TARGET: {n.get('target', '-')}", (8, 64), 0.55, (120, 255, 120))
        vfh = n.get("vfh") or {}
        if vfh.get("blocked"):
            _text(c, f"VFH+ AVOIDING OBSTACLE   deviation {vfh.get('deviation_deg', 0):+.0f} deg   nearest {vfh.get('nearest_obstacle_m') or 0:.1f} m",
                  (8, 88), 0.6, (60, 60, 255), 2)
        elif n.get("path_partial"):
            _text(c, "A* EXPLORING: routing through known-free space only", (8, 88), 0.5, (255, 220, 40))
        else:
            _text(c, "A* global route active", (8, 88), 0.5, (255, 220, 40))
        if state == "LOST_HOLD":
            _text(c, "SLAM TRACKING LOST - HOLDING", (8, 112), 0.65, (0, 0, 255), 2)
        if self.t_sim < self.banner_until:
            cv2.rectangle(c, (0, 396), (800, 450), (0, 0, 160), -1)
            _text(c, self.banner_text[:78], (8, 430), 0.55, (255, 255, 255), 1, bg=False)
            _text(c, "GEO-TAG CREATED", (8, 412), 0.5, (0, 255, 255), 1, bg=False)

    def _status(self, c):
        tl, n, s = self.tel, self.nav, self.slam
        gps = tl.get("gps", {})
        lines = [
            ("FLIGHT CONTROLLER", (0, 255, 255)),
            (f"mode {tl.get('mode', '-')}  {'ARMED' if tl.get('armed') else 'disarmed'}", (255, 255, 255)),
            (f"batt {tl.get('battery_v', float('nan')):.1f} V ({tl.get('battery_pct', -1)}%)", (255, 255, 255)),
            ("GPS (FC fused)", (0, 255, 255)),
            (f"lat {gps.get('lat', 0):.6f}", (255, 255, 255)),
            (f"lon {gps.get('lon', 0):.6f}", (255, 255, 255)),
            (f"alt {gps.get('alt', 0):.1f} m", (255, 255, 255)),
            ("SLAM (RGB-D + IMU)", (0, 255, 255)),
            (f"{s.get('state', '-')}  inl {s.get('inliers', '-')}  kf {s.get('keyframes', '-')}", (255, 255, 255)),
            ("MAP / NAV", (0, 255, 255)),
            (f"speed {n.get('speed', 0.0):.2f} m/s   replans {(n.get('stats') or {}).get('replans', 0)}", (255, 255, 255)),
            (f"survivors {len(self.known_confirmed)}   hazards {len([h for h in self.sem.get('hazards', []) if h.get('status') == 'CONFIRMED'])}", (255, 255, 255)),
        ]
        y = 470
        for txt, col in lines:
            _text(c, txt, (1088, y), 0.42, col, 1)
            y += 19

    def close(self):
        if self.vw is None:
            return
        self.vw.release()
        self.vw = None
        self.get_logger().info(f"video closed: {self.frames_written} frames -> {self.out_path}")


def main(args=None):
    rclpy.init(args=args)
    n = Compositor()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        n.close()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
