"""
ONE command for the final demonstration:

    python scripts/run_disaster_demo.py                       # full autonomous mission + recorded video + logs
    python scripts/run_disaster_demo.py --no-video            # same mission without the compositor (faster)
    python scripts/run_disaster_demo.py --scenario dynamic    # + secondary collapse / moving debris (forces live re-planning)
    python scripts/run_disaster_demo.py --slam-backend orb_pnp_vo   # choose the SLAM backend (default auto: ORB-SLAM3 if built, else VO)

Starts, as separate processes on one ROS 2 domain:
    1. PyBullet simulation + simulated flight controller (MAVLink UDP 14540) + RGB / depth / thermal / chase-view publishers
    2. ROS 2 <-> MAVLink bridge (/uav/cmd_vel ENU -> SET_POSITION_TARGET_LOCAL_NED; telemetry, IMU, GPS back)
    3. SLAM node          (RGB-D + IMU -> pose)
    4. Perception + mapping node (kit YOLO detector, depth localisation, thermal fusion, geo-semantic map, tracker)
    5. Navigation node    (deterministic search mission -> A* + VFH+ -> /uav/cmd_vel)
    6. Video compositor   (records logs/demo/<run>/disaster_demo.mp4)
then waits for the mission to finish, shuts everything down cleanly, scores the run against ground truth OFFLINE and writes the logs.
While it runs it writes the command-centre stream command_centre/status.jsonl (1 Hz, mission/companion data only) and at the end
rescue_report.json / rescue_report.md (geo-tagged survivors, hazards, mission events).
Requires the ROS 2 + pybullet environment (see README). Ground truth is only read by the scoring step after the flight.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ROS_DOMAIN_ID", "88")

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import Bool, String

from common.config import load_config
from mission.command_centre import one_line, rescue_report, status_record
from ros2_bridge.ros_util import stamp_to_s
from slam.orbslam3_backend import resolve_backend


class Monitor(Node):
    def __init__(self):
        super().__init__("demo_runner")
        self.done = False
        self.t_sim = 0.0
        self.nav = {}
        self.sem = {}
        self.slam = {}
        self.create_subscription(Bool, "/demo/mission_done", lambda m: setattr(self, "done", bool(m.data)), 2)
        self.create_subscription(Odometry, "/uav/odom", lambda m: setattr(self, "t_sim", stamp_to_s(m.header.stamp)), 5)
        self.create_subscription(String, "/uav/nav/status", lambda m: setattr(self, "nav", json.loads(m.data)), 5)
        self.create_subscription(String, "/uav/map/semantic", lambda m: setattr(self, "sem", json.loads(m.data)), 5)
        self.create_subscription(String, "/uav/slam/status", lambda m: setattr(self, "slam", json.loads(m.data)), 5)
        self.pub_stop = self.create_publisher(Bool, "/demo/stop", 2)
        self.run = None
        self.f_tel = self.f_gps = self.f_frames = None
        self.counts = {"rgb": 0, "depth": 0, "thermal": 0}
        self.tel = {}
        self.f_cc = None
        self.last_rec = None

    def command_centre(self, run):
        os.makedirs(os.path.join(run, "command_centre"), exist_ok=True)
        self.f_cc = open(os.path.join(run, "command_centre", "status.jsonl"), "w", buffering=1)

    def write_status(self):
        rec = status_record(time.time(), self.nav, self.sem, self.tel, self.slam)
        self.last_rec = rec
        if self.f_cc:
            self.f_cc.write(json.dumps(rec) + "\n")
        return rec

    def open_logs(self, run):
        os.makedirs(os.path.join(run, "telemetry"), exist_ok=True)
        self.f_tel = open(os.path.join(run, "telemetry", "fc_telemetry.csv"), "w", buffering=1)
        self.f_tel.write("t_wall,fc_time_ms,mode,armed,system_status,failsafe_critical,lat,lon,alt,battery_v,battery_pct,setpoints_sent\n")
        self.f_gps = open(os.path.join(run, "telemetry", "gps.csv"), "w", buffering=1)
        self.f_gps.write("t_sim,lat,lon,alt,status,cov_h_m2\n")
        self.f_frames = open(os.path.join(run, "telemetry", "frames.csv"), "w", buffering=1)
        self.f_frames.write("stream,index,t_sim\n")
        self.create_subscription(String, "/uav/telemetry", self._on_tel, 20)
        self.create_subscription(NavSatFix, "/uav/gps/fix", self._on_gps, 20)
        for k, tp in (("rgb", "/uav/camera/color/image_raw"), ("depth", "/uav/camera/depth/image_raw"), ("thermal", "/uav/camera/thermal/image_raw")):
            self.create_subscription(Image, tp, lambda m, k=k: self._on_img(k, m), 20)

    def _on_tel(self, m):
        d = json.loads(m.data)
        self.tel = d
        g = d.get("gps", {})
        self.f_tel.write(f"{time.time():.3f},{d.get('fc_time_boot_ms')},{d.get('mode')},{int(bool(d.get('armed')))},{d.get('system_status')},"
                         f"{int(d.get('system_status') == 5)},{g.get('lat')},{g.get('lon')},{g.get('alt')},{d.get('battery_v')},{d.get('battery_pct')},{d.get('setpoints_sent')}\n")

    def _on_gps(self, m):
        self.f_gps.write(f"{stamp_to_s(m.header.stamp):.3f},{m.latitude:.8f},{m.longitude:.8f},{m.altitude:.2f},{m.status.status},{m.position_covariance[0]:.3f}\n")

    def _on_img(self, k, m):
        self.f_frames.write(f"{k},{self.counts[k]},{stamp_to_s(m.header.stamp):.3f}\n")
        self.counts[k] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--rtf", type=float, default=None)
    ap.add_argument("--max-wall-min", type=float, default=40.0)
    ap.add_argument("--slam-backend", default="auto", choices=["auto", "orbslam3", "orb_pnp_vo"],
                    help="auto = config slam.backend when it can run here (ORB-SLAM3 needs the Linux build), else the orb_pnp_vo fallback")
    ap.add_argument("--scenario", default=None, choices=["static", "dynamic"], help="disaster scene variant (default: config world.scenario)")
    a = ap.parse_args()
    cfg = load_config()
    slam_backend, slam_note = resolve_backend(cfg["slam"]["backend"], a.slam_backend)
    scenario = a.scenario or cfg["world"].get("scenario", "static")
    run = a.out or os.path.join(ROOT, "logs", "demo", time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=True)
    rtf = a.rtf or cfg["demo"]["sim_rtf"]
    py = sys.executable
    env = dict(os.environ)
    procs = {}

    def start(name, args):
        f = open(os.path.join(run, f"{name}.log"), "w")
        procs[name] = (subprocess.Popen([py, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env), f)
        print(f"[demo] started {name} (pid {procs[name][0].pid})", flush=True)

    print(f"[demo] run directory: {run}", flush=True)
    print(f"[demo] SLAM backend: {slam_backend}" + (f"   ** {slam_note}" if slam_note else ""), flush=True)
    print(f"[demo] scenario: {scenario}", flush=True)
    start("sim", ["-m", "sim.run_fc_sim", "--headless", "--ros-sensors", "--rtf", str(rtf), "--log-dir", os.path.join(run, "sim"), "--scenario", scenario,
                  "--ready-file", os.path.join(run, "sim_ready.json"), "--duration", str(cfg["demo"]["duration_limit_s"] + 30)])
    t0 = time.time()
    while not os.path.exists(os.path.join(run, "sim_ready.json")):
        if procs["sim"][0].poll() is not None:
            sys.exit("simulation died at start-up; see sim.log")
        if time.time() - t0 > 120:
            sys.exit("simulation did not start")
        time.sleep(0.5)
    start("bridge", ["-m", "ros2_bridge.uav_mavlink_bridge", "--ros-args", "-p", "cmd_frame:=enu",
                     "-p", f"log_path:={os.path.join(run, 'companion_mavlink.log')}"])
    start("slam", ["-m", "ros2_bridge.slam_node", "--ros-args", "-p", f"log_dir:={os.path.join(run, 'slam')}", "-p", f"backend:={slam_backend}"])
    start("perception_mapping", ["-m", "ros2_bridge.perception_mapping_node", "--ros-args", "-p", f"log_dir:={os.path.join(run, 'perception')}"])
    start("nav", ["-m", "ros2_bridge.nav_node", "--ros-args", "-p", f"log_dir:={os.path.join(run, 'nav')}"])
    video = os.path.join(run, "disaster_demo.mp4")
    if not a.no_video:
        start("video", ["-m", "video.compositor_node", "--ros-args", "-p", f"output:={video}", "-p", f"log_dir:={os.path.join(run, 'video')}"])

    rclpy.init()
    mon = Monitor()
    mon.open_logs(run)
    mon.command_centre(run)
    t_wall0 = time.time()
    last_print = 0.0
    last_cc = 0.0
    reason = "unknown"
    try:
        while rclpy.ok():
            rclpy.spin_once(mon, timeout_sec=0.2)
            now = time.time()
            if now - last_cc >= 1.0:
                last_cc = now
                rec = mon.write_status()
                if now - last_print > 5.0:
                    last_print = now
                    print(f"[cc] wall {now - t_wall0:5.0f}s  {one_line(rec)}  slam={mon.slam.get('state','-')}", flush=True)
            if mon.done:
                reason = "mission complete"
                break
            if mon.t_sim > cfg["demo"]["duration_limit_s"]:
                reason = "simulation-time limit"
                break
            if now - t_wall0 > a.max_wall_min * 60:
                reason = "wall-clock limit"
                break
            for name, (p, _) in procs.items():
                if p.poll() is not None and name in ("sim", "bridge", "slam", "perception_mapping", "nav"):
                    reason = f"{name} exited early ({p.returncode})"
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    print(f"[demo] stopping: {reason}", flush=True)
    t_end = time.time()
    while time.time() - t_end < 2.5:                          # let the last frames reach the compositor
        rclpy.spin_once(mon, timeout_sec=0.1)
    for _ in range(10):
        mon.pub_stop.publish(Bool(data=True))
        rclpy.spin_once(mon, timeout_sec=0.1)
    open(os.path.join(run, "sim", "STOP"), "w").close()
    try:                                                     # let the simulator exit by itself (writes fc_mavlink_summary.json);
        procs["sim"][0].wait(timeout=60)                     # terminate() on Windows is a hard kill that skips its clean-up
    except subprocess.TimeoutExpired:
        print("[demo] simulator did not stop within 60 s; terminating", flush=True)
    for name in ("video",):
        if name in procs:
            try:
                procs[name][0].wait(timeout=20)
            except subprocess.TimeoutExpired:
                procs[name][0].kill()
    time.sleep(1.0)
    summary_sem = mon.sem
    mon.destroy_node()
    rclpy.shutdown()
    for name, (p, f) in procs.items():
        if p.poll() is None:
            p.terminate()
    for name, (p, f) in procs.items():
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
        f.close()
    final_rec = status_record(time.time(), mon.nav, summary_sem, mon.tel, mon.slam)
    summary = {"reason": reason, "wall_s": time.time() - t_wall0, "sim_s": mon.t_sim, "scenario": scenario, "slam_backend": slam_backend,
               "slam_backend_note": slam_note, "final_state": mon.nav.get("state"), "end_reason": mon.nav.get("end_reason"),
               "skipped_waypoints": mon.nav.get("skipped_waypoints", []), "nav_stats": mon.nav.get("stats", {}),
               "fc_final": final_rec["flight_controller"], "mavlink_final": final_rec["mavlink"], "semantic_map_final": summary_sem}
    json.dump(summary, open(os.path.join(run, "demo_summary.json"), "w"), indent=1)
    events = []
    ev_path = os.path.join(run, "nav", "nav_events.jsonl")
    if os.path.exists(ev_path):
        events = [e for e in (json.loads(l) for l in open(ev_path) if l.strip()) if e.get("kind") == "event"]
    doc, md = rescue_report(final_rec, events, {k: summary[k] for k in ("reason", "final_state", "sim_s", "scenario", "slam_backend", "end_reason")})
    json.dump(doc, open(os.path.join(run, "rescue_report.json"), "w"), indent=1, default=float)
    open(os.path.join(run, "rescue_report.md"), "w", encoding="utf-8").write(md)
    print(md, flush=True)
    # ---- offline scoring against ground truth (evaluation only)
    try:
        from eval.demo_scoring import score
        rep = score(run)
        print(json.dumps(rep, indent=1, default=float))
    except Exception as e:                                  # noqa: BLE001
        print(f"[demo] scoring failed: {e!r}")
    if not a.no_video and os.path.exists(video):
        print(f"[demo] VIDEO: {video}  ({os.path.getsize(video)/1e6:.1f} MB)")
    print(f"[demo] logs: {run}")


if __name__ == "__main__":
    main()
