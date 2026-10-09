"""
Run the PyBullet simulation with the simulated flight controller listening on MAVLink UDP.

    python -m sim.run_fc_sim --headless                 # FC on udpin:0.0.0.0:14540
    python -m sim.run_fc_sim --gui
    python -m sim.run_fc_sim --headless --duration 60 --log-dir logs/run1

This process is the "Pixhawk + physics" side. The companion computer / ROS 2 bridge talks to it ONLY
over MAVLink UDP. Ground truth is written to logs/<run>/ground_truth.csv (evaluation only).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.config import load_config  # noqa: E402
from sim.sim_core import UavSim  # noqa: E402
from sim.mavlink_fc_server import MavlinkFcServer  # noqa: E402
from eval.ground_truth import GroundTruthRecorder  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gui", action="store_true", help="open the PyBullet GUI")
    g.add_argument("--headless", action="store_true", help="no GUI (default)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=None, help="MAVLink UDP port (default from config)")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many simulated seconds (0 = run until Ctrl+C)")
    ap.add_argument("--log-dir", default=None, help="default logs/run_<timestamp>")
    ap.add_argument("--eval-port", type=int, default=None, help="UDP port of the evaluation-only ground-truth JSON stream")
    ap.add_argument("--rtf", type=float, default=None, help="real-time factor (0.5 = simulate at half speed; default sim.realtime_factor)")
    ap.add_argument("--no-realtime", action="store_true", help="run as fast as possible (watchdogs use simulation time)")
    ap.add_argument("--no-world", action="store_true", help="flat ground only (fast; for physics tests)")
    ap.add_argument("--scenario", choices=["static", "dynamic"], default=None, help="disaster scene variant (default: config world.scenario)")
    ap.add_argument("--ros-sensors", action="store_true", help="publish RGB-D camera_info/images on ROS 2 topics (needs rclpy)")
    ap.add_argument("--spawn-height", type=float, default=None, help="start this many metres above the ground (in the air)")
    ap.add_argument("--ready-file", default=None, help="write this file when the MAVLink server is up")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = load_config(a.config)
    if a.port:
        cfg["mavlink"]["udp_port"] = a.port
        cfg["mavlink"]["fc_bind"] = f"udpin:0.0.0.0:{a.port}"
    if a.eval_port:
        cfg["sim"]["eval_udp_port"] = a.eval_port
    if a.scenario:
        cfg["world"]["scenario"] = a.scenario
    if a.spawn_height is not None:
        cfg["spawn"]["height_above_ground_m"] = a.spawn_height
    log_dir = a.log_dir or os.path.join(ROOT, "logs", time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    sim = UavSim(cfg, gui=a.gui, world=not a.no_world)
    server = MavlinkFcServer(sim, log_path=os.path.join(log_dir, "fc_mavlink.log"))
    rec = GroundTruthRecorder(sim, log_dir, hz=cfg["sim"]["log_hz"], udp_port=cfg["sim"]["eval_udp_port"],
                              world_info=sim.world_info)
    ros_pub = None
    if a.ros_sensors:
        from sim.ros_sensor_publisher import RosSensorPublisher
        ros_pub = RosSensorPublisher(sim)
        sim.hooks.append(ros_pub.on_step)

    print(f"[fc-sim] MAVLink FC listening on {cfg['mavlink']['fc_bind']} | physics {sim.hz} Hz | "
          f"log dir {log_dir}", flush=True)
    if a.ready_file:
        with open(a.ready_file, "w") as f:
            json.dump({"pid": os.getpid(), "log_dir": log_dir, "port": cfg["mavlink"]["udp_port"]}, f)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: stop.__setitem__("flag", True))
    stop_file = os.path.join(log_dir, "STOP")
    realtime = cfg["sim"]["realtime"] and not a.no_realtime
    rtf = a.rtf or cfg["sim"].get("realtime_factor", 1.0)
    wall0 = time.perf_counter()
    try:
        while not stop["flag"]:
            sim.step()
            if a.duration and sim.t >= a.duration:
                break
            if sim.step_count % 250 == 0 and os.path.exists(stop_file):
                break
            if realtime:
                ahead = wall0 + sim.t / rtf - time.perf_counter()
                if ahead > 0.002:
                    time.sleep(ahead - 0.001)
                elif ahead < -1.0:            # more than 1 s behind wall clock: re-anchor, never "catch up"
                    wall0 = time.perf_counter() - sim.t / rtf
            if a.gui and sim.step_count % 20 == 0:
                import pybullet as p
                p.resetDebugVisualizerCamera(6, 50, -25, sim.state.pos_enu.tolist(), physicsClientId=sim.pb)
    finally:
        with open(os.path.join(log_dir, "fc_mavlink_summary.json"), "w") as f:
            json.dump({"rx": dict(server.rx_counts), "tx": dict(server.tx_counts), "sim_time_s": sim.t,
                       "wall_time_s": time.perf_counter() - wall0}, f, indent=1)
        server.close()
        rec.close()
        sim.close()
        print(f"[fc-sim] stopped at t={sim.t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
