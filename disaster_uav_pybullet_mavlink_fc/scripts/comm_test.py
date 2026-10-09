"""
Dedicated ROS 2 -> MAVLink -> FC -> PyBullet -> FC -> MAVLink -> ROS 2 communication test.
NO autonomous algorithm is involved. Requires rclpy (ROS 2) + pybullet + pymavlink in ONE python env.

    python scripts/comm_test.py [--no-world] [--out logs/comm_test_x]

Steps (each reports PASS/FAIL with the measured evidence):
  1  start PyBullet + simulated FC (MAVLink server, UDP 14540)      9  physical response to /uav/cmd_vel (climb)
  2  start the ROS 2 <-> MAVLink bridge                             10 change velocity command; observe acceleration + displacement
  3  wait for HEARTBEAT                                             11 ros2 CLI: topic list / echo / pub (real ROS command -> motion)
  4  heartbeat + telemetry topics received                          12 stop commands (setpoint stream stops)
  5  arm command                                                    13 failsafe / hold behaviour
  6  COMMAND_ACK verified (ROS response + FC log)                   14 land
  7  continuous setpoint stream verified at the FC (20 Hz)          15 disarm
  8  OFFBOARD mode accepted                                         16 motor thrust falls
Ground truth (evaluation-only UDP stream) is used ONLY to check the results.
Outputs: <out>/comm_test.log, report.json, REPORT.md, fc_mavlink.log, ground_truth.csv, bridge.log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ROS_DOMAIN_ID", "77")

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from eval.ground_truth import EvalListener      # evaluation-side only (this is a test harness, not autonomy)
from tests.fc_process import FcProcess

PORT, EVAL_PORT = 14540, 14560


class Report:
    def __init__(self, out):
        self.out, self.steps = out, []
        self.f = open(os.path.join(out, "comm_test.log"), "w", buffering=1)

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")

    def step(self, n, name, ok, evidence):
        self.steps.append({"step": n, "name": name, "result": "PASS" if ok else "FAIL", "evidence": evidence})
        self.log(f"STEP {n:2d} {'PASS' if ok else 'FAIL'}  {name}  |  {evidence}")

    def finish(self):
        allok = all(s["result"] == "PASS" for s in self.steps)
        json.dump({"overall": "PASS" if allok else "FAIL", "steps": self.steps}, open(os.path.join(self.out, "report.json"), "w"), indent=1)
        with open(os.path.join(self.out, "REPORT.md"), "w") as f:
            f.write(f"# Communication test: {'PASS' if allok else 'FAIL'}\n\n| # | step | result | evidence |\n|---|---|---|---|\n")
            for s in self.steps:
                f.write(f"| {s['step']} | {s['name']} | **{s['result']}** | {s['evidence']} |\n")
        self.log(f"OVERALL: {'PASS' if allok else 'FAIL'}  ({sum(s['result']=='PASS' for s in self.steps)}/{len(self.steps)} steps)")
        self.f.close()
        return allok


class Driver:
    def __init__(self, ev):
        rclpy.init()
        self.node = rclpy.create_node("comm_test_driver")
        self.ev = ev
        self.tel = None
        self.tel_times = []
        self.odom = None
        self.n = {"imu": 0, "gps": 0, "odom": 0, "mode": 0}
        self.node.create_subscription(String, "/uav/telemetry", self._on_tel, 10)
        self.node.create_subscription(Odometry, "/uav/odom", lambda m: (setattr(self, "odom", m), self.n.__setitem__("odom", self.n["odom"] + 1)), 10)
        self.node.create_subscription(Imu, "/uav/imu", lambda m: self.n.__setitem__("imu", self.n["imu"] + 1), 200)
        self.node.create_subscription(NavSatFix, "/uav/gps/fix", lambda m: self.n.__setitem__("gps", self.n["gps"] + 1), 10)
        self.node.create_subscription(String, "/uav/mode", lambda m: self.n.__setitem__("mode", self.n["mode"] + 1), 10)
        self.pub_cmd = self.node.create_publisher(Twist, "/uav/cmd_vel", 10)
        self.clients = {n: self.node.create_client(t, n) for n, t in
                        (("/uav/arm", SetBool), ("/uav/offboard", SetBool), ("/uav/land", Trigger),
                         ("/uav/setpoint_stream", SetBool), ("/uav/force_disarm", Trigger))}

    def _on_tel(self, m):
        self.tel = json.loads(m.data)
        self.tel_times.append(time.time())

    def spin(self, seconds):
        t_end = time.time() + seconds
        while time.time() < t_end:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def wait_for(self, cond, timeout, period=0.05):
        t_end = time.time() + timeout
        while time.time() < t_end:
            rclpy.spin_once(self.node, timeout_sec=period)
            if cond():
                return True
        return False

    def call(self, name, req, timeout=5.0):
        cl = self.clients[name]
        if not cl.wait_for_service(timeout_sec=timeout):
            return None
        fut = cl.call_async(req)
        t_end = time.time() + timeout
        while time.time() < t_end and not fut.done():
            rclpy.spin_once(self.node, timeout_sec=0.02)
        return fut.result() if fut.done() else None

    def cmd(self, x=0.0, y=0.0, z=0.0, wz=0.0):
        m = Twist()
        m.linear.x, m.linear.y, m.linear.z, m.angular.z = float(x), float(y), float(z), float(wz)
        self.pub_cmd.publish(m)

    def stream_cmd(self, seconds, **kw):
        t_end = time.time() + seconds
        while time.time() < t_end:
            self.cmd(**kw)
            self.spin(0.05)

    def truth(self):
        return self.ev.latest()


def ros2_cli(*args, timeout=60):
    """Run the real `ros2` CLI (same python env). Uses --no-daemon where supported."""
    script = os.path.join(os.path.dirname(sys.executable), "Library", "bin", "ros2-script.py")
    if not os.path.exists(script):
        script = None
    cmd = [sys.executable, script, *args] if script else ["ros2", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=dict(os.environ))
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return -1, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-world", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "logs", time.strftime("comm_test_%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    R = Report(out)
    R.log(f"python {sys.executable}; output {out}")
    fc = bridge = drv = None
    ev = EvalListener(EVAL_PORT)
    try:
        # ---- 1. PyBullet + FC
        fc = FcProcess(PORT, EVAL_PORT, log_dir=out, world=not a.no_world)
        fc.wait_ready(120)
        R.step(1, "PyBullet + simulated FC MAVLink server started", fc.proc.poll() is None,
               f"pid {fc.proc.pid}, udpin:0.0.0.0:{PORT}, world={'earthquake scene' if not a.no_world else 'flat ground'}")
        # ---- 2. bridge
        blog = open(os.path.join(out, "bridge.log"), "w")
        bridge = subprocess.Popen([sys.executable, "-m", "ros2_bridge.uav_mavlink_bridge", "--ros-args", "-p",
                                   f"connection:=udp:127.0.0.1:{PORT}", "-p", f"log_path:={os.path.join(out, 'companion_mavlink.log')}"],
                                  cwd=ROOT, stdout=blog, stderr=subprocess.STDOUT)
        drv = Driver(ev)
        R.step(2, "ROS 2 <-> MAVLink bridge node started", bridge.poll() is None, f"pid {bridge.pid}, connection udp:127.0.0.1:{PORT}")
        # ---- 3/4. heartbeat + telemetry
        got_hb = drv.wait_for(lambda: drv.tel is not None and drv.n["mode"] > 0, 40)
        blog.flush()
        bl = open(os.path.join(out, "bridge.log")).read()
        R.step(3, "HEARTBEAT received from FC", got_hb and "FC heartbeat received" in bl,
               "bridge log: 'FC heartbeat received (sys 1)'; /uav/mode publishing" if got_hb else "no heartbeat")
        drv.spin(3.0)
        rate = len(drv.tel_times[-60:]) / max(drv.tel_times[-1] - drv.tel_times[-min(len(drv.tel_times), 60)], 1e-3) if len(drv.tel_times) > 2 else 0
        keys = ["position_enu", "velocity_enu", "orientation_xyzw_enu_flu", "gps", "mode", "armed"]
        ok4 = drv.tel is not None and all(k in drv.tel for k in keys) and drv.n["odom"] > 20 and drv.n["imu"] > 100 and drv.n["gps"] > 5 and rate > 10
        R.step(4, "telemetry received (/uav/telemetry, /uav/odom, /uav/imu, /uav/gps/fix)", ok4,
               f"telemetry {rate:.1f} Hz, odom n={drv.n['odom']}, imu n={drv.n['imu']}, gps n={drv.n['gps']}, mode={drv.tel['mode']}, armed={drv.tel['armed']}")
        # reverse-path check on the ground: ROS position vs PyBullet truth
        tr = drv.truth()
        pos_ros = np.array(drv.tel["position_enu"])
        R.log(f"reverse telemetry check (ground): ROS ENU {pos_ros.round(3)} vs PyBullet truth {np.array(tr['pos_enu']).round(3)}")
        # ---- 5/6. arm + ACK
        r = drv.call("/uav/arm", SetBool.Request(data=True))
        drv.spin(0.5)
        R.step(5, "arm command sent (ROS service /uav/arm -> MAVLink COMMAND_LONG)", r is not None and r.success and drv.tel["armed"],
               f"service: success={getattr(r,'success',None)} message={getattr(r,'message',None)}; telemetry armed={drv.tel['armed']}")
        fclog = open(os.path.join(out, "fc_mavlink.log")).read()
        R.step(6, "COMMAND_ACK verified", r is not None and "MAV_RESULT=0" in r.message and "COMMAND_ACK command=400 result=0" in fclog,
               "ROS response MAV_RESULT=0 (ACCEPTED); FC log: 'TX COMMAND_ACK command=400 result=0'")
        # ---- 7. continuous setpoint stream
        n0 = drv.tel["setpoints_sent"]; t0 = time.time(); drv.spin(2.0)
        hz = (drv.tel["setpoints_sent"] - n0) / (time.time() - t0)
        R.step(7, "setpoints streamed continuously by the bridge (offboard-style)", 15 <= hz <= 25, f"{hz:.1f} Hz measured from bridge counter (configured 20 Hz)")
        # ---- 8. offboard
        r = drv.call("/uav/offboard", SetBool.Request(data=True))
        drv.spin(0.6)
        R.step(8, "OFFBOARD mode accepted", r is not None and r.success and drv.tel["mode"] == "OFFBOARD",
               f"service success={getattr(r,'success',None)}, telemetry mode={drv.tel['mode']}")
        # ---- 9. physical response: climb
        z0 = drv.truth()["pos_enu"][2]
        drv.stream_cmd(5.0, z=1.0)                      # ENU +z = up 1 m/s
        tr = drv.truth()
        dz = tr["pos_enu"][2] - z0
        thr = sum(tr["thrust_n"])
        R.step(9, "vehicle responds physically to /uav/cmd_vel (climb)", dz > 2.0 and tr["vel_ned"][2] < -0.6 and thr > 15,
               f"PyBullet altitude +{dz:.2f} m, vD={tr['vel_ned'][2]:.2f} m/s, total motor thrust {thr:.1f} N; ROS telemetry z={drv.tel['position_enu'][2]:.2f}")
        # ---- 10. change velocity: North (ENU +y) 1.5 m/s
        drv.stream_cmd(1.0, z=0.0)
        y0 = drv.truth()["pos_enu"][1]; ts, vs = [], []
        t_start = time.time()
        while time.time() - t_start < 4.0:
            drv.cmd(y=1.5); drv.spin(0.05)
            tr = drv.truth(); ts.append(tr["t"]); vs.append(tr["vel_ned"][0])
        tr = drv.truth()
        dy = tr["pos_enu"][1] - y0
        acc = (vs[min(20, len(vs) - 1)] - vs[0]) / max(ts[min(20, len(ts) - 1)] - ts[0], 1e-3)
        pitch = math.degrees(tr["rpy"][1])
        R.step(10, "velocity command changed -> acceleration + displacement observed", dy > 3.0 and acc > 0.4 and vs[-1] > 1.0,
               f"ENU +y (North) cmd 1.5 m/s: displacement {dy:.2f} m, initial accel {acc:.2f} m/s^2, final vN={vs[-1]:.2f} m/s, pitch {pitch:.1f} deg (nose-down)")
        pos_ros = np.array(drv.tel["position_enu"]); pos_true = np.array(tr["pos_enu"])
        R.log(f"reverse telemetry check (flight): ROS ENU {pos_ros.round(2)} vs PyBullet truth {pos_true.round(2)} (|diff|={np.linalg.norm(pos_ros-pos_true):.3f} m)")
        # ---- 11. ros2 CLI
        rc1, txt1 = ros2_cli("topic", "list", "--no-daemon", timeout=90)
        has_topics = "/uav/cmd_vel" in txt1 and "/uav/telemetry" in txt1
        rc2, txt2 = ros2_cli("topic", "echo", "--once", "--full-length", "/uav/telemetry", "--no-daemon", timeout=90)
        x0 = drv.truth()["pos_enu"][0]
        drv.cmd(0, 0, 0); drv.spin(0.2)
        _s = os.path.join(os.path.dirname(sys.executable), "Library", "bin", "ros2-script.py")
        _ros2 = [sys.executable, _s] if os.path.exists(_s) else ["ros2"]          # Windows RoboStack launcher | Linux `ros2`
        pub = subprocess.Popen([*_ros2,
                                "topic", "pub", "-r", "10", "--times", "60", "/uav/cmd_vel", "geometry_msgs/msg/Twist",
                                "{linear: {x: 1.5, y: 0.0, z: 0.0}}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        drv.spin(9.0)
        dx = drv.truth()["pos_enu"][0] - x0
        pub.terminate()
        R.step(11, "ros2 CLI: topic list / echo / pub cause real motion", has_topics and "position_enu" in txt2 and dx > 1.5,
               f"`ros2 topic list` shows /uav/cmd_vel + /uav/telemetry={has_topics}; `topic echo --once /uav/telemetry` returned JSON={'position_enu' in txt2}; "
               f"`ros2 topic pub /uav/cmd_vel linear.x=1.5 (East)` -> PyBullet East displacement {dx:.2f} m")
        with open(os.path.join(out, "ros2_cli_output.txt"), "w") as f:
            f.write("$ ros2 topic list --no-daemon\n" + txt1 + "\n$ ros2 topic echo --once /uav/telemetry\n" + txt2)
        # ---- 12/13. stop commands -> failsafe
        drv.stream_cmd(1.0, x=1.5)                       # moving East at speed
        v_before = float(np.linalg.norm(drv.truth()["vel_ned"]))
        r = drv.call("/uav/setpoint_stream", SetBool.Request(data=False))
        t_stop = time.time()
        R.step(12, "stop commands (setpoint stream stopped)", r is not None and r.success, f"speed at stop {v_before:.2f} m/s; {getattr(r,'message',None)}")
        ok_fs = drv.wait_for(lambda: drv.tel["mode"] == "FAILSAFE_HOLD", 3.0)
        t_fs = time.time() - t_stop
        drv.spin(6.0)
        tr = drv.truth()
        v_after = float(np.linalg.norm(tr["vel_ned"]))
        fclog = open(os.path.join(out, "fc_mavlink.log")).read()
        R.step(13, "failsafe / hold when setpoints stop", ok_fs and v_after < 0.35 and tr["pos_enu"][2] > 1.5 and "OFFBOARD_SETPOINT_LOSS" in fclog,
               f"mode FAILSAFE_HOLD after {t_fs:.2f} s (FC timeout 0.5 s + telemetry latency); speed {v_before:.2f}->{v_after:.2f} m/s; still airborne z={tr['pos_enu'][2]:.2f} m; FC log: 'Failsafe: OFFBOARD_SETPOINT_LOSS'")
        # ---- 14. land
        drv.call("/uav/setpoint_stream", SetBool.Request(data=True))
        r = drv.call("/uav/land", Trigger.Request())
        ok_land = drv.wait_for(lambda: bool(drv.truth()["landed"]), 40.0)
        tr = drv.truth()
        R.step(14, "land command -> vehicle descends and lands", r is not None and r.success and ok_land and tr["pos_enu"][2] < 0.3,
               f"LAND accepted={getattr(r,'success',None)}; landed flag={tr['landed']}, altitude {tr['pos_enu'][2]:.2f} m")
        # ---- 15. disarm
        hover_ref = 21.582
        r = drv.call("/uav/arm", SetBool.Request(data=False))
        t_dis = time.time()
        R.step(15, "disarm", r is not None and r.success, f"service success={getattr(r,'success',None)} {getattr(r,'message',None)}")
        # ---- 16. thrust falls
        drv.spin(1.0)
        tr = drv.truth()
        thr = sum(tr["thrust_n"]); rpm = max(tr["rpm"])
        R.step(16, "motor thrust falls after disarm", (not tr["armed"]) and thr < 0.05 * hover_ref and rpm < 300,
               f"armed={tr['armed']}, total thrust {thr:.3f} N (hover ~{hover_ref:.1f} N), max rpm {rpm:.0f}; ROS telemetry armed={drv.tel['armed']}")
    except Exception as e:                         # noqa: BLE001
        import traceback
        R.log("EXCEPTION: " + traceback.format_exc())
        R.steps.append({"step": len(R.steps) + 1, "name": "exception", "result": "FAIL", "evidence": repr(e)})
    finally:
        try:
            if drv:
                drv.node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        if bridge:
            bridge.terminate()
        ev.close()
        if fc:
            fc.stop()
            s = fc.summary()
            if s:
                R.log(f"FC MAVLink rx counts: {s['rx']}")
                R.log(f"FC MAVLink tx counts: {s['tx']}")
        ok = R.finish()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
