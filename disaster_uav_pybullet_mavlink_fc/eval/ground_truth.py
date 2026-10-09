"""
EVALUATION-ONLY ground-truth channel.

    +--------------------------------------------------------------+
    |  GROUND-TRUTH FIREWALL                                       |
    |  Nothing in `companion/`, `algo/` or `ros2_bridge/` may      |
    |  import this module, `sim.*`, `worlds.*` or `pybullet`.      |
    |  (Enforced by tests/test_ground_truth_firewall.py)           |
    +--------------------------------------------------------------+

Autonomy inputs are RGB, depth, camera info, IMU, GPS and MAVLink telemetry only. This module records
what the physics engine *really* did (position, orientation, velocity, motor thrust, contacts, survivor
positions, ...) into `logs/<run>/ground_truth.csv` and optionally streams it as JSON over UDP
(default port sim.eval_udp_port) so that tests and evaluation scripts can compare against it. It is
attached to the simulation process only and is off the autonomy data path.
"""
from __future__ import annotations

import json
import os
import socket

import numpy as np

COLUMNS = ["t", "pn", "pe", "pd", "vn", "ve", "vd", "roll", "pitch", "yaw", "wx", "wy", "wz",
           "thr1", "thr2", "thr3", "thr4", "rpm1", "rpm2", "rpm3", "rpm4", "u1", "u2", "u3", "u4",
           "batt_v", "batt_a", "soc", "armed", "mode", "landed", "contacts", "sp_rx", "sp_rejected"]


class GroundTruthRecorder:
    def __init__(self, sim, log_dir: str, hz: float = 50.0, udp_port: int | None = None, world_info=None):
        self.sim = sim
        self.div = max(1, int(round(sim.hz / hz)))
        os.makedirs(log_dir, exist_ok=True)
        self.csv = open(os.path.join(log_dir, "ground_truth.csv"), "w", buffering=1)
        self.csv.write(",".join(COLUMNS) + "\n")
        if world_info is not None:
            with open(os.path.join(log_dir, "world_truth.json"), "w") as f:
                json.dump({"humans": world_info["humans"], "hazards": world_info.get("hazards", []), "scenario": world_info.get("scenario", "static"),
                           "dynamic_obstacles": world_info.get("dynamic_obstacles", [])}, f, indent=1)
        self.log_dir = log_dir
        self.world_info = world_info
        self.sock = None
        self.addr = ("127.0.0.1", udp_port) if udp_port else None
        if self.addr:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sim.hooks.append(self._on_step)

    def snapshot(self) -> dict:
        s, fc = self.sim, self.sim.fc
        st = s.state
        r, p, y = st.euler_ned
        return {
            "t": s.t, "pos_ned": st.pos_ned.tolist(), "vel_ned": st.vel_ned.tolist(), "rpy": [r, p, y],
            "omega_frd": st.omega_frd.tolist(), "thrust_n": s.motor_thrusts.tolist(), "rpm": s.motor.rpm().tolist(),
            "u": s.u.tolist(), "batt_v": s.battery.voltage_v, "batt_a": s.battery.current_a, "soc": s.battery.soc,
            "armed": fc.armed, "mode": fc.mode, "landed": fc.landed, "failsafe": fc.failsafe_reason,
            "pos_enu": st.pos_enu.tolist(),
        }

    def _on_step(self, sim):
        if sim.step_count % self.div:
            return
        st, fc = sim.state, sim.fc
        r, p, y = st.euler_ned
        row = [sim.t, *st.pos_ned, *st.vel_ned, r, p, y, *st.omega_frd, *sim.motor_thrusts, *sim.motor.rpm(), *sim.u,
               sim.battery.voltage_v, sim.battery.current_a, sim.battery.soc, int(fc.armed), fc.mode, int(fc.landed),
               len(sim.vehicle.contacts_with_world()), fc.n_sp_rx, fc.n_sp_rejected]
        self.csv.write(",".join(f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + "\n")
        if self.sock:
            try:
                self.sock.sendto(json.dumps(self.snapshot()).encode(), self.addr)
            except OSError:
                pass

    def close(self):
        self.csv.close()
        dyn = (self.world_info or {}).get("dynamic")
        if dyn is not None:
            with open(os.path.join(self.log_dir, "world_events.json"), "w") as f:
                json.dump(dyn.events, f, indent=1)


def load_ground_truth_csv(path):
    """Load a ground_truth.csv into a dict of numpy arrays (mode as str array)."""
    with open(path) as f:
        head = f.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    out = {}
    for i, h in enumerate(head):
        col = [r[i] for r in rows]
        out[h] = np.array(col) if h == "mode" else np.array(col, dtype=float)
    return out


class EvalListener:
    """EVALUATION-SIDE receiver of the JSON ground-truth stream (used by tests / comm_test, never by autonomy)."""

    def __init__(self, port: int, keep: int = 20000):
        import threading
        from collections import deque
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)
        self.buf = deque(maxlen=keep)
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        while not self._stop:
            try:
                data, _ = self.sock.recvfrom(65535)
            except (socket.timeout, OSError):
                continue
            try:
                self.buf.append(json.loads(data.decode()))
            except ValueError:
                pass

    def latest(self):
        return self.buf[-1] if self.buf else None

    def nearest(self, t_sim: float):
        best = None
        for s in reversed(self.buf):
            if best is None or abs(s["t"] - t_sim) < abs(best["t"] - t_sim):
                best = s
            if s["t"] < t_sim - 0.5:
                break
        return best

    def close(self):
        self._stop = True
        self._th.join(timeout=1)
        self.sock.close()
