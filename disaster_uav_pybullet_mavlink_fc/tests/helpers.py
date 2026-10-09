"""Shared helpers for FC-level physics tests (no MAVLink; the wire is tested in test_mavlink_wire.py)."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.flight_controller_sim import IGN_POS, IGN_VEL, IGN_ACC, IGN_YAW, IGN_YAW_RATE  # noqa: E402
from sim.sim_core import UavSim  # noqa: E402

PY_ENV_HINT = ("Run with the environment that has pybullet + pymavlink, e.g. "
               "%USERPROFILE%\\mm\\root\\envs\\uav\\python.exe -m pytest tests")


class Flyer:
    """Drives the simulated FC's command interface directly at 20 Hz (like a companion would, minus the wire)."""

    def __init__(self, sim: UavSim, rate_hz=20.0):
        self.sim, self.fc = sim, sim.fc
        self.period = 1.0 / rate_hz

    def vel(self, vn, ve, vd, yaw_rate=0.0):
        self.fc.handle_position_target(self.sim.t, 1, IGN_POS | IGN_ACC | IGN_YAW, np.zeros(3), (vn, ve, vd), np.zeros(3), 0.0, yaw_rate)

    def pos(self, n, e, d, yaw=None):
        mask = IGN_VEL | IGN_ACC | IGN_YAW_RATE | (IGN_YAW if yaw is None else 0)
        self.fc.handle_position_target(self.sim.t, 1, mask, (n, e, d), np.zeros(3), np.zeros(3), 0.0 if yaw is None else yaw, 0.0)

    def stream(self, seconds, send):
        """Call send() every period for `seconds` of simulated time."""
        for _ in range(int(round(seconds / self.period))):
            send()
            self.sim.run(self.period)

    def takeoff(self, alt_m=3.0, settle_s=8.0):
        self.sim.run(0.3)
        self.stream(0.5, lambda: self.vel(0, 0, 0))
        assert self.fc.arm(self.sim.t) == 0
        assert self.fc.set_mode("OFFBOARD", self.sim.t) == 0
        n0, e0 = self.sim.state.pos_ned[:2]
        d0 = self.sim.state.pos_ned[2]
        self.stream(settle_s, lambda: self.pos(n0, e0, d0 - alt_m, 0.0))
        return n0, e0
