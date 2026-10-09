"""Launch / stop the FC simulator as a separate process (real UDP, real wall clock)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FcProcess:
    def __init__(self, port=14641, eval_port=14661, log_dir=None, extra_args=(), world=False, python=None,
                 ros_sensors=False, spawn_height=None):
        self.port, self.eval_port = port, eval_port
        self.log_dir = log_dir or tempfile.mkdtemp(prefix="fcsim_")
        os.makedirs(self.log_dir, exist_ok=True)
        self.ready = os.path.join(self.log_dir, "ready.json")
        if os.path.exists(self.ready):
            os.remove(self.ready)
        cmd = [python or sys.executable, "-m", "sim.run_fc_sim", "--headless", "--port", str(port),
               "--eval-port", str(eval_port), "--log-dir", self.log_dir, "--ready-file", self.ready]
        if not world:
            cmd.append("--no-world")
        if spawn_height is not None:
            cmd += ["--spawn-height", str(spawn_height)]
        if ros_sensors:
            cmd.append("--ros-sensors")
        cmd += list(extra_args)
        self.out = open(os.path.join(self.log_dir, "fc_sim_stdout.log"), "w")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=self.out, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(self.ready):
                time.sleep(0.2)
                return True
            if self.proc.poll() is not None:
                raise RuntimeError(f"FC sim exited early ({self.proc.returncode}); see {self.out.name}")
            time.sleep(0.1)
        raise TimeoutError("FC sim did not become ready")

    def stop(self):
        if self.proc.poll() is None:
            open(os.path.join(self.log_dir, "STOP"), "w").close()      # graceful stop request
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.out.close()

    def summary(self):
        p = os.path.join(self.log_dir, "fc_mavlink_summary.json")
        return json.load(open(p)) if os.path.exists(p) else None
