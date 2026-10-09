"""
ORB-SLAM3 backend (the attached kit's SLAM engine) behind the same interface as slam.vo_backend.OrbPnpVO.

STATUS: built and validated on Ubuntu 22.04 (see docs/SLAM_INTEGRATION_UBUNTU.md, PATCHES.md):

    bash slam/build_orbslam3.sh            # the kit's setup script + our streaming runner  ->  ~/slam_deps/ORB_SLAM3/Examples/RGB-D/rgbd_stream
    # config/drone.yaml:  slam.backend: orbslam3   (default)

What is reused from the kit: ORB-SLAM3 itself, the ORB-SLAM3 settings file written from the camera intrinsics (`record.write_orbslam_yaml`),
RGB-D input conventions (BGR image, depth in millimetres -> RGBD.DepthMapFactor 1000, tracking state 2 = OK) and the runner's frame loop
(slam/rgbd_stream.cc is rgbd_offline.cc turned into a streaming process). The pose comes back as Twc in ORB-SLAM3's world frame
(first camera frame), exactly what slam.adapter.SlamFrontEnd expects from any backend.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile

import cv2
import numpy as np

from slam import kit
from slam.backend import SlamBackend, SlamResult


def _quat_to_R(qx, qy, qz, qw):
    from common.frames import R_from_quat
    return R_from_quat([qw, qx, qy, qz])


def orbslam3_dir():
    return os.environ.get("ORB_SLAM3_DIR", os.path.expanduser("~/slam_deps/ORB_SLAM3"))


def orbslam3_available(orb_dir=None):
    """True when the streaming runner and the vocabulary built by slam/build_orbslam3.sh exist (Linux / Jetson only)."""
    d = orb_dir or orbslam3_dir()
    return os.path.exists(os.path.join(d, "Examples", "RGB-D", "rgbd_stream")) and os.path.exists(os.path.join(d, "Vocabulary", "ORBvoc.txt"))


def resolve_backend(configured: str, requested: str = "") -> tuple:
    """
    Which SLAM backend to run.  requested: "" (= use the config), "orbslam3", "orb_pnp_vo" or "auto".
    "auto" keeps the configured backend when it can run and otherwise falls back to the pure-Python RGB-D VO (the case on Windows,
    where ORB-SLAM3 is not built). Returns (backend, note); the note is non-empty when a fallback happened, so callers can report it.
    """
    want = requested or configured
    if want == "auto":
        if configured == "orbslam3" and not orbslam3_available():
            return "orb_pnp_vo", (f"ORB-SLAM3 runner not found under {orbslam3_dir()} (build it with slam/build_orbslam3.sh on Linux); "
                                  "using the orb_pnp_vo fallback (RGB-D ORB + PnP odometry, no loop closure)")
        return (configured if configured != "auto" else ("orbslam3" if orbslam3_available() else "orb_pnp_vo")), ""
    return want, ""


class OrbSlam3Backend(SlamBackend):
    name = "orbslam3"

    def __init__(self, K, cfg, orb_dir=None):
        self.orb_dir = orb_dir or orbslam3_dir()
        self.runner = os.path.join(self.orb_dir, "Examples", "RGB-D", "rgbd_stream")
        self.vocab = os.path.join(self.orb_dir, "Vocabulary", "ORBvoc.txt")
        if not os.path.exists(self.runner):
            raise FileNotFoundError(f"{self.runner} not found. Build it with slam/build_orbslam3.sh on a Linux/Jetson machine.")
        rec = kit.record_module()
        self.yaml = os.path.join(tempfile.mkdtemp(prefix="orbslam_"), "orbslam.yaml")
        intr = dict(width=K["width"], height=K["height"], fx=K["fx"], fy=K["fy"], ppx=K["cx"], ppy=K["cy"], model="none", coeffs=[0] * 5)
        rec.write_orbslam_yaml(self.yaml, intr, depth_scale=0.001, fps=cfg["sensors"]["camera"]["rate_hz"], baseline=0.05,
                               n_features=cfg["slam"]["n_features"])
        self.p = None
        self.reset()

    def _spawn(self):
        self.p = subprocess.Popen([self.runner, self.vocab, self.yaml, "0"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        line = self.p.stdout.readline().decode().strip()          # "READY" after the vocabulary is loaded (several seconds)
        assert line == "READY", line

    def reset(self):
        if self.p is not None:
            self.close()
        self.restarts, self.n_runner_crashes = 0, 0
        self._spawn()
        self.n_ok = self.n_lost = 0

    def _respawn_after_crash(self):
        """ORB-SLAM3 can abort on a numerical failure inside Sophus/g2o (NaN pose) - observed in dynamic scenes. The runner is restarted; its new map
        gets a distinct id (map_id + 1000 * restarts) so the adapter anchors it at the last accepted pose instead of mixing two frames."""
        self.n_runner_crashes += 1
        try:
            self.p.kill()
        except Exception:
            pass
        self.restarts += 1
        self._spawn()

    def process(self, t, bgr, depth_m) -> SlamResult:
        h, w = bgr.shape[:2]
        d16 = np.nan_to_num(depth_m * 1000.0, nan=0.0).clip(0, 65535).astype(np.uint16)
        try:
            if self.p.poll() is not None:
                raise EOFError("runner not running")
            self.p.stdin.write(b"FRM1" + struct.pack("<dii", t, w, h) + np.ascontiguousarray(bgr).tobytes() + d16.tobytes())
            self.p.stdin.flush()
            parts = self.p.stdout.readline().decode().split()
            if len(parts) < 10:
                raise EOFError("runner closed its pipe")
        except (EOFError, BrokenPipeError, OSError):
            self._respawn_after_crash()
            self.n_lost += 1
            return SlamResult(t, "LOST", None, {"orb_state": -1, "runner_crashed": True, "n_runner_crashes": self.n_runner_crashes})
        _, tt, st, tx, ty, tz, qx, qy, qz, qw = parts[:10]
        map_id = (int(parts[10]) if len(parts) > 10 else 0) + 1000 * self.restarts
        if int(st) != 2:                                             # ORB-SLAM3: 2 = OK
            self.n_lost += 1
            return SlamResult(t, "LOST", None, {"orb_state": int(st)})
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(float(qx), float(qy), float(qz), float(qw))
        T[:3, 3] = [float(tx), float(ty), float(tz)]
        self.n_ok += 1
        return SlamResult(t, "OK", T, {"orb_state": 2, "map_id": map_id})

    def close(self):
        try:
            self.p.stdin.write(b"FRM1" + struct.pack("<dii", -1.0, 0, 0))
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
