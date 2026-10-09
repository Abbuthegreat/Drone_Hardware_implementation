"""The ORB-SLAM3 runner can abort (observed: NaN inside Sophus in a dynamic scene). The backend must survive: LOST for that frame, respawn,
and a new map id so the adapter re-anchors at the last accepted pose."""
import os
import stat
import sys

import numpy as np

from slam.orbslam3_backend import OrbSlam3Backend

FAKE = f"""#!{sys.executable}
import struct, sys
out = sys.stdout
out.write("READY\\n"); out.flush()
n = 0
while True:
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4: break
    t, w, h = struct.unpack("<dii", sys.stdin.buffer.read(16))
    if t < 0: break
    sys.stdin.buffer.read(w * h * 3 + w * h * 2)
    n += 1
    if n == 3:
        import os; os._exit(134)                       # abort, like a failed Sophus ensure
    out.write("POSE %.6f 2 %.3f 0 0 0 0 0 1 0\\n" % (t, 0.1 * n)); out.flush()
"""


def test_backend_survives_runner_abort_and_reports_new_map_id(cfg, tmp_path):
    d = tmp_path / "Examples" / "RGB-D"
    d.mkdir(parents=True)
    r = d / "rgbd_stream"
    r.write_text(FAKE)
    r.chmod(r.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "Vocabulary").mkdir()
    K = {"fx": 400.0, "fy": 400.0, "cx": 2.0, "cy": 2.0, "width": 4, "height": 4}
    be = OrbSlam3Backend(K, cfg, orb_dir=str(tmp_path))
    bgr, depth = np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32)
    a = be.process(0.2, bgr, depth)
    b = be.process(0.4, bgr, depth)
    assert a.state == b.state == "OK" and a.info["map_id"] == b.info["map_id"] == 0
    c = be.process(0.6, bgr, depth)                          # the runner aborts on this frame
    assert c.state == "LOST" and c.T_w_c is None and c.info.get("runner_crashed") and be.n_runner_crashes == 1
    d_ = be.process(0.8, bgr, depth)                         # respawned runner answers again, in a different map id
    assert d_.state == "OK" and d_.info["map_id"] == 1000
    be.close()
