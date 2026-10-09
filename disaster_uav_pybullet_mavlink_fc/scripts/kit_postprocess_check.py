"""
Run the ATTACHED KIT'S OWN, UNMODIFIED postprocess.py (YOLO detection + torso-depth localisation + clustering + map/report) on a
simulated flight recorded in the kit's run-folder format, and compare with what our online pipeline found.

    python scripts/kit_postprocess_check.py [--out logs/kit_check]

Chain: PyBullet flight (scripted through the FC) -> RGB-D + IMU -> our SLAM front-end -> CameraTrajectory.txt (EuRoC format, like ORB-SLAM3)
       -> slam_kit_3/slam_kit/postprocess.py (unmodified) -> results/humans.json, report.html, map_topdown.png.
Pass criterion: the kit finds exactly the survivor our online tracker found, and the camera-relative distance agrees within 0.8 m.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "kit_check"))
    a = ap.parse_args()
    from eval.rigs.survivor_offline_test import run
    run_dir = os.path.join(a.out, "kit_run")
    rep = run(out=a.out, verbose=False, save_kit_run_dir=run_dir)
    print(f"[kit-check] online pipeline survivors: {[(s['track_id'], s['status'], s['local_xyz']) for s in rep['survivor_nodes']]}")
    cmd = [sys.executable, os.path.join(ROOT, "slam_kit_3", "slam_kit", "postprocess.py"), run_dir, "--backend", "onnxruntime",
           "--conf", "0.35", "--stride", "1", "--min-obs", "3", "--no-video"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(os.path.join(a.out, "kit_postprocess.log"), "w").write(r.stdout + r.stderr)
    print("\n".join(l for l in r.stdout.splitlines() if "[post]" in l or "H1" in l)[-1500:])
    humans = json.load(open(os.path.join(run_dir, "results", "humans.json")))
    cam0 = np.array(rep["camera0_in_map"])
    ours = [s for s in rep["survivor_nodes"] if s["status"] == "CONFIRMED"]
    ok = len(humans) >= 1 and len(ours) >= 1
    if ok:
        d_kit = math.sqrt(humans[0]["x"] ** 2 + humans[0]["y"] ** 2 + humans[0]["z"] ** 2)
        d_ours = float(np.linalg.norm(np.array(ours[0]["local_xyz"]) - cam0))
        print(f"[kit-check] kit H1 distance from the first camera pose: {d_kit:.2f} m; our node: {d_ours:.2f} m")
        ok = abs(d_kit - d_ours) < 0.8 and len(humans) == len(ours)
    json.dump({"pass": bool(ok), "kit_humans": humans, "online_nodes": rep["survivor_nodes"]}, open(os.path.join(a.out, "kit_check.json"), "w"), indent=1)
    print("[kit-check]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
