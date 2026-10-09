"""
First SLAM test (no autonomy): take off -> slow translation -> rotation -> translation -> stop, with the REAL ORB-SLAM3 (kit) fed only
RGB-D + IMU + the FC heading. Ground truth is recorded separately and used ONLY for the offline comparison.

    python scripts/test_slam_pipeline.py [--backend orbslam3|orb_pnp_vo] [--out logs/slam_pipeline]

Outputs <out>/report.json (tracking continuity, loss count, path length, ATE, drift, frame gaps, timestamp monotonicity, per-frame latency),
trajectory.csv (slam vs ground truth, evaluation only), trajectory.png (top-down + height plot), kit_run/ (kit-format dataset).
Exit code 0 only if ORB-SLAM3 tracked >= 95 % of the frames, did not just reproduce ground truth (ATE > 0) and ATE RMSE < 0.5 m.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from eval.rigs.slam_offline_eval import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="orbslam3", choices=["orbslam3", "orb_pnp_vo"])
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "slam_pipeline"))
    a = ap.parse_args()
    rep = run(out=a.out, backend_name=a.backend, verbose=False)
    L = np.genfromtxt(os.path.join(a.out, "trajectory.csv"), delimiter=",", names=True)
    # top-down plot drawn with OpenCV (the system matplotlib is built against numpy 1.x)
    Wd, Hd, m = 900, 700, 60
    pts = np.vstack([L["gt_x"], L["gt_y"], L["slam_x"], L["slam_y"]]).reshape(2, 2, -1)
    allx = np.concatenate([pts[0, 0], pts[1, 0]]); ally = np.concatenate([pts[0, 1], pts[1, 1]])
    ok = np.isfinite(allx) & np.isfinite(ally)
    lo = np.array([allx[ok].min(), ally[ok].min()]); sc = min((Wd - 2 * m) / max(np.ptp(allx[ok]), 1e-6), (Hd - 2 * m) / max(np.ptp(ally[ok]), 1e-6))
    px = lambda x, y: (int(m + (x - lo[0]) * sc), int(Hd - m - (y - lo[1]) * sc))
    img = np.full((Hd, Wd, 3), 255, np.uint8)
    for i in range(len(L) - 1):
        cv2.line(img, px(L["gt_x"][i], L["gt_y"][i]), px(L["gt_x"][i + 1], L["gt_y"][i + 1]), (0, 160, 0), 3)
        if np.isfinite(L["slam_x"][i]) and np.isfinite(L["slam_x"][i + 1]):
            cv2.line(img, px(L["slam_x"][i], L["slam_y"][i]), px(L["slam_x"][i + 1], L["slam_y"][i + 1]), (0, 0, 220), 2)
    cv2.putText(img, "green: ground truth (offline evaluation only)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 130, 0), 2)
    cv2.putText(img, f"red: {rep['backend']} estimate  ATE RMSE {rep['ate_rmse_m']:.3f} m  max {rep['ate_max_m']:.3f} m  tracked {100 * rep['tracked_fraction']:.0f}%",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    cv2.putText(img, f"top-down, East right / North up, 1 m = {sc:.0f} px", (20, Hd - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.imwrite(os.path.join(a.out, "trajectory.png"), img)
    print(json.dumps(rep, indent=1))
    ok = rep["tracked_fraction"] >= 0.95 and rep["ate_rmse_m"] is not None and 0 < rep["ate_rmse_m"] < 0.5 and rep["timestamps_monotonic"]
    print("SLAM PIPELINE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
