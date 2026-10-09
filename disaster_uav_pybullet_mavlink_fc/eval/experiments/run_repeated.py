"""
Repeated, isolated full-stack missions (ROS 2 + MAVLink + FC + PyBullet + live ORB-SLAM3) for variance studies.

    python3 eval/experiments/run_repeated.py --tag static_base --n 15
    python3 eval/experiments/run_repeated.py --tag static_memoff --n 10 --config logs/experiments/cfg_memory_off.yaml
    python3 eval/experiments/run_repeated.py --tag dynamic_base --n 8 --scenario dynamic

Every mission is its own `scripts/run_disaster_demo.py` process tree (never several heavy simulations in one process), run strictly one after
the other so all runs see the same machine load. Re-running the same tag resumes: finished runs (score_report.json present) are skipped.
CPU load is sampled every 5 s into <run>/load.csv. Nothing here feeds the autonomy stack; ground truth is only read later by analyze_runs.py.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cpu():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]


def sample_load(path, stop):
    with open(path, "w") as f:
        f.write("wall_s,load1,cpu_busy_pct\n")
        t0, (a0, i0) = time.time(), _cpu()
        while not stop.is_set():
            time.sleep(5.0)
            a1, i1 = _cpu()
            busy = 100.0 * (1 - (i1 - i0) / max(a1 - a0, 1))
            a0, i0 = a1, i1
            f.write(f"{time.time() - t0:.0f},{open('/proc/loadavg').read().split()[0]},{busy:.1f}\n")
            f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scenario", default="static", choices=["static", "dynamic"])
    ap.add_argument("--config", default=None, help="alternate drone.yaml (exported as UAV_CONFIG to every process)")
    ap.add_argument("--video", action="store_true", help="also record the video compositor (default: --no-video, as tests/test_ros_demo.py)")
    a = ap.parse_args()
    base = os.path.join(ROOT, "logs", "experiments", a.tag)
    os.makedirs(base, exist_ok=True)
    env = dict(os.environ)
    if a.config:
        env["UAV_CONFIG"] = os.path.abspath(a.config)
    manifest = os.path.join(base, "manifest.jsonl")
    for i in range(1, a.n + 1):
        out = os.path.join(base, f"run_{i:02d}")
        if os.path.exists(os.path.join(out, "score_report.json")):
            continue
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_disaster_demo.py"), "--slam-backend", "orbslam3", "--out", out]
        if a.scenario == "dynamic":
            cmd += ["--scenario", "dynamic"]
        if not a.video:
            cmd.append("--no-video")
        os.makedirs(out, exist_ok=True)
        stop = threading.Event()
        th = threading.Thread(target=sample_load, args=(os.path.join(out, "load.csv"), stop), daemon=True)
        th.start()
        t0 = time.time()
        with open(os.path.join(out, "launcher.log"), "w") as lf:
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        stop.set()
        th.join(timeout=10)
        rec = {"run": f"run_{i:02d}", "rc": rc, "wall_s": round(time.time() - t0, 1), "cmd": " ".join(cmd), "config": env.get("UAV_CONFIG", "config/drone.yaml"),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
