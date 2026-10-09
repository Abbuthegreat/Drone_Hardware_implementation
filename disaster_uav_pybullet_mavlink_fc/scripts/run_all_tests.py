"""
Run every check and write logs/validation/<timestamp>/summary.json (+ per-command output files).

    python scripts/run_all_tests.py                       # everything that this python env supports
    python scripts/run_all_tests.py --ros-python PATH     # also run the ROS 2 end-to-end communication test with that python
    python scripts/run_all_tests.py --fast                # skip the slow tuning + wire suites

Nothing is faked: a check that cannot run is reported as SKIPPED with the missing dependency.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(name, cmd, out_dir, results, timeout=1800, env=None):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
        rc, out = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        rc, out = -1, "TIMEOUT"
    with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(out)
    status = "PASS" if rc == 0 else "FAIL"
    tail = [l for l in out.strip().splitlines() if l.strip() and "merge " not in l and "relink" not in l and "processing" not in l][-3:]
    results.append({"check": name, "status": status, "seconds": round(time.time() - t0, 1), "tail": tail})
    print(f"{status:5s} {name:38s} {time.time()-t0:6.1f}s  {tail[-1] if tail else ''}", flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ros-python", default=None)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "logs", "validation", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    py, res = sys.executable, []
    run("python_syntax_compileall", [py, "-m", "compileall", "-q", "common", "companion", "sim", "worlds", "algo", "eval", "ros2_bridge",
                                     "calibration", "tools", "scripts", "tests", "slam", "perception", "mapping", "planning", "mission", "video"], out_dir, res)
    run("urdf_and_yaml_up_to_date", [py, "-m", "tools.build_urdf", "--check"], out_dir, res)
    run("diagnose_environment", [py, "scripts/diagnose.py"], out_dir, res)
    pytest_args = [py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-m", "not ros"]   # ROS tests run as their own checks below
    if a.fast:
        pytest_args = [py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-m", "not ros and not wire and not slow", "--deselect", "tests/test_tuning.py"]
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")      # ROS ships pytest plugins that break newer pytest
    run("pytest_all", pytest_args, out_dir, res, timeout=3600, env=env)
    for name, mod in (("calibration_1_2_mass_inertia", "calibration.verify_mass_inertia"),
                      ("calibration_3-7_propulsion_fit_selftest", "calibration.fit_propulsion"),
                      ("calibration_3-7_propulsion_report", "calibration.propulsion_report"),
                      ("calibration_9_10_camera", "calibration.camera_calibration")):
        cmd = [py, "-m", mod] + (["selftest"] if mod.endswith("fit_propulsion") else [])
        run(name, cmd, out_dir, res)
    if not a.fast:
        run("calibration_8_controller_tuning", [py, "-m", "calibration.tune_controller"], out_dir, res, timeout=1800)
    if not a.fast:
        run("kit_postprocess_on_simulated_flight", [py, "scripts/kit_postprocess_check.py", "--out", os.path.join(out_dir, "kit_check")], out_dir, res, timeout=1200)
    if a.ros_python:
        renv = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        run("ros2_sensor_topics_test", [a.ros_python, "-m", "pytest", "tests/test_ros_sensors.py", "tests/test_ros2.py", "-q", "-p", "no:cacheprovider", "-m", "ros"],
            out_dir, res, timeout=900, env=renv)
        if not a.fast:
            run("ros2_mavlink_autonomous_mission_test", [a.ros_python, "-m", "pytest", "tests/test_ros_demo.py", "-q", "-p", "no:cacheprovider", "-m", "ros"],
                out_dir, res, timeout=2400, env=renv)
    if a.ros_python:
        run("ros2_end_to_end_comm_test", [a.ros_python, "scripts/comm_test.py", "--out", os.path.join(out_dir, "comm_test")], out_dir, res, timeout=900)
    else:
        has = importlib.util.find_spec("rclpy") is not None
        if has:
            run("ros2_end_to_end_comm_test", [py, "scripts/comm_test.py", "--out", os.path.join(out_dir, "comm_test")], out_dir, res, timeout=900)
        else:
            res.append({"check": "ros2_end_to_end_comm_test", "status": "SKIPPED", "tail": ["rclpy not importable in this python; pass --ros-python PATH"]})
            print("SKIP  ros2_end_to_end_comm_test: rclpy not importable in this python; pass --ros-python", flush=True)
    json.dump(res, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    bad = [r for r in res if r["status"] == "FAIL"]
    print(f"\n{len(res) - len(bad)}/{len(res)} checks passed or skipped; summary: {out_dir}\\summary.json")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
