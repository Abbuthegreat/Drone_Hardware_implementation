# Validation report

> **Latest (2026-10-04, Windows):** [docs/WINDOWS_LINUX_STATUS.md](docs/WINDOWS_LINUX_STATUS.md). The Ubuntu ORB-SLAM3 runs are in logs/ubuntu/ and logs/validation/20261002_*.


> **Stage 2 (autonomous demo: SLAM + perception + thermal + geo-semantic map + A\* + VFH+ + video) is reported in [docs/DEMO_REPORT.md](docs/DEMO_REPORT.md).**
> Latest complete run: `logs/validation/20260930_193123/` — 13/13 checks passed (105 pytest tests incl. the stage-1 suites, calibration, the attached kit's own `postprocess.py` on a simulated flight,
> ROS 2 sensor-topic tests, the full ROS 2 → MAVLink → FC → PyBullet autonomous mission test, and the 16-step communication test).
> The stage-1 report below describes the earlier run (73 tests) and is kept for reference; its numbers are superseded by that run.

Everything below was **actually executed** on 2026-09-30 on Windows 11 with the `uavros` environment
(Python 3.11.16, pybullet (conda-forge build of Oct 21 2025), pymavlink 2.4.50, numpy 1.26, ROS 2 Humble via RoboStack).
Raw evidence: `logs/validation/20260930_170749/` (`summary.json` + one output file per check), `docs/COMM_TEST_REPORT.md`,
`docs/comm_test.log`, `docs/ros2_cli_output.txt`, `logs/tuning/`, `logs/calibration/`.

Re-run everything: `python scripts/run_all_tests.py --ros-python <uavros python>` (≈ 8 min).

## Summary of the integrated run (`scripts/run_all_tests.py`)

| Check | Result | Time |
|---|---|---|
| Python syntax (`compileall`, also verified with the system Python 3.14) | PASS | 0.1 s |
| URDF + `drone.yaml` derived block up to date (`tools.build_urdf --check`) | PASS | 0.3 s |
| Environment diagnostic (`scripts/diagnose.py`) | PASS | 0.9 s |
| **pytest: 73 passed, 1 deselected** (the deselected one is the ROS end-to-end test, run as its own check below) | PASS | 200 s |
| Calibration 1+2 mass/inertia | PASS | 0.6 s |
| Calibration 3–7 propulsion fitters (synthetic self-test) | PASS | 0.3 s |
| Calibration 3–7 propulsion report on the simulator | PASS | 0.5 s |
| Calibration 9+10 camera intrinsics / extrinsics | PASS | 4.2 s |
| Calibration 8 controller tuning (7 scenarios) | PASS | 94 s |
| **ROS 2 end-to-end communication test (16 steps)** | **PASS 16/16** | 54 s |

## Required items

| Item | Result | Evidence |
|---|---|---|
| URDF XML validation | PASS | `tests/test_model_urdf.py::test_urdf_is_valid_xml_with_required_links`: well-formed; 17 links (base_link, flight_controller_link, raspberry_pi_link, gps_link, rgbd_camera_link, rgbd_camera_optical_link, battery_link, esc_link, motor_1..4_link, rotor_1..4_link, airframe_link); all joints fixed with explicit `xyz`/`rpy` |
| Total mass | **2.200000 kg** | summed from the URDF XML link masses (not from the yaml); PyBullet effective mass by pushing with 10 N: 2.2000 kg |
| Inertia validation | PASS | positive definite; analytic composite = drone.yaml (2e-6); **independent point-cloud integration** agrees within 3 % (diag) / 1e-4 (Ixz); PyBullet dynamic torque test within 3 % of the yaml diagonal. Ixx 0.029517, Iyy 0.035333, Izz 0.053935, Ixy 0, Ixz 0.000453, Iyz 0 kg·m²; CG = (0, 0, −0.0209) m |
| PyBullet startup | PASS | every test/sim; headless `DIRECT` (GUI mode **not** exercised — headless machine session) |
| MAVLink server startup | PASS | `udpin:0.0.0.0:14540`, `fc_sim_stdout.log` |
| Heartbeat connection | PASS | FC→companion 2.0 Hz measured; companion→FC heartbeats counted at the FC (45 in the comm test) |
| Telemetry reception | PASS | LOCAL_POSITION_NED 30.0 Hz, ATTITUDE 50.0, GLOBAL_POSITION_INT 10.0, SYS_STATUS 2.0, HIGHRES_IMU 100.0, GPS_RAW_INT 10.0 (`diagnose.py --live`) |
| Command acknowledgement | PASS | ARM → `COMMAND_ACK command=400 result=0`; unsupported command → result 3; offboard without setpoint stream → result 2 (DENIED) |
| Arm / disarm path | PASS | arm ACK accepted; disarm on ground accepted; in-flight disarm refused unless forced; forced disarm → motor thrust 0 within 0.6 s; LAND → auto-disarm |
| ROS 2 topic availability | PASS | `ros2 topic list --no-daemon` shows `/uav/cmd_vel /uav/cmd_pos /uav/telemetry /uav/odom /uav/imu /uav/gps/fix /uav/mode /uav/armed` |
| ROS → MAVLink command test | PASS | FC log (`fc_mavlink.log`): ROS `linear.y=1.5` (ENU North) → `RX SET_POSITION_TARGET_LOCAL_NED frame=1 mask=0x05c7 vel=(1.50,0.00,-0.00) accepted=True`; `ros2 topic pub ... linear.x=1.5` (ENU East) → `vel=(0.00,1.50,-0.00) accepted=True` (mask 0x05c7 = velocity + yaw-rate) |
| MAVLink → FC → PyBullet motion | PASS | climb +4.74 m at 1.02 m/s with 21.9 N total thrust; North 1.5 m/s cmd: 5.49 m, initial accel 1.28 m/s², final 1.52 m/s; East via CLI: 4.57 m |
| Telemetry return test | PASS | ROS `/uav/odom` vs PyBullet truth: 0.03 m; wire test: local position < 0.25 m, velocity < 0.25 m/s, yaw < 0.05 rad vs PyBullet; lat/lon-derived NED within 0.1 m of local NED; PyBullet +X/+Y = MAVLink E/N verified; timestamps strictly increasing |
| Setpoint-loss failsafe | PASS | FAILSAFE_HOLD 0.64–0.77 s after the stream stops (FC timeout 0.5 s + telemetry latency); speed 1.5 → 0.01 m/s; stays airborne; does not resume without an explicit OFFBOARD request |
| RGB-D capture | PASS | 640×480 RGB uint8 + float32 depth in metres; depth at 4 pixels equals analytic ray/ground intersection within 2 %; camera pose == body pose ∘ URDF extrinsics; same pixel sees ≥ 1.4× farther ground after climbing; yaw rotates the camera; survivor shirt colour visible |

## Physics sanity tests A–N (`tests/`)
A mass ✔ · B free fall when unarmed/unpowered (a = g in the first 0.1 s; drag slows it later, motors 0 N) ✔ · C hover: mean total thrust 21.58 N (5.3955 N/motor), altitude drift < 0.15 m, tilt < 1° ✔ ·
D climb ✔ · E descend ✔ · F forward → North, nose-down pitch ✔ · G lateral → East, right-wing-down roll ✔ · H positive yaw rate → heading increases (clockwise), FLU z-rate negative ✔ ·
I mixer signs (mathematical, `r×F`, and inside PyBullet) ✔ · J motor lag (first-order: 63 % after one τ; thrust does not jump) ✔ · K setpoint loss ✔ · L MAVLink bidirectional (11 wire tests over UDP) ✔ ·
M ROS 2 (16-step end-to-end; frame-conversion unit tests) ✔ · N RGB-D ✔.

## Controller tuning (`logs/tuning/*.csv`, `summary.json`)
| Test | Overshoot | Rise | Settling (2 %) | Steady-state error |
|---|---|---|---|---|
| Altitude step 2 m | 0.02 % | 1.30 s | 2.15 s | 0.7 mm |
| Roll / pitch 10° (attitude target) | 7.5 % | 0.20 s | 0.45 s | < 0.04° |
| Yaw 45° | 0.55 % | 0.45 s | 0.75 s | 0.013° |
| Velocity 0→2 m/s | 2.0 % | 0.40 s | 1.45 s | 0.014 m/s; cross-track 0.004 m/s, altitude dev. 3.9 cm |
| Position step 5 m | 0 % | 1.65 s | 6.1 s | 2 cm; cross-track 2 mm |
| Hover 40 s (IMU + estimator noise on) | — | — | — | max drift 4.4 mm, RMS tilt 0.074°, thrust 21.582 ± 0.045 N |
| 4 m/s constant wind, hold | — | — | — | peak 0.19 m, final 7 mm, lean 3.2° |

## Calibration tool results
* Mass/inertia: see above. · Propulsion fitters recover kT (0.005 %), kQ (1.3 %), τ (0.12 %) and the V² exponent from synthetic data.
* Simulator's implied numbers (placeholders!): hover command 0.75 / 0.63 / 0.55 at 19.2 / 22.2 / 25.2 V (6035 rpm); max thrust 48.0 N (T/W 2.22) at 22.2 V; recovered kQ/kT from a PyBullet yaw test 0.01495 vs configured 0.0150 (clockwise as required); hover electrical power ≈ 304 W, time to low-voltage failsafe at hover ≈ 29.8 min.
* Camera intrinsics via rendered checkerboard + OpenCV: fx, fy within 0.29 % / 0.39 %, principal point within 1.3 px, reprojection RMS 0.34 px. Extrinsics via marker back-projection through the URDF: mean error 8.3 mm, max 10.7 mm.

## Not run / cannot be validated here
* **PyBullet GUI** (`--gui`): not opened in this headless session. Run `python run_sim.py`.
* **Linux / Raspberry Pi 5 / real Pixhawk 6C Mini / real serial link**: not available. The serial path is a different `mavutil` connection string only; it is untested.
* **Physical parameters**: nothing was compared with a real S3110/propeller/battery. All propulsion, battery, drag and sensor-noise values are placeholders, and the fitters were validated on synthetic data only.
* **`ros2` daemon**: `ros2 topic list/echo/pub` work with `--no-daemon`; the default daemon hung on this Windows/RoboStack install.
* `tests/test_ros2.py::test_ros2_end_to_end_comm_test` is skipped (not passed) in a Python without `rclpy`; install ROS 2 or use the `uavros` environment.
* Real-time performance of camera streaming (software renderer ≈ 0.2–0.35 s/frame at 640×480).
