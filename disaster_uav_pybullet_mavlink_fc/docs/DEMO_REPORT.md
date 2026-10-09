# Stage 2 report — autonomous disaster-response demonstration

> **Superseded in parts (2026-10-04):** this is the 2026-09-30 Windows-stage report. ORB-SLAM3 was later built and validated on Ubuntu (docs/SLAM_INTEGRATION_UBUNTU.md); the mission now also returns and lands, has a safety layer, a command-centre output and a dynamic scenario. Current status and results: [WINDOWS_LINUX_STATUS.md](WINDOWS_LINUX_STATUS.md).


Everything here was executed on Windows 11 in the `uavros` environment (Python 3.11, ROS 2 Humble via RoboStack, pybullet, pymavlink 2.4.50, onnxruntime 1.30 CPU, OpenCV).
Evidence: `logs/validation/20260930_193123/` (13 checks), `logs/demo/final/` (the demonstration run, video, logs, score report), `docs/demo_frames/` (stills from that video).

**Result in one paragraph.** One command, `python scripts/run_disaster_demo.py`, starts PyBullet + the simulated flight controller, the ROS 2 ↔ MAVLink bridge and four autonomy processes, flies an autonomous search mission through
the earthquake scene, and records `logs/demo/final/disaster_demo.mp4` (86.4 s, 1280×720, 10 fps, H.264, 24 MB). The UAV took off, SLAM-localised itself, mapped the scene live, went around a wall and a pole hidden behind it (A\* + VFH+), detected two survivors with the kit's YOLO on RGB, localised them
with depth, confirmed them with thermal evidence, geo-tagged them from a GPS anchor, found and mapped a fire hazard, and ended in a hover. **Two important caveats:** the SLAM is a visual-odometry backend written for this repository because **ORB-SLAM3 could not be built on this machine**, and the thermal camera is **synthetic**.

## A. Existing files inspected
* **Attached kit `slam_kit_3/slam_kit/`** (all read completely): `README.md`, `record.py`, `detector.py`, `postprocess.py`, `process.sh`, `setup_orbslam3.sh`, `start_rec.sh`, `stop_rec.sh`, `slam/rgbd_offline.cc`; `models/yolov8n.onnx` (loaded and run; `yolov8n.pt` not used).
  Findings: an **offline** RealSense D435i/D455 pipeline — record → ORB-SLAM3 **RGB-D** (C++, Pangolin, Linux/Jetson build) → YOLOv8n person detection → median torso depth → 3-D point → greedy clustering (radius 1.5 m, ≥ 3 sightings) → top-down map / PLY / HTML report.
  It has **no IMU use in SLAM** (IMU only gives gravity "up" at start), no online mode, no thermal, no A\*/VFH+, and no ROS.
* The whole validated stage-1 project (URDF, FC, MAVLink, ROS bridge, world, tests, calibration) was re-read where touched.

## B. SLAM-kit modules reused
| Kit item | Status |
|---|---|
| `detector.py` `Detector` and `models/yolov8n.onnx` | used **unmodified** for all person detection (onnxruntime backend) |
| `postprocess.py` `localize()` | used **unmodified** (torso-median depth → 3-D camera point) |
| `postprocess.py` `rot_a_to_b()` | used unmodified for gravity alignment of the SLAM world |
| `postprocess.py` clustering logic (radius, min-obs, merge < 0.8·radius) | re-implemented online as a tracker with the same parameters (`mapping/geo_semantic_map.py`) |
| `record.py` `write_orbslam_yaml`, run-folder format, `postprocess.py` (whole script) | `scripts/kit_postprocess_check.py` writes a simulated flight in the kit's format and runs the kit's **own unmodified `postprocess.py`** on it: it found exactly one human at **14.86 m** from the first camera pose; our online node had it at **14.89 m** (PASS) |
| `slam/rgbd_offline.cc`, `setup_orbslam3.sh` | adapted into `slam/rgbd_stream.cc` (streaming runner) and `slam/build_orbslam3.sh` (+ `slam/orbslam3_backend.py`). **Written, not built, not run** (needs Linux/Jetson) |
| ORB-SLAM3 itself | **not run.** The demo uses `slam/vo_backend.py`: RGB-D ORB + PnP-RANSAC visual odometry with gyro-based rotation gating, keyframes and relocalisation. It is odometry, not SLAM: no loop closure or bundle adjustment |

Why ORB-SLAM3 was not used: no Pangolin package for win-64 (ORB-SLAM3's viewer is a hard dependency), no WSL, MSVC x64 tools not installed, ~3 GB free disk. The backend interface (`slam/backend.py`) makes it a one-line switch (`slam.backend: orbslam3`) on a Linux/Jetson build.

## C. New files
`slam/{kit,backend,vo_backend,adapter,orbslam3_backend}.py`, `slam/rgbd_stream.cc`, `slam/build_orbslam3.sh` · `perception/{person_detector,thermal_fusion}.py` · `mapping/{metric_map,geo_semantic_map,geo_anchor,pipeline}.py` ·
`planning/{astar,vfh_plus}.py` · `mission/disaster_demo.py` · `video/{map_panel,compositor_node,textutil}.py` · `ros2_bridge/{slam_node,perception_mapping_node,nav_node,ros_util}.py` · `sim/{thermal_camera,ros_sensor_publisher}.py` (publisher rewritten) ·
`worlds/humans.py` · `common/config.py` · `eval/rigs/{slam_offline_eval,survivor_offline_test,inprocess_rig}.py`, `eval/demo_scoring.py` · `scripts/{run_disaster_demo,kit_postprocess_check}.py` ·
tests `test_scene_environment, test_perception_mapping, test_planning, test_closed_loop, test_ros_sensors, test_ros_demo` · `docs/DEMO_REPORT.md`, `docs/demo_frames/`.

## D. Modified files
`config/drone.yaml` (thermal camera, demo/slam/perception/survivors/map/astar/vfh/nav/video sections, world scenario, derived `sensor_extrinsics`), `tools/build_urdf.py` + `urdf/disaster_uav.urdf` (thermal camera link with 30 g taken from the ESC allowance — total still 2.200 kg; visible airframe colours; sensor-extrinsics block),
`worlds/earthquake_world.py` (new humans, demo obstacles, fire, textures), `sim/{rgbd_camera,vehicle_model,run_fc_sim,mixer}.py`, `ros2_bridge/uav_mavlink_bridge.py` (IMU/GPS/odom stamped in FC time), `eval/ground_truth.py`, `scripts/run_all_tests.py`, `tests/{test_model_urdf,test_rgbd,test_mixer,test_ground_truth_firewall}.py`, `README.md`, `ALGORITHM_INTERFACE.md`, `requirements.txt` (not changed; `onnxruntime` is needed for the detector).
A real bug found on the way and fixed: the saturating mixer assumed a perfectly centred CG (it leaked ~0.02 N·m at a 1 mm CG offset); it now uses the true collective allocation vector.

## E. ROS 2 topics
See README §1b and `ALGORITHM_INTERFACE.md`. Sensors: `/uav/camera/{color,depth,thermal}/{image_raw,camera_info}` (rgb8 / 32FC1 metres / 32FC1 °C), `/uav/imu` (FLU), `/uav/gps/fix`, `/uav/telemetry`. Autonomy: `/uav/slam/{pose,status,path}`, `/uav/map/{planning_grid,semantic}`, `/uav/perception/detections`,
`/uav/nav/{status,path}`, **`/uav/cmd_vel`**. Video only: `/demo/{main_view/image_raw,map_panel,rgb_annotated,mission_done,stop}`.

## F. MAVLink messages
Unchanged from stage 1. In the demo run the FC received 7 673 `SET_POSITION_TARGET_LOCAL_NED`, 387 `HEARTBEAT` and 18 `COMMAND_LONG` (arm / mode / land) from the companion, and streamed `LOCAL_POSITION_NED`, `ATTITUDE`, `GLOBAL_POSITION_INT`, `SYS_STATUS`, `HIGHRES_IMU`, `GPS_RAW_INT`, `RAW_RPM`, `COMMAND_ACK`, `STATUSTEXT` back.

## G. Coordinate-frame conventions
`/uav/cmd_vel` is **ENU** (x East, y North, z Up; angular.z CCW), documented in the node, the bridge, README and `ALGORITHM_INTERFACE.md`; the conversion to MAVLink NED lives only in `common/frames.py:ros_twist_to_ned` (used by the bridge; unit-tested).
`slam_map`: x East, y North, z Up, origin = take-off body position, aligned with gravity (accelerometer at rest, as the kit does) and with the FC heading at start. Camera optical frame: z forward, x right, y down. IMU: FRD on MAVLink, FLU on ROS. Camera / thermal / IMU / GPS extrinsics are static calibration in `config/drone.yaml → sensor_extrinsics` (derived from the URDF).

## H. Survivor geo-tagging pipeline
RGB → kit YOLO box → kit `localize()` (median torso depth, valid-pixel rule) → camera 3-D point → reject if the torso depth patch spans several surfaces (p90−p10 > 0.9 m) → thermal check by projecting the point into the thermal image (extrinsics from config) → fused confidence → `T_map_cam = T_map_body(SLAM) · T_body_cam` → local map point → tracker
(associate ≤ 1.5 m, 1/σ² filtering, confirm after ≥ 3 sightings and score ≥ 1.6, merge duplicates < 2.5 m) → `GeoFrame(anchor).enu_to_geodetic`. The anchor is the mean of on-ground GPS fixes with the antenna lever arm removed (`mapping/geo_anchor.py`). The tag is the survivor's own computed position, not the UAV's GPS position (unit-tested). Nothing from PyBullet is used.

## I. Thermal simulation method
Synthetic, **not radiometric**: the scene is rendered through the thermal camera's real geometry (own URDF link, 160×120, 57°×44°); a segmentation pass maps each object to a table temperature (skin 34 °C, clothing 29 °C, ground/rubble ambient ± sun/shade from brightness, burning debris 450 °C, sky −20 °C), then optics blur and 0.05 °C NETD noise. 32FC1 °C.
The table exists only inside the sensor. Thermal supports (+0.25) or contradicts (−0.15) an RGB detection and is the sole detector of hot hazards; it is not used by SLAM.

## J. A\* configuration (`config → astar`)
0.25 m cells on the live planning layer (a 1.1 m altitude band of the 3-D voxel map at 2 m flight altitude), 8-connected, clearance 0.75 m from occupied cells, unknown never traversable (partial path to the closest reachable cell while exploring), hazard regions + 0.6 m blocked, turn penalty 0.15, ≤ 60 000 nodes, re-plan every 1 s or when new map data blocks the path, string-pulled path.

## K. VFH+ configuration (`config → vfh`)
72 sectors, active radius 3.5 m, vehicle radius 0.45 m + 0.15 m margin, certainty weighting a = 5, b = 1.4, hysteresis thresholds 1.2 / 2.4, minimum opening 4 sectors, smoothing 3, cost weights target 5 / current 2 / previous 2, speed scaled down within 1.6 m of obstacles. Input: OCCUPIED cells of the live map.

## L. Mission sequence (from `nav_events.jsonl` of the final run, simulation time)
1.3 s ARM → 1.5 s OFFBOARD → 2.0 s TAKEOFF (2 m) → 4.8 s SEARCH WP1 (A\* plans around the 4.6 m wall on the growing map; from 12.0 s VFH+ flags a blocked target direction with obstacles inside its 3.5 m window, max deviation 28°) → 33.8 s **INSPECT: survivor #01 confirmed** (first seen 33.4 s) → WP2 → WP3 (the fire region is blocked in the planning layer) → 62.5 s **survivor #02 confirmed** (first seen 62.0 s) → WP4 → 80.0 s HOLD_END → 86.1 s DONE. Search waypoints are scripted; all motion is A\* + VFH+ over the live map.
The mission implements SLAM-loss handling (hold, slow sweep back to a familiar view, never ground truth) and gates take-off on SLAM + GPS-anchor + map readiness; **tracking loss did not occur in the final run**, so that branch was not exercised end to end (it is covered by logic only).

## M. Test results (all executed; `logs/validation/20260930_193123/summary.json`)
| Check | Result |
|---|---|
| Python syntax (all packages) | PASS |
| URDF / yaml up to date (thermal link, mass) | PASS (mass 2.200000 kg) |
| pytest (108 tests collected) | **105 passed**, 3 ROS-marked tests run separately below |
| — existing stage-1 suites (model, mixer, motor/battery, physics, wire, RGB-D, tuning, firewall) | pass (mixer test tolerance and the RGB-D survivor test were updated for the new CG / scene; a real mixer bug was fixed) |
| — **environment/scene test** (8): UAV loads at 2.2 kg with no initial collision, camera 10° down & optical axes, RGB/depth/thermal validity, thermal humans + fire aligned with RGB, IMU/GPS at rest, obstacle collisions, survivors detected by the kit YOLO, stable hover in the disaster world | pass |
| — perception/mapping unit tests (11): thermal projection round-trip, support/contradiction, hot-hazard localisation, GPS anchor lever arm, geo-tag from map position, tracker persistence / merge, unknown-is-not-free, SLAM frame chain | pass |
| — planner tests (7): A\* clearance / around wall / never through unknown, VFH+ free / avoid / surrounded | pass |
| — **closed-loop** (4, slow): SLAM flight, survivor pipeline, navigation around wall + pole, full in-process mission | pass |
| Calibration 1–10 (mass/inertia, propulsion fitters + report, controller tuning, camera intrinsics/extrinsics) | PASS |
| **Kit's own `postprocess.py` on a simulated flight** | PASS (kit 14.86 m vs online 14.89 m) |
| ROS 2 sensor topics (encodings, frames, camera_info, stamps in sync) | PASS (2 tests) |
| **Full ROS 2 → MAVLink → FC → PyBullet autonomous mission** (`tests/test_ros_demo.py`, 6 min) | PASS |
| 16-step ROS/MAVLink communication test from stage 1 | PASS 16/16 |

Individual stage tests (SLAM, survivor, navigation, thermal, geo-tag, MAVLink) are the ones listed above; they ran, and the failures found on the way were fixed and re-run (see "Problems found" below).

**Final demo run** (`logs/demo/final/score_report.json`, ground truth used offline only): 87 s simulated, wall 390 s. SLAM tracked all 433 poses (100 %); ATE RMSE **0.68 m**, final error 1.38 m over a 35.7 m path (**3.7 % drift**). Survivors: S1 confirmed with 64 observations, 48 with thermal support, local error 0.51 m, geo error 0.46 m; S2 64 observations, 27 thermal, local error 1.0 m, geo error 1.06 m;
no false or duplicate confirmed nodes; the lying survivor S3 (off-route, hard for the COCO detector) was correctly not found. Fire hazard: confirmed, localised 0.59 m from the true centre, 450 °C. In-flight collisions: 0. VFH+ blocked-target ticks: 301 (max deviation 28°).
**Run-to-run variability is real:** over my ROS runs of this same mission the SLAM ATE ranged **0.10–0.68 m** (0.07–0.31 m in the in-process rigs); the asynchronous nodes and RANSAC make runs non-identical, and survivor position error follows it (local error 0.05–1.0 m across runs). The acceptance test allows ATE < 0.8 m, so a bad run could fail it.

## N. Video output location
`logs/demo/final/disaster_demo.mp4` (86.4 s, 24 MB). Stills: `docs/demo_frames/` (take-off, map building, survivor detection, survivor #1 geo-tag, hazard/search, survivor #2 geo-tag, end hover). Every demo run writes its own `logs/demo/run_<timestamp>/`.

## O. Remaining limitations (nothing below is hidden by the demo)
* **SLAM is not ORB-SLAM3** and has no loop closure; drift 0.5–3.7 % of path. ORB-SLAM3 integration is code only (Linux/Jetson), untested.
* Thermal is synthetic (object-table temperatures, no radiometry). Visual scene is procedural (textured boxes/ellipsoids), not photorealistic; survivors are procedural humanoids that the COCO detector recognises when standing or sitting, not when lying at a low oblique angle.
* Survivors S1/S2 are detected only from within ≈ 10 m and at the demo's fixed 2 m altitude; the mission and waypoints are deterministic and were tuned on this scene. Other scenes / seeds were not evaluated.
* Planning uses a single altitude band; obstacles below ≈ 1.5 m (most rubble) are mapped (surface classes) but do not block the band. One hazard class (fire).
* SLAM-loss handling was implemented but not triggered in the final run. GPS-denied operation and SLAM reset were not exercised.
* The simulation runs at ≈ 0.22× real time with the video compositor (390 s wall for 87 s of flight) because software rendering costs ≈ 0.3 s per frame; cameras run at 5 Hz.
* Windows-only validation; no Linux, Jetson, Raspberry Pi, or hardware.

## P. Exact command to reproduce
```powershell
cd "C:\Users\Chaitra Rane\disaster_uav_pybullet_mavlink_fc"
. .\scripts\env_micromamba.ps1 uavros          # PATH + $env:UAV_PYTHON for the micromamba env (pybullet, ROS 2, onnxruntime)
python scripts/run_disaster_demo.py            # ~6-7 min wall -> logs/demo/run_<timestamp>/disaster_demo.mp4
python scripts/run_all_tests.py --ros-python $env:UAV_PYTHON     # the full validation (~45 min)
```
(`onnxruntime` must be installed in that environment: `python -m pip install onnxruntime`.)

## Simulated vs reused vs not real
* **Simulated:** PyBullet dynamics, the flight controller, motors/battery, IMU, GPS, RGB-D, the **thermal camera**, the disaster scene and the humans.
* **Real / reused software:** the attached kit's YOLOv8n detector + model, `localize()`, gravity alignment, its clustering ideas, and its `postprocess.py` (run unmodified on a simulated flight). ORB-SLAM3 was **not** run.
* **Not yet real:** Pixhawk hardware, real propulsion (all propulsion/battery parameters are placeholders), a real thermal camera, a real GPS environment, real flight.

## Problems found and fixed during this stage (root cause → fix → re-run)
1. SLAM lost tracking on blocky low-frequency wall textures → richer multi-scale textures; then on repetitive ground → non-repeating earth-tone 4096 px ground texture (SLAM 100 % tracked afterwards).
2. Duplicate confirmed survivor nodes from background-biased torso depth (sitting person) → depth-spread rejection gate + online tracker merge (radius 2.5 m).
3. ROS image publishing cost ~285 ms per frame (rclpy validates a `bytes` payload per element) → `array.array('B')` (0.5 ms).
4. Video HUD text ghosting (OpenCV glyph spacing differs by thickness) → outline drawn by offsets; H.264 encoding (71 MB → 24 MB).
5. Saturating mixer leaked torque with an off-centre CG → true collective allocation vector.
6. nav-node crash in the A\* logging (`KeyError: 'stats'`) → fixed and the demo re-run.
