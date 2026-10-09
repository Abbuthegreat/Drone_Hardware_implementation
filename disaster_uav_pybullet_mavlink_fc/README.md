# Disaster-response UAV — PyBullet + MAVLink flight-controller simulation

PyBullet-only simulation of a 2.2 kg X-quadrotor (Pixhawk 6C Mini + Raspberry Pi 5 + Matek 3901-10X GPS + RGB-D camera) flying an
**autonomous search mission** in an outdoor earthquake scene. **No Gazebo, no PX4 SITL.** The simulated flight controller is our own code,
sits behind a *real* MAVLink 2 / UDP link, and is a drop-in replaceable layer for the real Pixhawk.

On top of that vehicle stack runs one connected autonomy chain: RGB-D SLAM (ORB-SLAM3 on Linux, a VO fallback on Windows), a live
metric + semantic + geographic map, person detection with the attached SLAM kit's YOLO detector, depth-based 3-D survivor localisation,
synthetic-thermal support, GPS geo-tagging of persistent survivor nodes, A\* global planning, VFH+ local avoidance, mission-level failsafes,
a command-centre status stream and a rescue report. The autonomy talks to the vehicle **only** through ROS 2 → MAVLink.

```
 PyBullet disaster world (static or dynamic scenario)
   │ RGB / depth / synthetic thermal (rendered through the URDF camera poses)      IMU / GPS (simulated sensors on the FC side)
   ▼                                                                                 │ MAVLink HIGHRES_IMU, GPS_RAW_INT, ATTITUDE ...
 ROS 2 sensor topics ◄───────────────────────────────────────────── ROS 2 ↔ MAVLink bridge ◄─┘
   │
   ├─► SLAM node (ORB-SLAM3 RGB-D | orb_pnp_vo) ──► /uav/slam/pose  (estimated pose; never the simulator pose)
   │                                                     │
   ├─► perception + mapping node: kit YOLO → torso depth → 3-D camera point → T_map_cam(SLAM) → map point → GPS anchor → lat/lon/alt
   │     + thermal support, survivor tracker (persistent nodes), fire hazards, 3-D log-odds voxel map → /uav/map/planning_grid, /uav/map/semantic
   │                                                     │
   └─► navigation node: mission (search waypoints → return → land) → A* on the live grid → VFH+ → safety layer
                                                         │  /uav/cmd_vel (ENU Twist) + /uav/arm /uav/offboard /uav/land
                                                         ▼
                     ROS 2 ↔ MAVLink bridge: validate → ENU→NED → SET_POSITION_TARGET_LOCAL_NED / COMMAND_LONG   (UDP 14540)
                                                         ▼
       simulated FC: setpoint watchdog → position/velocity/attitude/rate control → X-mixer → motor + ESC lag → battery
                                                         ▼
                       PyBullet rigid-body physics (rotor forces/torques) → sensors → telemetry → MAVLink → ROS 2 → back to the top
```

Real hardware swaps only the middle: `Algorithm → ROS 2 → MAVLink **serial** → Pixhawk 6C Mini → ESCs → S3110 motors`
(see [Hardware transition](#hardware-transition)). Navigation code never imports PyBullet (enforced by `tests/test_ground_truth_firewall.py`).

### What runs where

| Component | Windows 11 (`uavros` micromamba env: Python 3.11, ROS 2 Humble via RoboStack) | Ubuntu 22.04 + ROS 2 Humble |
|---|---|---|
| PyBullet world, vehicle, motors, mixer, simulated FC | ✅ | ✅ |
| MAVLink (pymavlink, real UDP), bridge, failsafe watchdog | ✅ | ✅ |
| ROS 2 nodes (bridge, SLAM node, perception/mapping, navigation, video) | ✅ (RoboStack Humble) | ✅ |
| RGB-D / synthetic thermal / IMU / GPS simulation | ✅ | ✅ |
| **ORB-SLAM3** (kit, `slam/rgbd_stream`) live SLAM | ❌ not built (needs Pangolin + a C++ toolchain; no WSL here) | ✅ built and validated (`docs/SLAM_INTEGRATION_UBUNTU.md`) |
| SLAM fallback `orb_pnp_vo` (RGB-D ORB + PnP odometry, no loop closure) | ✅ used automatically (`--slam-backend auto`) | ✅ (optional) |
| Survivor detection (kit YOLOv8n, onnxruntime), depth localisation, geo-tagging, thermal support | ✅ | ✅ |
| A\* + VFH+ + mission + failsafes + command-centre output | ✅ | ✅ |
| Full autonomous mission over ROS 2 + MAVLink (`scripts/run_disaster_demo.py`) | ✅ with the VO fallback | ✅ with ORB-SLAM3 |

The SLAM backend is behind one interface (`slam/backend.py`), so the Linux ORB-SLAM3 path is unchanged; Windows does **not** run ORB-SLAM3.

---------------------------------------------------------------------------------------------------

## 1. Quick start

### 1.1 Windows (validated on Windows 11, the current development machine)
pybullet has no PyPI wheel for Windows/Python >= 3.12 and needs MSVC to build, and ROS 2 is not a pip package, so everything comes from
conda-forge + RoboStack in one micromamba environment `uavros`:
```powershell
# once: micromamba (single exe, no admin) -> C:\Users\<you>\mm
mkdir $env:USERPROFILE\mm; cd $env:USERPROFILE\mm
curl.exe -L -o mm.tar.bz2 https://micro.mamba.pm/api/micromamba/win-64/latest; tar -xjf mm.tar.bz2
$env:MAMBA_ROOT_PREFIX = "$env:USERPROFILE\mm\root"
.\Library\bin\micromamba.exe create -y -n uavros -c conda-forge -c robostack-staging --channel-priority flexible `
    python=3.11 ros-humble-ros-base pybullet numpy scipy pyyaml pytest opencv pip
& "$env:USERPROFILE\mm\root\envs\uavros\python.exe" -m pip install pymavlink onnxruntime
```
Every new PowerShell:
```powershell
cd <repo>
. .\scripts\env_micromamba.ps1 uavros            # PATH + $env:UAV_PYTHON (numpy / pybullet / rclpy DLLs)
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"         # ROS's launch_testing pytest plugin breaks pytest >= 8 (run_all_tests.py sets this itself)
```
| What | Command (Windows, after the two lines above) |
|---|---|
| PyBullet + simulated FC on MAVLink UDP 14540 | `.\scripts\start_sim.ps1 -Gui`  (or `& $env:UAV_PYTHON -m sim.run_fc_sim --gui [--scenario dynamic]`) |
| ROS 2 <-> MAVLink bridge (second terminal) | `.\scripts\start_bridge.ps1` |
| MAVLink only, no ROS (takeoff / square / land) | `& $env:UAV_PYTHON scripts/rpi_companion_demo.py` (simulator running) |
| 16-step ROS 2 -> MAVLink -> FC -> PyBullet communication test | `& $env:UAV_PYTHON scripts/comm_test.py` |
| **Autonomous disaster mission** (static scene, + video) | `& $env:UAV_PYTHON scripts/run_disaster_demo.py`  (`--no-video` is faster) |
| Autonomous mission, dynamic scene | `& $env:UAV_PYTHON scripts/run_disaster_demo.py --scenario dynamic --no-video` |
| Fast tests (about 2 min) | `& $env:UAV_PYTHON -m pytest tests -m "not ros and not slow" -p no:cacheprovider` |
| Slow closed-loop tests (in-process missions, about 15 min) | `& $env:UAV_PYTHON -m pytest tests/test_closed_loop.py -p no:cacheprovider` |
| Everything (pytest, calibration, ROS 2 tests, ROS mission, comm test) | `.\scripts\run_tests.ps1` |

On Windows the SLAM node runs the **`orb_pnp_vo` fallback**: `--slam-backend auto` (default) picks it because ORB-SLAM3 is not built there, and the run
log, `/uav/slam/status` and `demo_summary.json` say so. Call `python.exe` from the env directly (`micromamba run` and the `ros2.exe` launcher break on
paths with spaces); the ROS CLI works as `python <env>\Library\bin\ros2-script.py topic list --no-daemon`.

### 1.2 Linux (Ubuntu 22.04 + ROS 2 Humble) - required for live ORB-SLAM3
```bash
sudo apt install ros-humble-ros-base ros-humble-tf2-tools build-essential cmake git pkg-config libeigen3-dev libglew-dev libepoxy-dev \
                 libgl1-mesa-dev libboost-all-dev libssl-dev libopencv-dev
pip install --user -r requirements.txt onnxruntime        # system python3.10; rclpy comes from /opt/ros/humble
bash slam/build_orbslam3.sh                               # once: clones + patches + builds ORB-SLAM3 into ~/slam_deps (PATCHES.md)
source /opt/ros/humble/setup.bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/test_orbslam3.py -q               # ORB-SLAM3 runner present + YAML / camera consistency
python3 scripts/test_slam_pipeline.py                     # SLAM-only validation flight (ATE vs ground truth, offline)
python3 scripts/run_disaster_demo.py                      # full mission; auto-selects ORB-SLAM3 when built
python3 scripts/run_disaster_demo.py --scenario dynamic --no-video
python3 scripts/run_all_tests.py --ros-python $(which python3)
ROS_DOMAIN_ID=88 python3 tools/diagnose_slam.py           # while a demo runs: CAMERA/DEPTH/IMU/TF/SLAM/POSE/GPS checks
```
Manual bring-up: `./scripts/start_sim.sh --gui` (terminal 1) and `./scripts/start_bridge.sh` (terminal 2). Details: [docs/SLAM_INTEGRATION_UBUNTU.md](docs/SLAM_INTEGRATION_UBUNTU.md).

### Command the drone through ROS 2
```bash
ros2 topic list
ros2 topic echo --once --full-length /uav/telemetry
ros2 service call /uav/arm      std_srvs/srv/SetBool "{data: true}"
ros2 service call /uav/offboard std_srvs/srv/SetBool "{data: true}"
ros2 topic pub -r 10 /uav/cmd_vel geometry_msgs/msg/Twist "{linear: {z: 1.0}}"       # climb 1 m/s
ros2 topic pub -r 10 /uav/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 1.5}}"       # fly EAST 1.5 m/s
ros2 service call /uav/land     std_srvs/srv/Trigger {}
```

---------------------------------------------------------------------------------------------------

## 1b. Autonomous disaster-response mission

`run_disaster_demo.py [--no-video] [--scenario static|dynamic] [--slam-backend auto|orbslam3|orb_pnp_vo]` starts six processes on one ROS 2 domain and
shuts them down cleanly: **(1)** PyBullet + simulated FC + RGB / depth / thermal / chase-view publishers, **(2)** the ROS 2 <-> MAVLink bridge,
**(3)** SLAM node, **(4)** perception + mapping node, **(5)** navigation node, **(6)** video compositor. Nothing is operated by hand after the start:

`WAIT_SLAM (SLAM + GPS anchor + map ready) -> ARM -> OFFBOARD -> TAKEOFF (2 m) -> SEARCH WP1..WP4 (A* + VFH+ on the live map; INSPECT pause per confirmed
survivor) -> RETURN home -> HOLD_END -> LANDING (MAVLink LAND) -> DONE (FC auto-disarm)`.

The search waypoints (`config -> demo.search_waypoints_local`) are the *search objective* (regions to look at), not a route: no obstacle position is
given to navigation; every motion command comes from A\* + VFH+ over the map built from the RGB-D camera.

```
 RGB-D + IMU ──► SLAM node ──► /uav/slam/pose (UAV body, slam_map = ENU-like, origin = take-off point)
     │                                   │
     ├─► kit YOLOv8n person detector     │
     │        └─► kit localize(): torso-median depth ──► 3-D camera point
     │                └─► thermal check (project into the thermal camera) ──► fused confidence
     │                        └─► T_map_cam = SLAM pose x URDF extrinsics ──► map point ──► GPS anchor ──► lat / lon / alt
     │                                └─► SurvivorTracker (associate / filter / confirm) ──► persistent SURVIVOR nodes
     ├─► depth ray casting ─► 3-D log-odds voxel map (FREE / OCCUPIED / UNKNOWN + surface classes)   ┐ ONE geo-semantic map
     └─► thermal hot spots (+ RGB flame colour) ─► HAZARD nodes (fire)                                ┘ (/uav/map/planning_grid, /uav/map/semantic)
                                                                     │
                     deterministic search waypoints ──► A* on the live map ──► VFH+ ──► /uav/cmd_vel (ENU Twist)
                                                                                           │
                              ROS 2 bridge ──► SET_POSITION_TARGET_LOCAL_NED ──► simulated FC ──► PyBullet
```

**What is reused from the attached SLAM kit (`slam_kit_3/`, left unmodified)** — see `slam/kit.py`:

| Kit component | How it is used here |
|---|---|
| `detector.py` `Detector` + `models/yolov8n.onnx` | the person detector on the simulated RGB (onnxruntime backend), exactly as in the kit |
| `postprocess.py` `localize()` | torso-median depth → 3-D camera point of every detection (kit rule, incl. `min valid pixels`) |
| `postprocess.py` `rot_a_to_b()` + the kit's "gravity 'up' from the accelerometer at start" idea (`record.py`) | gravity alignment of the SLAM world (`slam/adapter.py`) |
| `postprocess.py` clustering (`--radius`, `--min-obs`, merge < 0.8·radius) | applied ONLINE in the survivor tracker (`mapping/geo_semantic_map.py`), same parameters |
| `postprocess.py` `TopDown`/`VoxelMap` map ideas, `record.py` `write_orbslam_yaml`, run-folder format | dataset export in the kit's format; `scripts/kit_postprocess_check.py` runs the kit's **own** `postprocess.py` on a simulated flight and compares |
| `slam/rgbd_offline.cc`, `setup_orbslam3.sh` | adapted to a streaming runner `slam/rgbd_stream.cc` + `slam/build_orbslam3.sh` (built and run on Ubuntu 22.04) |

**SLAM backend.** Default `slam.backend: orbslam3`: the attached kit's **ORB-SLAM3 (RGB-D)**, built on Ubuntu 22.04 by `bash slam/build_orbslam3.sh` and run as the streaming process `rgbd_stream`
(`slam/orbslam3_backend.py`). Validation on this machine: `python scripts/test_slam_pipeline.py` (100 % tracked, ATE RMSE ≈ 0.11 m over a 10.6 m take-off/translate/rotate/translate flight) and the full demo (ATE RMSE 0.06–0.09 m over 37 m);
see [docs/SLAM_INTEGRATION_UBUNTU.md](docs/SLAM_INTEGRATION_UBUNTU.md) and [PATCHES.md](PATCHES.md). `slam.backend: orb_pnp_vo` (`slam/vo_backend.py`, pure-Python RGB-D ORB + PnP odometry, no loop closure) remains as a fallback; it was the only option on the earlier Windows setup.
The IMU is used only for gravity/heading alignment at start (as in the kit); ORB-SLAM3 runs RGB-D only (the physical kit does not use VIO). `python tools/diagnose_slam.py` checks CAMERA/DEPTH/IMU/TF/SLAM/POSE/GPS on a running system.

**ROS 2 topics** (frame conventions in brackets)

| Topic | Type | Notes |
|---|---|---|
| `/uav/camera/color/image_raw`, `/uav/camera/depth/image_raw`, `/uav/camera/thermal/image_raw` | `Image` `rgb8` / `32FC1` metres / `32FC1` °C | one synchronised capture, same stamp (simulation time), 5 Hz |
| `/uav/camera/{color,depth,thermal}/camera_info` | `CameraInfo` | frames `rgbd_camera_optical_link`, `thermal_camera_optical_link` (z fwd, x right, y down) |
| `/uav/imu` (FLU, 100 Hz), `/uav/gps/fix`, `/uav/telemetry` | from MAVLink via the bridge | stamped with the FC clock = the same time base as the images |
| `/uav/slam/pose`, `/uav/slam/status`, `/uav/slam/path` | `PoseStamped` / JSON / `Path` | `slam_map`: x East, y North, z Up, origin = take-off body position |
| `/uav/map/planning_grid`, `/uav/map/semantic` | `OccupancyGrid` (0 free, 100 occ, −1 unknown) / JSON | survivor + hazard nodes with lat/lon |
| `/uav/perception/detections`, `/demo/rgb_annotated`, `/demo/map_panel`, `/demo/main_view/image_raw` | JSON / `Image` | the last three are for the video only |
| `/uav/nav/status`, `/uav/nav/path`, **`/uav/cmd_vel`** | JSON / `Path` / `Twist` | **cmd_vel is ENU: x East, y North, z Up [m/s], angular.z CCW [rad/s]** (bridge `cmd_frame:=enu`); converted to NED only inside the bridge |

**MAVLink** (unchanged): `SET_POSITION_TARGET_LOCAL_NED` (velocity + yaw-rate) at 20 Hz, `COMMAND_LONG` arm / mode; back: `HEARTBEAT`, `LOCAL_POSITION_NED`, `ATTITUDE`, `GLOBAL_POSITION_INT`,
`SYS_STATUS`, `HIGHRES_IMU`, `GPS_RAW_INT`, `RAW_RPM`, `COMMAND_ACK`, `STATUSTEXT`.

**Survivor geo-tag pipeline.** detection box → kit `localize()` → camera point → **× T_map_cam** (`T_map_body(SLAM) · T_body_cam(URDF extrinsics from config)`) → local map point → tracker (kit-style
association radius 1.5 m, confirm after ≥ 3 sightings **and** score ≥ 1.6, merge duplicates < 2.5 m, reject detections whose torso-depth patch straddles several surfaces) → `GeoFrame(anchor).enu_to_geodetic`.
The anchor is the mean of GPS fixes taken on the ground with the antenna lever arm removed (`mapping/geo_anchor.py`). The survivor's geo-tag is therefore its **own** position, not the UAV's.
Each node stores `track_id, class, confidence, local_x/y/z, latitude, longitude, altitude, position_uncertainty, first_seen, last_seen, observation_count, status (CANDIDATE|CONFIRMED), evidence_frame`.
The map panel draws yellow = candidate, red = confirmed with `SURVIVOR #nn / LAT / LON / ALT / CONF`.

**Thermal camera (synthetic — not radiometric).** PyBullet has no thermal model. `sim/thermal_camera.py` renders the scene through the *real* thermal camera geometry (its own link, 160×120,
57°×44°, extrinsics in the URDF) with a segmentation pass and converts each object to a temperature from a per-object table (skin 34 °C, clothing 29 °C, ground/rubble ambient ± sun/shade from scene brightness,
burning debris 450 °C, sky −20 °C), then applies optics blur and NETD noise (0.05 °C). Output `32FC1` °C. The table is used **only** inside the sensor; autonomy sees pixels. Thermal is not fed to SLAM.
Fusion: the RGB+depth 3-D point is projected into the thermal image; warm-and-contrasting → +0.25 confidence, in view but cold → −0.15 (nothing is invented when thermal shows nothing). Hot blobs > 120 °C become hazard nodes (localised by intersecting the thermal ray with the depth image).

**Planners.** `planning/astar.py` (2.5-D: x-y planning on an altitude band of the 3-D map around the flight altitude): 8-connected A* on the live planning layer, `clearance_m` 0.75 from occupied cells, UNKNOWN never traversable (partial path to the closest reachable cell while exploring), hazard regions blocked, re-plan every 1 s or when the path is blocked by new map data.
`planning/vfh_plus.py`: VFH+ (72 sectors, active radius 3.5 m, hysteresis thresholds, cost = target / current / previous heading) on the OCCUPIED cells of the live map around the UAV. The mission (`mission/disaster_demo.py`) is a scripted
search objective (4 waypoints + return, `config → demo`), yet every command is produced by A\* + VFH+. All parameters live in `config/drone.yaml` (`demo`, `slam`, `perception`, `survivors`, `map`, `astar`, `vfh`, `nav`, `video`).

**Failsafes / safe states** (mission layer `mission/disaster_demo.py`; vehicle layer = the FC watchdog, §5). Localisation lost → `LOST_HOLD` (hover, sweep back) → `LANDING` after
`nav.lost_land_s`; stale map (sensor / mapping loss) → `MAP_STALE` hover; no MAVLink telemetry → `LINK_LOST` (and the FC's own 0.5 s setpoint watchdog → `FAILSAFE_HOLD`);
FC in `FAILSAFE_HOLD` / `LAND` → the mission stops commanding and lands; disarm in flight → `ABORTED`; no A\* route → `NO_ROUTE_HOLD` (hover, look toward the goal, re-plan,
skip the waypoint after a timeout, reported); VFH+ without an opening → `VFH_BLOCKED`; obstacle closer than 0.7 m → `AVOID_RETREAT` (back away from the nearest occupied cell:
the side comes from the map, never a fixed left/right rule); end of a partial path → observe only, never flies into unknown space; non-finite commands → hover (the bridge
also rejects them); stale commands → the bridge streams zero velocity after 0.5 s. Table in [ALGORITHM_INTERFACE.md](ALGORITHM_INTERFACE.md); unit tests `tests/test_mission_safety.py`.

**Command-centre output.** During the mission `command_centre/status.jsonl` gets one record per second: UAV **estimated** position (SLAM; lat/lon through the GPS anchor),
altitude, mission state, current target, A\* path, VFH+ direction and nearest obstacle, survivors (id, status, confidence, local xyz, lat/lon/alt, uncertainty, observations,
thermal support, timestamps), hazards, MAVLink link state, FC mode / armed / battery, failsafe state, SLAM state and backend. A one-line human-readable version is printed every 5 s
(`[cc] ...`). At the end: `rescue_report.json` + `rescue_report.md` (confirmed survivors with geo-tags, candidates, hazards, mission events) and `demo_summary.json`.

**Static and dynamic disaster environment.** `world.scenario: static` (default) is the deterministic earthquake scene: collapsed buildings, leaning walls, rubble, fallen
beams / slabs, poles, trees, wrecked vehicles, a burning-debris fire hazard and three survivors (standing / sitting / lying). `--scenario dynamic` adds scripted world events
(`world.dynamic_obstacles`, `worlds/dynamic_obstacles.py`): an **aftershock slab** that falls across the WP3 → WP4 leg when the UAV approaches (never onto it) and
**sliding debris** moving back and forth at 0.2 m/s across the WP2 → WP3 corridor. The world moves them, not the UAV; navigation only learns about them through RGB-D → map →
A\* / VFH+. Their specs and trigger times are written to `sim/world_truth.json` / `sim/world_events.json` (evaluation only).

**Ground-truth firewall (extended).** `slam/ perception/ mapping/ planning/ mission/ video/ ros2_bridge/ algo/ companion/` may not import `pybullet`, `sim`, `worlds`, `eval`, `tests` (AST test) and the SLAM / mapping / navigation nodes may not subscribe to the FC pose (`/uav/odom`). Test rigs that need ground truth for scoring live in `eval/rigs/`; `eval/demo_scoring.py` reads ground truth only after the flight.

**Logs** (`logs/demo/<run>/`): `slam/slam_pose.csv`, `perception/detections.jsonl` (per frame: boxes, ranges, thermal checks, track ids), `perception/survivor_map.jsonl` (nodes + geo-tags + hazards), `perception/evidence/`, `nav/nav_log.csv`
(state, target, pose, command, VFH decision, A* partial flag), `nav/astar_paths.jsonl`, `telemetry/fc_telemetry.csv` (mode, armed, failsafe/critical flag, GPS, battery), `telemetry/gps.csv`, `telemetry/frames.csv` (RGB/depth/thermal frame index + stamp),
`sim/ground_truth.csv` (evaluation only), `sim/fc_mavlink.log`, `companion_mavlink.log`, `score_report.json`, `disaster_demo.mp4`, `command_centre/status.jsonl`,
`rescue_report.{json,md}`, `demo_summary.json`; `nav/nav_events.jsonl` also holds the mission / safety events (`kind: event`).

**Tests for this stage:** `tests/test_scene_environment.py` (scene validation), `test_perception_mapping.py`, `test_planning.py`, `test_mission_safety.py` (failsafes / safe states),
`test_closed_loop.py` (SLAM per backend, survivor pipeline, navigation, full mission, dynamic-scenario mission with return + landing; slow), `test_ros_sensors.py`,
`test_ros_demo.py` (full ROS + MAVLink mission), `scripts/kit_postprocess_check.py` (kit's own postprocess on a simulated flight). Results: [docs/DEMO_REPORT.md](docs/DEMO_REPORT.md).

---------------------------------------------------------------------------------------------------

## 2. Repository layout

| Path | Role |
|---|---|
| `config/drone.yaml` | **All** calibration: components, derived inertia, propulsion, battery, aero, FC gains, sensors, geo origin, MAVLink, world |
| `urdf/disaster_uav.urdf` | Generated by `python -m tools.build_urdf` from `drone.yaml` (also rewrites the derived inertia block) |
| `common/` | Shared pure-numpy code: `frames.py` (NED/ENU/FRD/FLU, quaternions), `geo.py` (map↔lat/lon), `mav.py`, `fc_modes.py` |
| `sim/` | Vehicle side: `sim_core.py`, `physics_pybullet.py`, `motor_model.py`, `battery.py`, `mixer.py`, `controllers.py`, `flight_controller_sim.py`, `sensors.py`, `mavlink_fc_server.py`, `rgbd_camera.py`, `run_fc_sim.py` |
| `worlds/earthquake_world.py` | Disaster scene (rubble, collapsed structures, poles, trees, wrecks, 5 survivors) |
| `companion/mavlink_client.py` | **The only vehicle interface for autonomy** (`CompanionMavlink`) |
| `ros2_bridge/uav_mavlink_bridge.py` | ROS 2 node: `/uav/cmd_vel` → MAVLink, MAVLink telemetry → ROS |
| `algo/` | Sensor/survivor data contracts (`interfaces.py`) and the original entry-point stub (superseded by `mission/` + the nodes) |
| `slam/` | `kit.py` (access to the unmodified attached kit), `backend.py`, `vo_backend.py` (ORB+PnP RGB-D VO, fallback), `orbslam3_backend.py` + `rgbd_stream.cc` + `build_orbslam3.sh` (kit's ORB-SLAM3, streaming runner; default on Linux; `resolve_backend()` picks the fallback where it is not built), `adapter.py` (IMU / gravity / heading / extrinsics → UAV pose) |
| `perception/` | `person_detector.py` (kit YOLO + kit `localize`), `thermal_fusion.py` (RGB↔thermal projection, support/contradiction, hot hazards) |
| `mapping/` | `metric_map.py` (3-D voxel map), `geo_semantic_map.py` (survivor / hazard nodes, tracker, geo reference), `geo_anchor.py`, `pipeline.py` |
| `planning/` | `astar.py`, `vfh_plus.py` |
| `mission/` | `disaster_demo.py` (search mission state machine + safety layer → A* / VFH+ velocity commands + ARM / OFFBOARD / LAND actions), `command_centre.py` (status stream + rescue report) |
| `video/` | `map_panel.py`, `compositor_node.py`, `textutil.py` |
| `ros2_bridge/` | + `slam_node.py`, `perception_mapping_node.py`, `nav_node.py`, `ros_util.py` |
| `sim/` | + `thermal_camera.py` (synthetic thermal), `ros_sensor_publisher.py` (RGB / depth / thermal / chase view) |
| `worlds/` | `earthquake_world.py` (scene, obstacles, fire; static / dynamic scenario), `dynamic_obstacles.py` (secondary collapse, sliding debris), `humans.py` (detector-compatible survivors) |
| `eval/ground_truth.py` | **Evaluation-only** ground truth (CSV + UDP JSON). Firewalled from autonomy |
| `eval/rigs/`, `eval/demo_scoring.py` | offline test rigs (SLAM flight, survivor test, in-process mission) and the post-flight scoring |
| `calibration/` | mass/inertia, propulsion fitters, controller tuning, camera intrinsics/extrinsics |
| `tests/` | pytest suites (model, mixer, physics, wire, RGB-D, firewall, ROS 2, tuning) |
| `scripts/` | start scripts, `comm_test.py`, `diagnose.py`, `run_all_tests.py`, `record_rgbd.py`, demo |
| `logs/` | tuning CSVs, calibration reports, communication-test logs, validation runs |

---------------------------------------------------------------------------------------------------

## 3. Physical model

### Mass budget and geometry (`components` in `config/drone.yaml`; positions in the URDF `base_link` frame, FLU)
Total = **2.200000 kg**, verified by summing the URDF XML (test `test_total_mass_is_2_2_kg_summed_from_urdf_links`).
Component masses are placeholders except for the total; **weigh the real parts and edit the table.**

| Link | Mass kg | Position (x fwd, y left, z up) m |
|---|---|---|
| airframe (2 plates, 4 arms, skids/struts) | 0.769 | plates z=±0.025, arms to (±0.2121, ±0.2121, 0), skids z=−0.14 |
| `base_link` (1 g reference node) | 0.001 | 0, 0, 0 |
| `flight_controller_link` (Pixhawk 6C Mini) | 0.070 | 0, 0, +0.035 (≈ 5.6 cm above the CG) |
| `raspberry_pi_link` | 0.070 | −0.075, 0, +0.038 |
| `esc_link` (4-in-1 ESC + PDB + wiring) | 0.172 | 0, 0, 0 |
| `gps_link` (Matek 3901-10X, on a mast) | 0.030 | −0.110, 0, +0.200 (0.17 m from the Pi) |
| `battery_link` (6S 8400 mAh) | 0.650 | **−0.013154 (solved so CG x = 0)**, 0, −0.0575 |
| `rgbd_camera_link` (+ `rgbd_camera_optical_link`) | 0.090 | +0.190, 0, −0.005, pitched **10° down** |
| `motor_i_link` ×4 / `rotor_i_link` ×4 | 0.080 / 0.007 | (±0.2121, ±0.2121, 0) / +0.022 above the motor |

X frame, arm length 0.30 m (placeholder — frame size was not specified). **Composite CG = (0, 0, −0.0209) m** in `base_link`.

### Inertia about the CG (derived, stored in `drone.yaml → mass_properties`)
Method: every component is a primitive (box / cylinder / rod) with its analytic inertia, combined with the parallel-axis
theorem (`tools/build_urdf.py`). An **independent point-cloud integration** in the test-suite reproduces it within 3 %.
The matrix is positive definite (principal moments 0.0295, 0.0353, 0.0539).

| Ixx | Iyy | Izz | Ixy | Ixz | Iyz |
|---|---|---|---|---|---|
| 0.029517 | 0.035333 | 0.053935 | 0.000000 | 0.000453 | 0.000000 kg·m² |

PyBullet's own merge of fixed links reports wrong Ixx/Iyy (verified dynamically: 44 vs 34 rad/s² for 1 N·m), so the
engine's inertia **diagonal is overridden from the yaml**; Ixz (~1 % cross-coupling) is neglected by the engine.

### Propulsion — all placeholders, **not** an S3110 measurement (`propulsion` in `drone.yaml`)
| Parameter | Value | Notes |
|---|---|---|
| `motor_kv_rpm_per_v` | 800 | datasheet, *no-load* only: 15 360 / 17 760 / 20 160 rpm at 19.2 / 22.2 / 25.2 V. Never used as loaded rpm |
| `max_loaded_rpm_at_nominal_v` | 9000 | placeholder; loaded speed ∝ bus voltage |
| `thrust_coeff_kT` | 1.351e-5 N/(rad/s)² | `T = kT·ω²` → 12 N/motor at 9000 rpm, T/W = 2.2 |
| `torque_coeff_kQ` | 2.027e-7 N·m/(rad/s)² | `Q = kQ·ω²` (kQ/kT = 0.015 m) |
| `motor_time_constant_s` | 0.045 | `dω/dt = (ω_target − ω)/τ` |
| `idle_rpm`, `command_expo`, `esc_motor_efficiency`, `avionics_power_w` | 900, 1.0, 0.70, 12 W | placeholders |

Hover sanity check (tested): 2.2 × 9.81 = **21.582 N** = 4 × **5.3955 N**; simulated hover thrust 21.58 N; hover ≈ 6035 rpm,
normalised command 0.63 at 22.2 V (0.75 at 19.2 V, 0.55 at 25.2 V). This is a property of the placeholder model, **not** a claim about a
real S3110 + propeller throttle value.

### Battery (`battery` in `drone.yaml`)
6S, 8400 mAh, 19.2 / 22.2 / 25.2 V. Open-circuit voltage from an SOC table, sag = I·R (R = 0.030 Ω placeholder), current from
electrical power, initial SOC 0.95, warn 21.0 V, failsafe 20.4 V filtered (or SOC < 15 %) → `LAND`, arming refused below 19.2 V.
Bus voltage limits the motors: available thrust ∝ V² (`calibration/propulsion_report.py` tabulates it).

### Aerodynamics
Body drag `F = −(k₁+k₂|v|)v` (separate xy / z), angular drag `τ = −(c₁+c₂|ω|)ω`, optional constant wind (`aero.wind_ned_mps`).
PyBullet's built-in damping is turned off so drag is explicit. Ground effect, blade flapping, propeller collisions: not modelled.

---------------------------------------------------------------------------------------------------

## 4. Coordinate conventions (also in `common/frames.py`)

| Frame | Axes | Where |
|---|---|---|
| MAVLink local / FC world | **NED**: x North, y East, z Down | MAVLink, FC internals, `CompanionMavlink` |
| PyBullet / ROS world | **ENU**: x East, y North, z Up | PyBullet, `/uav/cmd_vel` (default), `/uav/odom` |
| FC body | **FRD**: x forward, y right, z down | MAVLink attitude, IMU, mixer |
| URDF / ROS body | **FLU**: x forward, y left, z up | URDF, `/uav/imu`, orientation in `/uav/odom` |
| Camera optical | z forward, x right, y down | images, depth, intrinsics |

`v_enu = C·v_ned` with `C = [[0,1,0],[1,0,0],[0,0,−1]]`; `R_ned_frd = C·R_enu_flu·diag(1,−1,−1)`.
Heading (`yaw`) is 0 = North, **positive clockwise** seen from above; MAVLink `yaw_rate` is about the Down axis, so
**positive = clockwise = −(ENU z angular velocity)**. The conversion lives only inside the vehicle interface
(`sim/physics_pybullet.py` for the simulator, `common/frames.py` for the ROS bridge).
Map origin = PyBullet world origin; map North = +Y, East = +X.

---------------------------------------------------------------------------------------------------

## 5. Simulated flight controller

`SET_POSITION_TARGET_LOCAL_NED` → *setpoint interpretation* (type_mask, frames LOCAL_NED / LOCAL_OFFSET_NED / BODY_NED /
BODY_OFFSET_NED) → **position P** (kp 0.9/0.9/1.2) → **velocity PID** (kp 2.5/2.5/4, ki 0.4/0.4/1) → acceleration →
**thrust vector** (desired specific force `m(a−g)`, 35° tilt limit) → attitude setpoint with yaw → **attitude P** (tilt 6, yaw 3)
→ body-rate setpoint → **rate PID** (torque = J·demand + ω×Jω) → **mixer** with saturation handling → thrust→command map
(battery-voltage compensated) → **motor + ESC lag** → PyBullet forces at the rotor positions and yaw reaction torques.

Also implemented: arm/disarm (in-flight disarm refused unless forced, `param2 = 21196`), flight modes `HOLD`, `OFFBOARD`, `LAND`,
`FAILSAFE_HOLD`, landed detection / take-off from the ground only via an OFFBOARD climb demand, auto-disarm after landing,
low-battery failsafe → LAND, `SET_ATTITUDE_TARGET` (attitude + normalised thrust), telemetry.
The FC's own EKF is **not** modelled: it works from the simulated truth plus small configured errors
(`flight_controller.estimator`), with the gyro coming from the *simulated IMU*.

### Setpoint-loss watchdog
In `OFFBOARD`, no valid setpoint for `offboard_timeout_s = 0.5 s` → `FAILSAFE_HOLD` (brake to a stop, then hold position; reported as
`MAV_STATE_CRITICAL` + `STATUSTEXT`); after `failsafe_land_after_s = 20 s` → `LAND`. Late setpoints do **not** silently resume
control; the companion must request OFFBOARD again (which is refused unless a setpoint stream is present). *0.5 s is this
simulator's parameter — PX4's `COM_OF_LOSS_T` default is 1.0 s; check your own firmware. This is not PX4.*

### Motor mixer (`sim/mixer.py`, tested in `tests/test_mixer.py` and in the physics engine)
Numbering follows the **PX4 quad-X** convention so hardware outputs map one-to-one. Spin seen from above:

| Motor | Position (FRD x fwd, y right) | Spin | Effect of MORE thrust on this motor |
|---|---|---|---|
| 1 | front-right (+, +) | CCW | lift; roll **left** (τx −); nose **up** (τy +); body yaw **clockwise** (τz +) |
| 2 | rear-left (−, −) | CCW | lift; roll **right** (τx +); nose **down** (τy −); yaw **clockwise** (τz +) |
| 3 | front-left (+, −) | CW | lift; roll **right** (τx +); nose **up** (τy +); yaw **counter-clockwise** (τz −) |
| 4 | rear-right (−, +) | CW | lift; roll **left** (τx −); nose **down** (τy −); yaw **counter-clockwise** (τz −) |

`[T, τx, τy, τz] = A·[T₁..T₄]` with `τx = −Σ yᵢTᵢ` (positive = right wing down), `τy = ΣxᵢTᵢ` (positive = nose up),
`τz = Σ sᵢ(kQ/kT)Tᵢ` (positive = clockwise; `sᵢ = +1` CCW). Verified: +collective raises all four; +roll raises the **left** motors (2,3);
+pitch raises the **front** motors (1,3); +yaw raises the **CCW** motors (1,2); torque recomputed from `r×F` matches; and the same
patterns fed into PyBullet produce the right-signed rotation.

### Tuning (order: vertical → attitude → yaw → velocity → position; `python -m calibration.tune_controller`, logs in `logs/tuning/`)
Measured on this model: altitude step 2 m: 0.02 % overshoot, 2.2 s settling; roll/pitch 10° step: 7.5 % overshoot, 0.45 s settling; yaw 45°: 0.6 % overshoot, 0.75 s;
velocity 0→2 m/s: 2 % overshoot, 1.45 s; position 5 m: 0 % overshoot, 6.1 s settling; hover drift 4 mm, 0.07° RMS tilt; holding position in a constant 4 m/s wind: 0.19 m peak error, 7 mm final error, 3.2° steady lean (`logs/tuning/summary.json`).

---------------------------------------------------------------------------------------------------

## 6. MAVLink interface

MAVLink 2, `common` message definitions via pymavlink, UDP:
* FC simulator: `udpin:0.0.0.0:14540` (learns the companion's address from its first packet)
* companion: `udp:127.0.0.1:14540` — **translated to a client socket on an ephemeral local port** by `common/mav.py` because pymavlink's `udp:` prefix
  *listens* by default and its `udpout:` binds the destination port on Windows (both fail when the FC is on the same host).
  The companion must send its HEARTBEAT first; `CompanionMavlink.connect()` does.

| Companion → FC | Handling |
|---|---|
| `HEARTBEAT` | recorded |
| `SET_POSITION_TARGET_LOCAL_NED` | position / velocity / acceleration + yaw / yaw-rate per `type_mask` |
| `SET_ATTITUDE_TARGET` | attitude quaternion + normalised thrust |
| `COMMAND_LONG` | `MAV_CMD_COMPONENT_ARM_DISARM`, `MAV_CMD_DO_SET_MODE`, `MAV_CMD_NAV_LAND`, `MAV_CMD_SET_MESSAGE_INTERVAL`; others → `COMMAND_ACK` UNSUPPORTED |
| `SET_MODE` | base_mode + custom_mode |

| FC → companion | Rate (config) | Source |
|---|---|---|
| `HEARTBEAT` (type QUADROTOR, autopilot **GENERIC**, custom_mode with PX4's bit layout) | 2 Hz | FC state |
| `LOCAL_POSITION_NED` | 30 Hz | FC estimate |
| `ATTITUDE` | 50 Hz | FC estimate + gyro |
| `GLOBAL_POSITION_INT` | 10 Hz | FC estimate → lat/lon via `common/geo.py` |
| `SYS_STATUS` | 2 Hz | battery model |
| `COMMAND_ACK`, `STATUSTEXT` | on event | |
| `HIGHRES_IMU` | 100 Hz | simulated IMU (downsampled from 250 Hz) |
| `GPS_RAW_INT` | 10 Hz | simulated GPS |
| `RAW_RPM` (×4), `ACTUATOR_OUTPUT_STATUS` | 10 Hz | motor state / normalised commands |

Algorithm-facing API (`companion/mavlink_client.py`, NED, metres, radians): `send_velocity_ned(vn, ve, vd, yaw_rate)`,
`send_position_ned(north, east, down, yaw)`, `send_attitude_target(q, thrust)`, `arm()`, `disarm(force)`, `set_mode("OFFBOARD"|"HOLD"|"LAND")`,
`land()`, `telemetry()`, `gps()`, `pop_imu()`, `motor_rpm()`. Keep streaming setpoints ≥ 2 Hz (20 Hz typical).

---------------------------------------------------------------------------------------------------

## 7. ROS 2 interface (`ros2_bridge/uav_mavlink_bridge.py`)

`/uav/cmd_vel` (`geometry_msgs/Twist`) **frame is explicit** (parameter `cmd_frame`):

| `cmd_frame` | linear.x / y / z | angular.z |
|---|---|---|
| `enu` (default, REP-103) | East / North / **Up** m/s | yaw rate, **CCW positive** |
| `ned` | North / East / **Down** | yaw rate, CW positive |
| `body_flu` | forward / left / up, relative to current heading | CCW positive |

Validation: non-finite values rejected, speeds clamped (`max_speed_xy` 5, `max_speed_z` 2, `max_yaw_rate` 1.5). Then converted to NED and sent as
`SET_POSITION_TARGET_LOCAL_NED` (`MAV_FRAME_LOCAL_NED`, velocity + yaw-rate fields). The bridge streams at 20 Hz; when no command arrived for 0.5 s it keeps
streaming a **zero-velocity** setpoint (brake/hold). `/uav/cmd_pos` (`PoseStamped`, ENU) sends a position setpoint.

Published: `/uav/telemetry` (`std_msgs/String`, JSON — no custom message package needed), `/uav/odom` (`nav_msgs/Odometry`, `map`→`base_link`, ENU/FLU),
`/uav/imu` (`sensor_msgs/Imu`, FLU, 100 Hz), `/uav/gps/fix` (`NavSatFix`), `/uav/mode` (`String`), `/uav/armed` (`Bool`).
Services (`std_srvs`): `/uav/arm` (SetBool), `/uav/force_disarm`, `/uav/offboard` (SetBool), `/uav/land`, `/uav/setpoint_stream` (SetBool; `false` stops streaming to exercise the failsafe).

`/uav/telemetry` JSON: `stamp, fc_time_boot_ms, link{ok, heartbeat_age_s, telemetry_age_s}, failsafe, frame, position_enu, velocity_enu, orientation_xyzw_enu_flu, position_ned, velocity_ned,
roll_pitch_yaw_ned_frd, gps{lat,lon,alt}, mode, armed, system_status, battery_v, battery_pct, motor_rpm, setpoints_sent, cmd_rejected`.

RGB-D on ROS: `python -m sim.run_fc_sim --ros-sensors` publishes `/uav/camera/color/image_raw`, `/uav/camera/depth/image_raw` (32FC1 **metres**, NaN invalid),
`/uav/camera/{color,depth}/camera_info` (frame `rgbd_camera_optical_link`), stamped with simulation time. (Software rendering is ≈ 0.2–0.35 s per 640×480 frame here, so the
simulation runs slower than real time while streaming images; lower `sensors.camera.width/height/rate_hz` if needed.)

---------------------------------------------------------------------------------------------------

## 8. Sensors

**IMU** (`sim/sensors.py`, docstring is the normative spec): body FRD, 250 Hz. `gyro = ω + b_g + n_g`, `accel = R_bn(a_imu − g) + b_a + n_a` with the lever arm from the CG to the
Pixhawk (`a_imu = a_cg + α×r + ω×(ω×r)`); `a_cg, α` are finite differences of the engine's velocities. White noise from densities (gyro 5e-5 rad/s/√Hz, accel 7e-4 m/s²/√Hz),
turn-on bias (σ 0.002 rad/s, 0.02 m/s²), bias random walk (5e-6, 5e-5), ±2000 °/s / ±16 g clipping, timestamped. ICM-42688-class **datasheet-order placeholders**;
vibration, scale-factor, misalignment and temperature effects are not modelled (real flight noise will be higher — inflate for VIWO).

**GPS** (`Matek 3901-10X` class): lat/lon/alt, timestamp, fix type, HDOP, satellites, h/v accuracy at 10 Hz; position error = Gauss–Markov (σ_h 0.6 m, σ_v 1.0 m, τ 30 s), velocity noise 0.05 m/s;
`mode: healthy | degraded | denied` and `dropouts: [[t0, t1], ...]`. Baseline scenario: healthy. Antenna lever arm from the URDF.

**Geographic frame** (`common/geo.py`): map origin ↔ (lat 30.000000°, lon 76.000000°, alt 250 m) — a placeholder, edit `geo` in `drone.yaml`. Exact WGS-84 via ECEF (round-trip tested).
`lat, lon, alt = GeoFrame.from_config(cfg).ned_to_geodetic([N, E, D])`; the same class serves survivor geo-tagging on the companion side.

**RGB-D**: 640×480, **vertical FOV 58°** (PyBullet defines projection by vertical FOV) → hfov 72.9°, `fx = fy = 432.97`, `cx = 320`, `cy = 240`, near 0.2 m, far 12 m, depth noise σ = 0.001·z². Mounted on the body
through the URDF extrinsics (pitched 10° down); pose is derived from the physically simulated body each frame — never a separate teleported camera, never a SLAM pose. Recovered by simulated checkerboard calibration to
0.3 % (fx) and by marker back-projection to 8 mm through the URDF extrinsics.

---------------------------------------------------------------------------------------------------

## 9. Ground-truth firewall

```
NORMAL AUTONOMY (algo/, companion/, ros2_bridge/)         EVALUATION ONLY (eval/, tests/, logs/)
  RGB, depth, camera info, IMU, GPS, MAVLink telemetry      ground-truth pose/velocity/orientation, motor thrust,
                                                            contacts, survivor positions, paths
```
* `eval/ground_truth.py` writes `logs/<run>/ground_truth.csv`, `world_truth.json` (survivors) and streams JSON on UDP 14560.
* `tests/test_ground_truth_firewall.py` AST-scans `algo/ companion/ ros2_bridge/` and fails if they import `pybullet`, `sim`, `worlds` or `eval`, or mention survivor/world truth.
* Survivors exist only as PyBullet bodies (standing / sitting / lying, distinct shirt colours). Their coordinates appear in no MAVLink message, ROS topic or algorithm-side data class.
  Their future geo-tags must come from RGB detection + depth + intrinsics + estimated pose + GPS reference (`algo/interfaces.py: SurvivorNode`).

---------------------------------------------------------------------------------------------------

## 10. Tests and calibration

```bash
python scripts/run_all_tests.py --ros-python <path-to-ros-env-python>     # everything (≈ 15 min), summary in logs/validation/
pytest tests -m "not ros and not wire"                                     # fast subset
python scripts/comm_test.py [--no-world]                                   # 16-step ROS→MAVLink→FC→PyBullet test, PASS/FAIL report in logs/comm_test_*/
python scripts/diagnose.py [--live]                                        # environment / config / live-link diagnostics
```
`tests/`: `test_model_urdf` (mass, inertia incl. brute-force cross-check, geometry), `test_frames_geo`, `test_mixer`, `test_motor_battery`, `test_physics_fc` (gravity, hover, climb, descend,
forward, lateral, yaw, mixer-in-physics, motor lag, setpoint loss, disarm, landing), `test_mavlink_wire` (real UDP subprocess), `test_ros2` (needs rclpy), `test_rgbd`, `test_ground_truth_firewall`, `test_tuning`.

### Calibration workflow (all values land in `config/drone.yaml`)
| # | Tool | What you do with real hardware |
|---|---|---|
| 1, 2 | `python -m calibration.verify_mass_inertia` | weigh parts, edit `components`, `python -m tools.build_urdf`; check Izz etc. with a bifilar pendulum |
| 3 | `python -m calibration.fit_propulsion thrust bench.csv` (`rpm,thrust_n[,torque_nm]`) → `thrust_coeff_kT`, `torque_coeff_kQ`; `calibration.propulsion_report` shows hover command/rpm | thrust-stand sweep with the real prop |
| 4 | `python -m calibration.fit_propulsion step step.csv` (`t_s,rpm`) → `motor_time_constant_s` | ESC telemetry step 30 %→70 % |
| 5 | `propulsion_report` (yaw-torque recovery in PyBullet); with bench torque data `kQ/kT` comes from item 3 | torque sensor or yaw-rate flight logs |
| 6 | `propulsion_report` (max thrust / T/W vs voltage); `fit_propulsion voltage volt.csv` | full-throttle static thrust at several voltages |
| 7 | `propulsion_report` (sag, thrust margin, endurance vs SOC); `internal_resistance_ohm`, OCV table | discharge log |
| 8 | `python -m calibration.tune_controller` | re-run on the real vehicle log format later |
| 9 | `python -m calibration.camera_calibration` (checkerboard → OpenCV) | run on real RealSense images / use factory intrinsics |
| 10 | same script (marker back-projection through URDF extrinsics) | Kalibr / hand-eye camera–IMU calibration → `components.rgbd_camera_link` |

`python -m calibration.fit_propulsion selftest` proves the fitters recover known parameters from synthetic data.

---------------------------------------------------------------------------------------------------

## <a id="hardware-transition"></a>11. Hardware transition

```
Simulation:  algorithm → ROS 2 → MAVLink UDP  → simulated FC (PyBullet)
Hardware:    algorithm → ROS 2 → MAVLink serial → Pixhawk 6C Mini → ESCs → S3110 motors
```
1. Wire the Pi 5 to a Pixhawk 6C Mini TELEM port (UART, 3.3 V, common ground; see PX4 *Raspberry Pi companion* docs) and set the MAVLink instance on that port
   (PX4 `MAV_1_CONFIG` = the TELEM port, `MAV_1_MODE` = Onboard, `SER_TELx_BAUD` = 921600).
2. Run the bridge with `connection:=serial:/dev/serial0:921600`:
   `python -m ros2_bridge.uav_mavlink_bridge --ros-args -p connection:=serial:/dev/serial0:921600`. Nothing else changes.
3. Real PX4 sends its own HEARTBEAT autopilot id 12 and PX4 custom modes — `CompanionMavlink.set_mode()` already uses the PX4 bit layout. Request the streams the sim gives for free
   (`set_message_interval`, or PX4 `MAV_1_RATE` / mavlink `stream` commands): LOCAL_POSITION_NED, ATTITUDE, GLOBAL_POSITION_INT, SYS_STATUS, HIGHRES_IMU, GPS_RAW_INT.
4. PX4 Offboard rules: stream setpoints > 2 Hz *before* switching to Offboard (the simulator enforces the same), and set `COM_OF_LOSS_T` / `COM_OBL_RC_ACT` deliberately.
5. Fly with props off first, then tethered; keep the RC transmitter as manual override. The sim FC is a *stand-in*, its gains are not Pixhawk gains — tune on the airframe.
6. Cameras/IMU: replace the sim RGB-D publisher with the RealSense ROS driver (same topic names/frames); put measured extrinsics in `components.rgbd_camera_link`.

---------------------------------------------------------------------------------------------------

## 12. Known limitations

* Propulsion, battery resistance, drag, IMU noise, GPS error, component masses and the arm length are **placeholders**; nothing is validated against the physical S3110/propeller/battery.
* The FC is a simulation of a flight controller, **not PX4/ArduPilot**: no EKF, no sensor fusion, no real arming checks, no RC failsafe, simplified LAND, no gyroscopic rotor terms.
  Its state estimate is truth + small configured noise.
* Propellers have no collision geometry; no ground effect, wake, gusts/turbulence model (constant wind only); rigid airframe.
* PyBullet inertia uses the diagonal only (Ixz ≈ 0.45 g·m² neglected).
* Survivors are procedural humanoids (ellipsoid/capsule bodies, textured face) built to be recognised by the kit's COCO YOLOv8n; **standing and sitting** postures are detected reliably, a **lying** figure seen at a low oblique angle is *not* (the demo has one such survivor, S3, off the route, by design). They are not photo-realistic.
* **SLAM** is ORB-SLAM3 RGB-D on Linux (no IMU coupling; the demo route has no loop closure event, so drift is uncorrected: 0.3–0.4 % of path in the demo). ATE varies run to run (multi-threaded ORB-SLAM3): 0.06–0.11 m observed.
  **ORB-SLAM3 does not run on Windows**; Windows uses the `orb_pnp_vo` fallback, which has no loop closure and drifts noticeably more (ATE ≈ 0.3–0.7 m on the search
  flight, more on the longer search + return mission). The navigation altitude/position come from SLAM only, so this drift directly offsets where the UAV
  thinks it is (e.g. where it lands on return).
* Both SLAM backends assume a static scene. The `dynamic` scenario was added to exercise this: the VO fallback rejects physically impossible pose jumps and
  never keys poorly tracked frames, but a large moving object in view can still degrade tracking. ORB-SLAM3 in the dynamic scenario was run on Linux: see [docs/FINAL_ACCEPTANCE_LINUX.md](docs/FINAL_ACCEPTANCE_LINUX.md) (not reliably collision-free, survivor S2 missed, ATE 0.29–0.90 m).
* The mission holds altitude from the SLAM estimate (the FC's altitude is not used by navigation by design: in this simulator it is truth + small noise).
  On hardware the FC's own altitude estimate (barometer / EKF) would be the safer altitude reference.
* The thermal camera is **synthetic**: object-table temperatures, no radiometry, no atmospheric/emissivity effects; it is not physically calibrated.
* Perception rate is 5 Hz (software rendering); the simulation runs at ≈ 0.3–0.5× real time in the demo. Everything is in simulation time (the FC watchdog, the image stamps and the MAVLink timestamps share one clock).
* Occupancy/planning uses a 2-D altitude band of a 3-D voxel map at a fixed flight altitude (2 m); obstacles below the band (rubble under ≈ 1.5 m) do not affect planning. Only one hazard class (fire / hot debris) is implemented; the node structure supports more.
* The survivor tracker merges any two tracks closer than 2.5 m, so two real survivors that close would be reported as one.
* GPS geo-tags inherit the GPS anchor error (σ ≈ 0.6 m, correlated) — relative survivor positions are much more accurate than absolute ones.
* Software rendering (`ER_TINY_RENDERER`) is slow for 640×480 at 15 Hz; consider smaller frames or `realtime: false` when logging datasets (`scripts/record_rgbd.py`).
* MAVLink UDP has no packet authentication/signing; single companion peer assumed.
* IMU telemetry is downsampled to 100 Hz on the wire (250 Hz internally). For VIWO run `set_message_interval(105, 250)` or set `mavlink.rates_hz.highres_imu: 250`.

---------------------------------------------------------------------------------------------------

## 13. Validation status and what still needs Linux

Full results of the Windows validation: [docs/WINDOWS_LINUX_STATUS.md](docs/WINDOWS_LINUX_STATUS.md).

Runs on **Windows** (validated here): every pytest suite except the ORB-SLAM3 one, the 16-step ROS 2 → MAVLink → FC → PyBullet communication test,
the in-process closed-loop missions (static and dynamic) and the full ROS 2 + MAVLink autonomous mission with the `orb_pnp_vo` SLAM fallback.

Needs **Ubuntu 22.04 + ROS 2 Humble** (ORB-SLAM3 is built only there):
1. `bash slam/build_orbslam3.sh` (once), then `python3 -m pytest tests/test_orbslam3.py tests/test_closed_loop.py -k "orbslam3 or slam_flight"` — the ORB-SLAM3 runner and the
   SLAM-flight test with ORB-SLAM3 (skipped on Windows).
2. `python3 scripts/run_disaster_demo.py` and `python3 scripts/run_disaster_demo.py --scenario dynamic --no-video` — the final integrated mission with **live ORB-SLAM3**
   (`--slam-backend auto` selects it). Check `demo_summary.json → slam_backend == "orbslam3"`, `score_report.json` and `rescue_report.md`.
3. `python3 scripts/run_all_tests.py --ros-python $(which python3)` — the whole validation, including `tests/test_ros_demo.py` (ROS 2 + MAVLink mission test).
4. Done on Linux (2026-10-04/05): static accepted; dynamic only partially accepted. Results, limitations and exact commands: [docs/FINAL_ACCEPTANCE_LINUX.md](docs/FINAL_ACCEPTANCE_LINUX.md).

---------------------------------------------------------------------------------------------------

## 14. References
* PX4 Offboard control — https://docs.px4.io/main/en/flight_modes/offboard
* PX4 ROS 2 offboard — https://docs.px4.io/main/en/ros/offboard_control
* PX4 Raspberry Pi + Pixhawk — https://docs.px4.io/main/en/companion_computer/pixhawk_rpi
* PX4 companion peripherals — https://docs.px4.io/main/en/companion_computer/companion_computer_peripherals
* MAVLink common messages — https://mavlink.io/en/messages/common.html
* Pixhawk 6C Mini spec — https://docs.holybro.com/autopilot/pixhawk-6c-mini/technical-specification

No PX4 source is used or copied; only the public message definitions and documented conventions (quad-X motor order, custom-mode layout, Offboard streaming rule).
