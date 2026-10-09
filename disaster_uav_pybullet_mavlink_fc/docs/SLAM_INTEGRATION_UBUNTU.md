# ORB-SLAM3 integration on Ubuntu 22.04 (ROS 2 Humble)

## 1. Kit inventory (`slam_kit_3/slam_kit/`, read completely, unmodified)
| File | What it is |
|---|---|
| `setup_orbslam3.sh` | clones **UZ-SLAMLab/ORB_SLAM3 (HEAD)** + Pangolin v0.8, applies the C++14/OpenCV patches (see PATCHES.md), builds `rgbd_offline`. **ORB-SLAM3 source is not inside the zip.** |
| `slam/rgbd_offline.cc` | ORB-SLAM3 **RGB-D** runner: replays `associations.txt` + `rgb/*.jpg` + `depth/*.png`, writes `CameraTrajectory.txt` (EuRoC style). Depth PNG = uint16 mm |
| `record.py` | RealSense D435i/D455 recorder. `write_orbslam_yaml()` writes the per-flight YAML: PinHole, `Camera.RGB: 0` (BGR), `RGBD.DepthMapFactor = 1/depth_scale (1000)`, `Stereo.ThDepth 40`, 1500 features. IMU is only used to measure gravity 'up' at start |
| `postprocess.py` | trajectory + YOLO -> `localize()` (torso-median depth) -> clustering -> top-down map / PLY / HTML |
| `detector.py`, `models/yolov8n.onnx` (+ `.pt`) | YOLOv8n person detector (ultralytics / onnxruntime / OpenCV-DNN backends) |
| `process.sh`, `start_rec.sh`, `stop_rec.sh` | glue scripts |
Not in the kit: ROS wrapper, launch files, IMU-coupled SLAM (the physical system is **RGB-D only**; no VIO), thermal. -> **No ROS wrapper to adapt**: the ROS interface is this repo's `ros2_bridge/slam_node.py`, which feeds the kit-derived `slam/rgbd_stream` process (the external wrappers akchobby / gjcliff / sagar16812 were only conceptual references; **no code was copied from them**).

## 2. Data flow actually running
`PyBullet RGB-D (5 Hz, sim time) -> /uav/camera/{color,depth}/image_raw -> slam_node (exact-stamp RGB/depth pairing) -> slam/adapter.py -> slam/orbslam3_backend.py -> rgbd_stream (ORB-SLAM3 RGB-D) -> Twc -> gravity/heading-aligned UAV pose -> /uav/slam/pose, /uav/slam/path, /uav/slam/status, TF slam_map->base_link`.
The SLAM node subscribes only to RGB, depth, camera_info, `/uav/imu`, `/uav/telemetry` (FC heading at start). It cannot see ground truth (AST test + `tools/diagnose_slam.py` FIREWALL check). IMU is used for gravity alignment only (same as the kit), not fed into ORB-SLAM3.
GPS is not an input to SLAM; it only anchors the finished map (`mapping/geo_anchor.py`).

## 3. Calibration, depth, frames (numerically verified: `python tools/make_orbslam_yaml.py`, tests/test_orbslam3.py)
* `config/orbslam3/pybullet_rgbd.yaml` (new; physical calibration untouched), derived from the simulated camera: 640x480, vfov 58 deg, fx=fy=432.97, cx=320, cy=240, k=0. Max |sim - YAML| = 2.7e-7 px. (The backend regenerates the same numbers from the live `CameraInfo` at runtime.)
* Depth: simulation = float32 **metres** (`32FC1`, NaN/0 invalid, range 0.2–12 m) -> adapter converts to **uint16 mm** (`x1000`) -> YAML `RGBD.DepthMapFactor: 1000`. Round-trip error 3e-8 m. The global camera topics are unchanged.
* Frames: `slam_map` (ENU, origin = take-off) -> `base_link` (FLU, URDF origin; the SLAM pose is the CG pose, shifted by the URDF `cg_in_base_link`) -> `rgbd_camera_link` (x fwd, pitched 10 deg down, t = [0.19, 0, -0.005]) -> `rgbd_camera_optical_link` (rpy -pi/2, 0, -pi/2: z fwd, x right, y down = ORB-SLAM3 camera frame). Verified: optical z = (cos10, 0, -sin10) in base_link, x = -y_base, y ~ -z_base; URDF rotation == `config sensor_extrinsics` rotation (1e-5). Transforms are broadcast once, by `slam_node` (`ros2_bridge/tf_util.py`, parsed from the URDF). `docs/tf/frames_*.pdf` is `ros2 run tf2_tools view_frames`; `diagnose_slam.py` checks exactly 1 publisher on /tf and /tf_static.
* Time: all sensor stamps are FC/simulation time (the sim may run at rtf 0.5); RGB and depth of one capture carry the identical stamp and are paired by exact stamp (no approximate matching).
* Pangolin viewer: built in, `rgbd_stream <voc> <yaml> 1` enables it (headless runs use 0).

## 4. Run
```bash
# once (Ubuntu 22.04, ROS 2 Humble already installed)
sudo apt install build-essential cmake git pkg-config libeigen3-dev libglew-dev libepoxy-dev libgl1-mesa-dev libboost-all-dev libssl-dev libopencv-dev ros-humble-ros-base ros-humble-tf2-tools
pip install --user pybullet pymavlink onnxruntime       # system python3.10; no conda env needed (rclpy comes from /opt/ros/humble)
bash slam/build_orbslam3.sh                             # clones + builds ORB-SLAM3 into ~/slam_deps (Pangolin 0.6 must be installed; script builds v0.8 if missing)
# every time
source /opt/ros/humble/setup.bash
python scripts/test_slam_pipeline.py                    # SLAM-only validation flight (no autonomy)
python scripts/run_disaster_demo.py                     # full mission + video (config slam.backend: orbslam3)
ROS_DOMAIN_ID=88 python tools/diagnose_slam.py          # while the demo runs
```
Note: Pangolin 0.8 (what the kit script would build) is untested here; 0.6 was already installed and works.
