# PATCHES.md — every modification made to third-party code / kit scripts

ORB-SLAM3 (GPLv3, © UZ-SLAMLab; licence + copyright headers untouched) is **not vendored** in this repo (the attached kit ships only its build
script and runner). It is cloned by `slam/build_orbslam3.sh` into `~/slam_deps/ORB_SLAM3`, upstream commit
`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` ("Update Dependencies.md", 2022-02-10, the v1.0 line), and patched exactly as the kit's
`setup_orbslam3.sh` does. Nothing in ORB-SLAM3's algorithms was changed.

Build environment: Ubuntu 22.04.5, GCC 11.4.0, CMake 3.22.1, OpenCV 4.5.4 (apt `libopencv-dev`), Eigen 3.4.0 (apt), Boost 1.74, Pangolin **0.6** (pre-installed in /usr/local; the version ORB-SLAM3 was developed against).
The build succeeded on the first attempt; no compile errors, only Eigen deprecation warnings.

| # | File | Change | Why |
|---|---|---|---|
| 1 | ORB_SLAM3 `CMakeLists.txt` | `-std=c++11` -> `-std=c++14` (also DBoW2/g2o CMake, via the same `sed`) | kit script; GCC 11 + Pangolin; all parts must use the same C++ standard (Eigen alignment) |
| 2 | ORB_SLAM3 `CMakeLists.txt` | `find_package(OpenCV 4.4)` -> `find_package(OpenCV 4 REQUIRED)` | kit script; accepts Ubuntu 22.04's OpenCV 4.5.4 |
| 3 | ORB_SLAM3 `include/LoopClosing.h` | `std::pair<const KeyFrame*, g2o::Sim3>` -> `std::pair<KeyFrame* const, g2o::Sim3>` | kit script; no-op on this checkout (already fixed upstream) |
| 4 | ORB_SLAM3 `CMakeLists.txt` | appended `rgbd_offline` (the kit's runner) and `rgbd_stream` (ours) targets | kit script + streaming runner |
| 5 | `slam/build_orbslam3.sh` (adapted from kit `setup_orbslam3.sh`) | apt step only installs packages that are actually missing (`dpkg -s` check) | no sudo needed when dependencies already exist |
| 6 | `slam/rgbd_stream.cc` (adapted from kit `slam/rgbd_offline.cc`) | pose protocol moved to a private dup of fd 1; fd 1 redirected to stderr | ORB-SLAM3 prints banners to stdout, which corrupted the `READY`/`POSE` protocol (first integration attempt failed on exactly this) |
| 6b | `slam/rgbd_stream.cc` (final acceptance session) | additionally prints the Atlas **map id** of the pose (`mp->GetMap()->GetId()` of a tracked map point, public API) as the 10th field | after a tracking loss ORB-SLAM3 may continue in a NEW map with its own origin; without the id the adapter mixed two frames (measured: 19 m ATE, mission stuck in RETURN) |
| 7 | `scripts/comm_test.py` | `ros2 topic pub` uses `ros2` on Linux (Windows `ros2-script.py` path only if it exists) | the test could never publish on Linux (step 11 FAIL) |

Kit files `slam_kit_3/slam_kit/*` are unmodified.

## Final Linux acceptance session (2026-10-04): changes to this repository's own code (not third-party)
| File | Change | Why (measured) |
|---|---|---|
| `slam/adapter.py` | one frame alignment per ORB-SLAM3 map id; a new map is anchored at the last accepted pose (the mission hovers while localisation is lost); a physically impossible jump inside one map id is treated the same way | collapse at ~72 s -> ORB-SLAM3 lost tracking 3.4 s and continued in a new map: pose jumped back to the start (ATE 19 m, 2 of 3 runs never finished) |
| `slam/orbslam3_backend.py` | detects a dead runner (EOF / non-zero exit), respawns it, returns LOST for that frame, new map id = id + 1000 x restarts | ORB-SLAM3 aborted once with `Sophus ensure failed ... SO3::exp failed! omega: nan` (upstream numerical failure) and the exception killed the run |
| `perception/front_clearance.py`, `mapping/pipeline.py`, `ros2_bridge/{perception_mapping,nav}_node.py`, `mission/disaster_demo.py`, `config nav.front_guard` | sensor-frame guard: 5th-percentile depth in the vehicle's swept corridor ahead (camera frame, no SLAM pose) -> no forward motion below 1.0 m, back away below 0.75 m | a moving obstacle can offset the SLAM-frame map by ~1 m; the map-based 0.7 m check fired only at contact |
| `mapping/metric_map.py`, `planning/astar.py`, `mission/disaster_demo.py`, `config map.dynamic_memory`, `astar.dynamic_mode` | moving-obstacle memory: a confidently occupied cell later observed free keeps its neighbourhood blocked for A* (grid value 90; VFH+ ignores it) for 90 s | the sliding debris sweeps in from the side, outside the forward camera FOV; in-process dynamic runs collided in 4 of 5 runs without it and 0 of 6 with it (see docs/FINAL_ACCEPTANCE_LINUX.md) |
| `eval/rigs/inprocess_rig.py` | uses the configured SLAM backend (`UAV_RIG_SLAM` override) instead of a hard-wired VO; writes `trace.csv` | the in-process tests had been measuring the VO fallback, even on Linux |
| `eval/rigs/run_inprocess.py` | new CLI for the in-process rig that keeps its outputs | analysis |
