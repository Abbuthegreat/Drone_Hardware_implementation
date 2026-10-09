# Windows / Linux status and validation report (2026-10-04)

Machine: Windows 11 (10.0.26200), 16 cores, 15 GB RAM. Environment: micromamba `uavros`: Python 3.11.16, ROS 2 Humble (RoboStack), pybullet
(conda-forge, Oct 21 2025), pymavlink 2.4.50, onnxruntime 1.30 (CPU), pytest 9.1.1. No WSL. ORB-SLAM3 / Pangolin not built on this machine.
The Linux baseline (Ubuntu 22.04 + ROS 2 Humble + ORB-SLAM3) is described in [SLAM_INTEGRATION_UBUNTU.md](SLAM_INTEGRATION_UBUNTU.md); none of it was changed.

## 1. What was found (repository inspection)

All the listed components existed and were connected: PyBullet vehicle + motors + mixer + simulated FC, MAVLink server/client, ROS 2 ↔ MAVLink bridge,
RGB-D / synthetic thermal / IMU / GPS simulation, SLAM front-end with two backends (ORB-SLAM3 runner for Linux, `orb_pnp_vo` fallback), kit YOLO survivor
detection, depth localisation, thermal fusion, geo-semantic map, survivor tracker + geotagging, A\*, VFH+, a mission node and a one-command ROS demo.
Genuinely unfinished or broken:

| Item | Finding |
|---|---|
| Windows run of the demo | `slam.backend: orbslam3` (the validated Linux default) made the SLAM node crash on Windows (`FileNotFoundError`) |
| No-A\*-route behaviour | with no route the mission fell back to steering at the raw goal (through unknown space, VFH+ only) |
| Failsafe coverage | no reaction to: FC failsafe / disarm in flight, MAVLink link loss, stale map, obstacle too close, VFH+ without opening, permanent localisation loss (hovered forever) |
| Mission end | hover only; no return-to-home and no LAND |
| Command-centre output | per-node logs only; no combined status stream / rescue report |
| Dynamic environment | not implemented |
| SLAM flight test on Windows | hard-wired to ORB-SLAM3 → failed on Windows |
| Ubuntu validation `20261002_144737` | 2 failures: full in-process mission ATE 0.568 > 0.5 (VO run-to-run variability), comm-test step 14 (vehicle pushed into the debris field by the timing-dependent `ros2 topic pub` step, touched down on rubble at 1.19 m; the 3 repeat runs passed) |

## 2. What was changed

* **Mission safety layer** (`mission/disaster_demo.py`): states `LOST_HOLD → LANDING`, `MAP_STALE`, `LINK_LOST`, `FC_NOT_OFFBOARD`, `NO_ROUTE_HOLD` (+ waypoint skip),
  `VFH_BLOCKED`, `AVOID_RETREAT` (direction from the map), `LANDING` on FC failsafe, `ABORTED` on disarm in flight, command sanitising, `RETURN` home, `LANDING` → `DONE`
  at the end; mission events log. Never flies into unknown space.
* **VFH+** returns the bearing of the nearest obstacle (used by the retreat).
* **Bridge**: `/uav/telemetry` carries `link{ok, heartbeat_age_s, telemetry_age_s}` and `failsafe`; link health from the 30–50 Hz streams (a wall-clock heartbeat age gave
  false alarms because the simulator runs at ≈ 0.2–0.5× real time — found in a live run).
* **Nav node**: link check in simulation time (FC time of the telemetry vs the camera clock), map age, `/uav/land`, events, pose + A\* path in the status.
* **Command centre** (`mission/command_centre.py`): `command_centre/status.jsonl` (1 Hz), `[cc]` console line, `rescue_report.{json,md}`.
* **SLAM backend selection**: `--slam-backend auto|orbslam3|orb_pnp_vo` (`slam/orbslam3_backend.py: resolve_backend`), default config unchanged (`orbslam3`).
* **VO fallback robustness** (`slam/vo_backend.py`, ORB-SLAM3 untouched): rejects physically impossible pose jumps, discards keyframes contaminated by a moving object,
  re-initialises at the last accepted pose after a 3 s hover. Found by the dynamic scenario: a collapsing slab in view made the VO pose jump +1.7 m, the altitude
  hold descended and the UAV ended on the ground (no collision counted — the rig now counts unplanned touchdowns too).
* **Dynamic scenario** (`worlds/dynamic_obstacles.py`, `--scenario dynamic`): aftershock slab collapse + sliding debris; world events logged for evaluation.
* **Evaluation**: rig reports landing, unplanned touchdowns, world events; scoring reports unplanned touchdowns, landed/disarmed, MAVLink counts from the FC log if the
  summary is missing; the demo launcher lets the simulator stop cleanly.
* Tests: `test_mission_safety.py` (11), `test_dynamic_world.py` (3), SLAM-flight test parametrised by backend, dynamic closed-loop mission test, stricter end-of-mission
  checks in the full-mission and ROS-demo tests. No threshold was relaxed.

## 3. Test results on Windows

| Suite | Result | Notes |
|---|---|---|
| Baseline before changes (`pytest -m "not ros"`) | 108 passed, 1 skipped, **1 failed** | failure = SLAM-flight test hard-wired to ORB-SLAM3 (**Windows-only dependency**); skip = ORB-SLAM3 binary test (Linux-only) |
| Fast suite after changes (`-m "not ros and not slow"`) | **119 passed** | includes mission-safety, dynamic-world, planning, firewall, MAVLink wire, physics, mixer, RGB-D, scene |
| Closed loop, each test in its own process | **5 passed, 1 skipped** | SLAM flight (VO), survivor pipeline, navigation around wall + pole, full mission (search → land), dynamic mission (search → collapse → return → land); ORB-SLAM3 flight skipped (Linux-only) |
| 16-step ROS 2 → MAVLink → FC → PyBullet comm test | **16/16 PASS** | `logs/windows/comm_test/` |
| Full ROS 2 + MAVLink autonomous mission, static (`logs/demo/windows_static/`) | **mission complete, landed, disarmed** | see §4; `tests/test_ros_demo.py` assertions: all met **except SLAM ATE (0.98 m vs < 0.8 m)** with the VO fallback |
| Full ROS 2 + MAVLink mission, dynamic (`logs/demo/windows_dynamic/`) | **completed + landed, but FAIL on quality**: 1 collision episode (7 s), SLAM ATE 3.9 m, 1 duplicate survivor | see §4 — VO fallback is not adequate here; needs ORB-SLAM3 (Linux) |

Failures / problems seen during the work, classified:
* PyBullet native crash (`access violation` in `getCameraImage`, `Fatal Python error: Aborted` in `changeVisualShape`) twice while several heavy simulations ran in one
  process / under memory pressure (free virtual memory ≈ 7.5 GB): **Windows-specific flakiness**, not reproducible in fresh processes; run the slow tests one process each.
* Running two perception pipelines at once (tests + demo) starved the perception node (map > 3 s old → `MAP_STALE` hover): **timing**; run the demo alone.
* Regressions caused and fixed during this work: navigation test inheriting the new return/land defaults; impossible collapse trigger spec; nav-side link check
  in wall-clock time; a VO keyframe rule that broke tracking near walls (reverted).

## 4. Full-mission results (ground truth used offline only)

### Static, ROS 2 + MAVLink, VO fallback (`logs/demo/windows_static/`, video `disaster_demo.mp4`)
| | |
|---|---|
| Mission | TAKEOFF → WP1 (around wall + pole) → survivor #1 → WP2 → survivor #2 → WP3 → WP4 → RETURN → HOLD → LAND → DONE (FC auto-disarm) |
| Duration | 176 s simulated, 534 s wall (with video) |
| Survivors | 2/2 reachable found (S1 0.43 m local / 0.43 m geo error, 141 obs, 125 with thermal support; S2 0.67 m / 0.82 m, 45 obs, 32 thermal); 0 false/duplicate confirmed; 1 unconfirmed single-observation candidate (13 m off; never confirmed); S3 (lying, off route) not found |
| Hazards | fire F1 confirmed (1.3 m error) **plus a duplicate fire node 4.7 m away**, created on the return leg when the VO had drifted ≈ 2 m |
| Collisions / unplanned touchdowns | 0 / 0 |
| A\* re-plans / VFH+ active ticks / max deviation | 169 / 932 / 33.5° |
| Safety events | 2 × `AVOID_RETREAT` (obstacle at 0.70 m); no localisation loss, no link loss, no FC failsafe |
| MAVLink | FC received 10 597 `SET_POSITION_TARGET_LOCAL_NED` + 24 `COMMAND_LONG` (arm / mode / land); link OK throughout |
| SLAM (VO) | 879 poses, 100 % tracked, ATE RMSE 0.98 m, final error 1.94 m over 72.6 m (2.7 %); landed 1.8 m from the take-off point because of that drift |

### Dynamic, ROS 2 + MAVLink, VO fallback (`logs/demo/windows_dynamic/`, no video)
| | |
|---|---|
| Mission | TAKEOFF → WP1 → survivor #1 → WP2 → survivor #2 → WP3 → **collapse at 78.1 s** (jump rejected → LOST_HOLD 3 s at altitude → VO re-initialised → MAP_STALE 1 s) → around the slab → WP4 → RETURN → LAND → DONE (disarmed) |
| Duration | 227 s simulated, 1064 s wall |
| Survivors | S1 0.22 m, S2 1.07 m (geo 1.21 m); **1 duplicate confirmed node** (#4, 2.6 m from S1, created on the return leg while the VO was ≈ 2 m off) |
| Hazards | fire F1 confirmed, 3.7 m error |
| **Collisions** | **1 episode, 131.9–138.9 s**: pressed against the broken column (world 1.5, 21.0) at 1.5 m altitude; the VO estimate was 2.2 m off, so the column was mis-placed in the map. 0 unplanned touchdowns |
| SLAM (VO) | ATE RMSE **3.9 m**, final error **10.2 m** (12.6 % of 80.6 m) → landed 9.8 m from the take-off point |
| Safety events | LOST_HOLD 1, MAP_STALE 1, AVOID_RETREAT 3 episodes (254 ticks), **LINK_LOST 3 short episodes (2.1 s total) while the bridge reported the link OK** — FC-time lag of the relayed telemetry under heavy load; cause not fully diagnosed |
| MAVLink | FC received 21 113 `SET_POSITION_TARGET_LOCAL_NED`, 1 050 `HEARTBEAT`, 29 `COMMAND_LONG` |

**Verdict:** the integrated chain works end to end in the dynamic scene (perceive → re-plan → avoid → return → land over MAVLink), but with the VO fallback the
localisation is not good enough for it to be safe. This run must be repeated with ORB-SLAM3 on Linux.

### Dynamic, in-process rig (same autonomy classes, no ROS/UDP)
Collapse triggered at 75.5 s directly ahead of the UAV after WP3 → impossible VO jump rejected → 3 s hover → VO re-initialised at the last good pose → WP4 →
return → land → disarmed. 0 collisions, 0 unplanned touchdowns, both survivors (0.25 m / 0.85 m), fire confirmed, 225 s simulated; SLAM ATE 1.67 m.

## 5. Commands

Windows: see README §1.1. Linux: README §1.2 and §13.

## 6. What still requires Linux

1. `bash slam/build_orbslam3.sh`; `pytest tests/test_orbslam3.py tests/test_closed_loop.py -k "orbslam3 or slam_flight"`.
2. The final integrated mission with **live ORB-SLAM3**: `python3 scripts/run_disaster_demo.py` and `--scenario dynamic --no-video`.
   The ROS-demo test's SLAM threshold (ATE < 0.8 m) is expected to be met there (ORB-SLAM3 ATE 0.06–0.11 m was measured before), and the larger VO drift seen on Windows
   (duplicate hazard, 1.8 m landing offset) should shrink; this has to be confirmed by running it.
3. `python3 scripts/run_all_tests.py --ros-python $(which python3)`.
4. ORB-SLAM3 has never been run in the dynamic scenario.
