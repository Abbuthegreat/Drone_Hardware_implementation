# Final Linux acceptance (Ubuntu 22.04.5, ROS 2 Humble, Python 3.10.12, live ORB-SLAM3) — 2026-10-04/05

Evidence: `logs/final/` (every run's `demo_summary.json`, `score_report.json`, `rescue_report.md`, nav/slam/perception logs), `logs/validation/<ts>/`.
All ground truth is used offline only (`eval/`); the SLAM backend of every mission below was `orbslam3` (forced with `--slam-backend orbslam3`; `demo_summary.json → slam_backend`).

## Verdict
| Requirement | Static | Dynamic |
|---|---|---|
| Live ORB-SLAM3, estimated pose drives navigation, no GT in the loop | ✅ | ✅ |
| Takeoff → search → return home → LAND → FC disarm, over real MAVLink | ✅ | ✅ (2/2 runs) |
| 0 collisions, 0 unplanned touchdowns | ✅ | ✅ 2/2 ROS runs ⚠ see below |
| Both reachable survivors confirmed + geotagged | ✅ S1, S2 | ❌ **S1 only in 2/2 ROS runs** (S2 missed) |
| Fire hazard confirmed | ✅ | ✅ |
| ATE < 0.8 m (existing threshold, unchanged) | ✅ 0.42 m in the acceptance run, but ❌ **1.36 m in the `tests/test_ros_demo.py` run** | ✅ 0.29 m / ❌ **0.90 m** |
| Dynamic obstacle perceived, route adapts | – | ✅ (A* re-plans 234/280, VFH+ up to 74°/138°, front guard) |
**Not fully accepted.** Static: complete missions in all runs, but ORB-SLAM3 ATE varies run to run and exceeded the 0.8 m threshold in one of the three static ROS runs on the final code (the ROS mission pytest FAILED on exactly that assertion). Dynamic: not reliably collision-free, survivor S2 missed, ATE up to 0.90 m. Details and the measured trade-offs below.

## Final ROS 2 + MAVLink missions (final code, nothing else running)
| | static | dynamic #1 | dynamic #2 |
|---|---|---|---|
| end state / reason | DONE / mission complete | DONE / mission complete | DONE / mission complete |
| sim / wall time | 169 s / 345 s | 223 s / 452 s | 226 s / 457 s |
| collisions / unplanned touchdowns | 0 / 0 | 0 / 0 | 0 / 0 |
| landed, FC disarmed | yes | yes | yes |
| landing displacement from take-off (GT, offline) | 1.06 m | 0.76 m | 1.94 m |
| SLAM poses / ATE RMSE / max / final / drift | 833 / 0.42 / 1.17 / 0.56 m / 0.78 % | 1103 / 0.29 / 1.13 / 1.13 m / 1.38 % | 1102 / **0.90** / 2.24 / 2.24 m / 2.78 % |
| survivors (local err, geo err, obs, thermal obs) | S1 0.16/0.28 m 121/100; S2 0.14/0.07 m 46/42 | S1 0.04/0.16 m 183/119 | S1 0.08/0.12 m 135/88 |
| duplicate survivor nodes | 0 | 0 | 0 |
| fire hazard (error) | confirmed (0.82 m) | confirmed (0.94 m) | confirmed (1.17 m) |
| MAVLink setpoints / COMMAND_LONG at the FC | 6836 / 16 | 8984 / 20 | 9082 / 17 |
| A* re-plans / VFH+ ticks | 168 / 493 | 234 / 623 | 280 / 643 |
| localisation-loss episodes (ORB-SLAM3) | 0 | 2 (recovered) | 2 (recovered) |
| front-guard hold / retreat ticks | 0 / 0 | 8 / 36 | 15 / 32 |
Command centre: `command_centre/status.jsonl` 338 / 444 / 448 lines (~1 Hz), `rescue_report.{json,md}` generated from each mission (survivor lat/lon/alt, local xyz, uncertainty, observations, thermal observations, hazards, event list).
Videos: `logs/final/static/disaster_demo.mp4` (earlier static ORB-SLAM3 run, 116 MB) and `logs/final/ros_static_final/disaster_demo.mp4`.

## What the dynamic scene did to ORB-SLAM3 (measured, not assumed)
1. **Frame jump after a tracking loss (genuine bug, fixed).** The slab collapse (~72–79 s) made ORB-SLAM3 lose tracking for ~3.4 s; it then continued in a *new map* with a new origin, so our pose jumped back to the start: ATE 19 m, 2 of 3 in-process runs never finished. Fix: map-id-aware alignment in `slam/adapter.py` (new map anchored at the last accepted pose; the mission hovers meanwhile).
2. **ORB-SLAM3 abort (genuine bug, fixed).** One run: `Sophus ensure failed … SO3::exp failed! omega: nan` killed the runner and our exception ended the run. The backend now respawns the runner (new map id).
3. **Moving debris from the side (limitation, partly mitigated).** The sliding debris sweeps across the return corridor outside the forward camera's field of view while the map still holds its last-seen position. Without mitigation 4 of 5 in-process dynamic runs collided at t≈137 s at the same place. A sensor-frame front guard alone did not prevent it (the debris comes from the flank). A moving-obstacle memory (occupied → later observed free ⇒ region blocked for A*) gave 0 collisions in 6/6 in-process runs and in both final ROS runs, **but** its blocked region can strand the mission: 2 of 8 in-process runs skipped waypoints (one skipped RETURN home and landed elsewhere), and one early ROS run hovered until the simulation-time limit. A "wait then cross" safeguard (`nav.dynamic_wait_s`) re-introduced collisions (2 of 3) and is therefore **off by default**.
4. **Residual ORB-SLAM3 drift with the moving debris in view**: 1–2 m error over ~10 s in some runs (ATE 0.29–1.1 m across runs); one final ROS run exceeds the 0.8 m threshold (0.90 m). ORB-SLAM3 is multi-threaded and not bit-reproducible: static ATE ranged 0.12–0.42 m across ROS runs of the same code.
5. Survivor S2 sits beside WP3, which the UAV reaches ~12 s later in the dynamic scene; it was confirmed with only 7 observations in the one pre-change dynamic run and not at all in the 2 final runs.

## Windows vs Linux (same mission code)
| | Windows (VO fallback) | Linux (ORB-SLAM3) |
|---|---|---|
| static ATE | 0.98 m (FAIL vs 0.8) | 0.12–0.42 m |
| static landing displacement | 1.8 m | 1.06 m (0.53 m in an earlier run) |
| static duplicate hazard | yes | none |
| dynamic ATE | 3.9 m, final 10.2 m | 0.29 m / 0.90 m, final 1.1 / 2.2 m |
| dynamic collisions | 1 episode (7 s against a column) | 0 in both final ROS runs (but see in-process results above) |
| dynamic landing displacement | 9.8 m | 0.76 m / 1.94 m |
| dynamic duplicate survivor | yes | none |
| dynamic S2 | found | missed (2/2) |
Linux + ORB-SLAM3 is clearly better at localisation, landing accuracy and duplicate-node behaviour; it is **not** better at dynamic survivor coverage, and dynamic collision-freedom depends on the new mitigations.

## Static ATE variance (why the static ROS test failed once)
Static ROS missions with live ORB-SLAM3, ATE RMSE: before this session's changes 0.06, 0.12, 0.09 m (dynamic-scene run b); final code with the moving-obstacle memory ON: 0.42 m (acceptance run) and **1.36 m** (`test_ros_demo.py`, FAIL vs 0.8); A/B with the memory OFF: 0.21 m and 0.69 m. In the 1.36 m run neither the front guard nor the crossing logic acted; the error stepped up to ~1 m at t≈20–30 s while the UAV crept past the wall at ~0.8 m clearance, then grew steadily (no new map, one 0.2 s loss at 106 s). With 2 runs per arm and this spread I cannot prove the memory caused the higher errors; the data lean that way, so the memory's effect on the static scene is an open question. The ATE threshold was not relaxed.

## Test results (final code)
* `scripts/run_all_tests.py` on the final code (`logs/final/final_run_all_tests.log`): python syntax, URDF, diagnose, **pytest 138 passed / 0 failed / 3 deselected**, calibration (4 checks), kit post-process, ROS sensor topics: all PASS. The session ended before the suite's last two stages, so they were run directly: **ROS 2 + MAVLink comm test 16/16 PASS** (`logs/final/final_comm_test.log`); **ROS mission test `tests/test_ros_demo.py` FAILED** — `ate_rmse_m 1.36 < 0.8` (every other assertion before it held: mission complete, 0 collisions, landed and disarmed). It passed on the earlier baseline run with ATE < 0.8.
* New tests (all pass): `test_orbslam3.py` (5), `test_front_guard.py` (3), `test_slam_map_handling.py` (2), `test_orbslam3_runner_crash.py` (1), `test_dynamic_memory.py` (6).
* Baseline before changes: pytest 124 passed / 2 failed (in-process rig hard-wired to the VO fallback: full mission ATE 0.577 > 0.5; dynamic mission), comm test 15/16 once inside the suite (step 14 landed flag at 0.33 m vs the < 0.3 m check; 4/4 isolated runs 16/16 with 0.13 m).
* No threshold, assertion or test was relaxed or deleted.

## Ground-truth firewall audit
`tests/test_ground_truth_firewall.py` (AST import test, topic allow-list, FC-pose subscription ban) passes; the new code (`perception/front_clearance.py`, map-id handling, dynamic memory) imports only numpy/cv2. `tools/diagnose_slam.py` FIREWALL check: `/uav/odom` has no SLAM / mapping / nav subscriber. Ground truth appears only in `eval/`, `logs/*/sim/ground_truth.csv`, and `trace.csv`/`trajectory.npy` columns written by the offline rig.

## Remaining limitations
* Dynamic scene: moving-obstacle collisions are mitigated, not eliminated (forward-only camera); the memory can strand a mission (reported as skipped waypoints, safe landing); S2 missed; ATE can exceed 0.8 m.
* ORB-SLAM3 is non-deterministic; all numbers are single runs (static: 3 ROS runs, dynamic: 2 ROS runs + ~12 in-process runs).
* Thermal is synthetic and non-radiometric. Only a fire hazard class exists (no flood / electrical / chemical).
* Hardware flight is not validated; altitude behaviour is simulation behaviour. Live ORB-SLAM3 needs Linux in this implementation.

## Reproduce
```bash
source /opt/ros/humble/setup.bash
bash slam/build_orbslam3.sh                                  # once
python3 scripts/run_disaster_demo.py --slam-backend orbslam3                   # static, with video
python3 scripts/run_disaster_demo.py --slam-backend orbslam3 --scenario dynamic --no-video
python3 -m eval.rigs.run_inprocess --scenario dynamic --out logs/final/inproc_x   # in-process rig, keeps trace.csv
python3 scripts/run_all_tests.py --ros-python $(which python3)
ROS_DOMAIN_ID=88 python3 tools/diagnose_slam.py              # while a mission runs
```
