# Algorithm interface contract (stage 2: implemented)

The autonomy stack is the set of ROS 2 nodes `ros2_bridge/slam_node.py`, `perception_mapping_node.py`, `nav_node.py` (+ `video/compositor_node.py` for display)
built on the pure-Python modules in `slam/`, `perception/`, `mapping/`, `planning/`, `mission/`.

## Hard rules (enforced by `tests/test_ground_truth_firewall.py`)
1. `algo/ companion/ ros2_bridge/ slam/ perception/ mapping/ planning/ mission/ video/` never import `pybullet`, `sim`, `worlds`, `eval` or `tests`.
2. The autonomy nodes never subscribe to the FC pose estimate (`/uav/odom`); the UAV pose comes from SLAM only.
3. The algorithm never teleports or otherwise touches the vehicle. Its only output is `/uav/cmd_vel` (→ bridge → MAVLink → FC → PyBullet).
4. Ground truth (poses, survivor / obstacle positions, body ids, contacts) exists only in `eval/` and `logs/`; it is read only after the flight by `eval/demo_scoring.py`.

## Inputs (all legitimate on the real vehicle)
| Input | Topic | Type | Frame / units |
|---|---|---|---|
| RGB | `/uav/camera/color/image_raw` | `rgb8` 640×480 | `rgbd_camera_optical_link` (z fwd, x right, y down) |
| Depth | `/uav/camera/depth/image_raw` | `32FC1` | **metres** along the optical axis, NaN = invalid, registered to colour |
| Thermal | `/uav/camera/thermal/image_raw` | `32FC1` 160×120 | **°C** (SYNTHETIC in simulation), `thermal_camera_optical_link` |
| Camera info | `/uav/camera/{color,depth,thermal}/camera_info` | `CameraInfo` | pinhole K, no distortion |
| IMU | `/uav/imu` | `Imu` | REP-103 **FLU**, rad/s and m/s², stamped in FC (simulation) time |
| GPS | `/uav/gps/fix` | `NavSatFix` | WGS-84, stamped in FC time |
| FC state | `/uav/telemetry`, `/uav/armed`, `/uav/mode` | JSON / Bool / String | mode, armed, battery, FC heading at start, MAVLink link health (`link`), FC failsafe flag |
Static calibration (camera/IMU/GPS extrinsics in the body frame) is read from `config/drone.yaml → sensor_extrinsics` (derived from the URDF).
All stamps share one time base (simulation time), so images and IMU can be paired exactly.

## Outputs
| Topic | Meaning |
|---|---|
| `/uav/slam/pose` (`PoseStamped`) | UAV **body** pose in `slam_map` — x East, y North, z Up, origin = take-off body position, gravity- and FC-heading-aligned |
| `/uav/map/planning_grid` (`OccupancyGrid`) | live metric map layer for planning: 0 FREE, 100 OCCUPIED, −1 UNKNOWN (unknown is never free) |
| `/uav/map/semantic` (JSON) | survivor nodes and hazard nodes with local + geographic coordinates |
| **`/uav/cmd_vel` (`Twist`)** | **ENU of `slam_map`: linear.x East, linear.y North, linear.z Up [m/s]; angular.z counter-clockwise [rad/s]** (bridge parameter `cmd_frame: enu`). The bridge converts to MAVLink NED (`vn = y, ve = x, vd = −z, yaw_rate_cw = −angular.z`) and sends `SET_POSITION_TARGET_LOCAL_NED`. |

Keep publishing ≥ 2 Hz; if the setpoint stream stops for 0.5 s the simulated FC enters `FAILSAFE_HOLD`.

## Survivor node (`mapping/geo_semantic_map.py: SurvivorNode`)
`track_id, class, confidence, local_x, local_y, local_z, latitude, longitude, altitude, position_uncertainty, first_seen, last_seen, observation_count,
status (CANDIDATE | CONFIRMED), evidence_frame` (+ `thermal_support`, `score`, `confirmed_at`). One node per person: detections are associated within 1.5 m, filtered by 1/σ² weighting,
confirmed after ≥ 3 sightings with accumulated score ≥ 1.6, duplicates closer than 2.5 m are merged, detections whose torso-depth patch spans several surfaces are rejected.

## Backend selection
`config/drone.yaml → slam.backend: orbslam3` (default; the attached kit's ORB-SLAM3 via `slam/rgbd_stream.cc`, built and validated on Ubuntu 22.04 + ROS 2 Humble)
or `orb_pnp_vo` (pure-Python RGB-D ORB + PnP odometry, no loop closure; runs everywhere). The SLAM node parameter `backend` / the demo flag
`--slam-backend` override it; `auto` (the demo default) keeps ORB-SLAM3 when its runner is built and otherwise falls back to `orb_pnp_vo` with a
warning in the log, `/uav/slam/status.backend_note` and `demo_summary.json`. **ORB-SLAM3 is not built on Windows**, so Windows runs use the fallback.

## Mission actions and safety (`mission/disaster_demo.py`, executed by `nav_node.py`)
Besides `/uav/cmd_vel`, the mission requests `ARM`, `OFFBOARD` and `LAND` through the bridge services `/uav/arm`, `/uav/offboard`, `/uav/land`
(→ MAVLink `COMMAND_LONG`). It reads MAVLink link health and the FC failsafe flag from `/uav/telemetry` (`link.ok`, `link.heartbeat_age_s`,
`failsafe`; the FC position in that message is **not** used). Safety behaviour, all hover-first and never a fall-back to ground truth:

| Condition | Mission state | Behaviour |
|---|---|---|
| SLAM lost / pose stale | `LOST_HOLD` | hover, slow yaw back toward the last good view; still lost after `nav.lost_land_s` → `LANDING` |
| planning grid older than `nav.grid_stale_s` (sensor / mapping loss) | `MAP_STALE` | hover until the map is live again |
| no MAVLink heartbeat/telemetry for `nav.fc_link_timeout_s` | `LINK_LOST` | zero command (the FC's own 0.5 s setpoint watchdog protects the vehicle) |
| FC in `FAILSAFE_HOLD` / `LAND` (watchdog, low battery) | `LANDING` | stop commanding motion, request LAND |
| vehicle disarmed in flight | `ABORTED` | mission ends |
| no A\* route | `NO_ROUTE_HOLD` | hover, turn toward the goal to map it, re-plan; skip the waypoint after `nav.no_route_timeout_s` (reported) |
| VFH+ finds no opening | `VFH_BLOCKED` | hover, force an A\* re-plan; skip the waypoint after `nav.blocked_timeout_s` |
| obstacle closer than `nav.min_obstacle_dist_m` | `AVOID_RETREAT` | back away from the nearest occupied cell (direction from the map, never a fixed side), re-plan |
| end of a partial A\* path (next cells unknown) | (in `SEARCH`) | rotate / observe only; never flies into unknown space |
| non-finite command | — | replaced by a hover command (and the bridge rejects non-finite `/uav/cmd_vel` too) |
| mission complete | `HOLD_END` → `LANDING` → `DONE` | hover `demo.hold_after_mission_s`, then LAND (`demo.end_action: land`), done when the FC auto-disarms |
