"""
In-process closed-loop rig: sim + SLAM + perception + live geo-semantic map + A* + VFH+ + mission, without ROS processes.

Used for fast tuning and for the navigation / integration tests. The autonomy modules are the SAME classes the ROS nodes use.
The only shortcut versus the final demo: velocity commands are handed to the simulated FC's setpoint handler directly (exactly the
call the MAVLink server makes when it decodes SET_POSITION_TARGET_LOCAL_NED) instead of travelling over ROS + UDP. The full ROS ->
MAVLink -> FC path is exercised by scripts/run_disaster_demo.py.

    python -m eval.rigs.inprocess_rig [--seconds 120] [--out logs/rig] [--video]
Ground truth is recorded for OFFLINE scoring only (collisions, survivor error, trajectory error).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from common.frames import euler_from_R, ros_twist_to_ned  # noqa: E402
from common.geo import GeoFrame  # noqa: E402
from mapping.geo_anchor import GpsAnchor  # noqa: E402
from mapping.pipeline import PerceptionMappingPipeline  # noqa: E402
from mission.disaster_demo import MissionController  # noqa: E402
from sim.config import load_config  # noqa: E402
from sim.flight_controller_sim import IGN_POS, IGN_ACC, IGN_YAW  # noqa: E402
from sim.rgbd_camera import RGBDSensor  # noqa: E402
from sim.sim_core import UavSim  # noqa: E402
from sim.thermal_camera import ThermalSensor  # noqa: E402
from slam.adapter import SlamFrontEnd  # noqa: E402
from slam.orbslam3_backend import OrbSlam3Backend, resolve_backend  # noqa: E402
from slam.vo_backend import OrbPnpVO  # noqa: E402


def run(seconds=130.0, out=None, verbose=True, world=True, cfg=None, snapshot_every_s=0.0):
    cfg = cfg or load_config()
    out = out or os.path.join(ROOT, "logs", "rig")
    os.makedirs(out, exist_ok=True)
    s = UavSim(cfg, world=world)
    cam, th = RGBDSensor(s), ThermalSensor(s, s.world_info["thermal_c"])
    Krgb, Kth = cam.intrinsics(), th.intrinsics()
    # SLAM backend: config slam.backend with the same "auto" rule as the launcher (ORB-SLAM3 where it is built, VO fallback otherwise).
    # UAV_RIG_SLAM=orbslam3|orb_pnp_vo forces one (orbslam3 fails loudly if it is not built).
    backend_kind, _ = resolve_backend(cfg["slam"]["backend"], os.environ.get("UAV_RIG_SLAM", "auto"))
    backend = OrbSlam3Backend(Krgb, cfg) if backend_kind == "orbslam3" else OrbPnpVO(Krgb, cfg["slam"])
    fe = SlamFrontEnd(cfg, backend)
    pipe = PerceptionMappingPipeline(cfg, Krgb, Kth, evidence_dir=os.path.join(out, "evidence"))
    anchor = GpsAnchor(cfg["sensor_extrinsics"]["gps"]["xyz_body_flu_m"])
    mc = MissionController(cfg, pipe.map.metric.v, pipe.map.metric.origin[:2])
    fc = s.fc
    cam_div = int(round(s.hz / cam.rate_hz))
    nav_div = int(round(s.hz / cfg["nav"]["control_hz"]))
    s.hooks.append(lambda sim: (fe.feed_imu(sim.imu_sample.t, sim.imu_sample.gyro, sim.imu_sample.accel) if sim.new_imu else None,
                                anchor.add(sim.gps_sample.lat_deg, sim.gps_sample.lon_deg, sim.gps_sample.alt_m, sim.gps_sample.fix_type)
                                if sim.new_gps else None))
    s.run(0.2)
    yaw_ned0 = euler_from_R(s.estimate.R_ned_frd)[2]
    heading0 = math.pi / 2 - yaw_ned0
    fe.set_initial_heading(yaw_ned0)
    start_enu = s.vehicle.read_state().pos_enu.copy()

    cur = {"pose": None, "slam_ok": False, "slam_t": -1.0, "grid": None, "tr": None}
    traj, collisions, cmds = [], 0, []
    touchdowns, airborne = [], False             # unplanned touchdowns: FC 'landed' while the mission is in a flying state
    slam_log, collision_log = [], []
    t_wall0 = time.perf_counter()
    last_snap = -1e9
    contacts_seen = set()
    while s.t < seconds and not mc.done:
        s.step()
        if s.step_count % cam_div == 0:
            rgb, depth = cam.capture()
            T = th.capture()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pose = fe.process(s.t, bgr, depth)
            if not pipe.map.geo.ready:
                r = anchor.try_finish(heading0)
                if r:
                    pipe.map.geo.set_anchor(*r)
            slam_log.append((round(s.t, 2), pose.state, {k: v for k, v in pose.info.items()}))
            if pose.state != "OK" and snapshot_every_s < 0:
                cv2.imwrite(os.path.join(out, f"lost_{int(s.t*10):04d}.png"), bgr)
            cur["slam_ok"] = pose.T_map_body is not None and pose.state == "OK"
            if cur["slam_ok"]:
                cur["pose"], cur["slam_t"] = pose.T_map_body, s.t
                if pipe.map.geo.ready:
                    res_ = pipe.process(s.t, bgr, depth, T, pose.T_map_body)
                    cur["front"] = (s.t, res_.get("front_clearance_m"))
            if snapshot_every_s and s.t - last_snap >= snapshot_every_s:
                last_snap = s.t
                cv2.imwrite(os.path.join(out, f"rgb_{int(s.t):03d}.jpg"), pipe.annotate(bgr, {"persons": []}))
        if s.step_count % nav_div == 0:
            pose = cur["pose"]
            if pose is not None and pipe.map.geo.ready:
                z0 = cfg["demo"]["flight_altitude_m"]
                grid = pipe.map.planning_grid(z0)
                cur["grid"] = grid
                R = pose[:3, :3]
                yaw = math.atan2(R[1, 0], R[0, 0])
                confirmed = [t_.track_id for t_ in pipe.map.survivors.confirmed()]
                st = mc.step(s.t, pose[:3, 3], yaw, grid, pipe.map.metric.version, cur["slam_ok"], s.t - cur["slam_t"], fc.armed, fc.mode, confirmed, pipe.map.geo.ready,
                             front_clear_m=cur.get("front", (0, None))[1], front_age_s=s.t - cur.get("front", (-99.0, None))[0])
            else:
                st = mc.step(s.t, None, None, np.zeros((2, 2), np.int8), 0, False, 99.0, fc.armed, fc.mode)
            for a in st.actions:
                if a == "ARM":
                    fc.arm(s.t, battery_ok=s.battery.arming_ok())
                elif a == "OFFBOARD":
                    fc.set_mode("OFFBOARD", s.t)
                elif a == "LAND" and fc.mode != "LAND":
                    fc.set_mode("LAND", s.t)
            c = st.cmd
            vn, ve, vd, yr = ros_twist_to_ned("enu", (c.vx, c.vy, c.vz), c.yaw_rate, 0.0)
            # the FC's MAVLink handler call (mask = velocity + yaw rate), as sim/mavlink_fc_server.py does
            fc.handle_position_target(s.t, 1, IGN_POS | IGN_ACC | IGN_YAW, np.zeros(3), (vn, ve, vd), np.zeros(3), 0.0, yr)
            stt = s.vehicle.read_state()
            airborne = airborne or (fc.armed and not fc.landed and stt.pos_enu[2] - start_enu[2] > 1.0)
            if airborne and fc.landed and mc.state not in ("TAKEOFF", "LANDING", "DONE", "ABORTED") and                     (not touchdowns or s.t - touchdowns[-1]["t"] > 5.0):
                touchdowns.append({"t": round(s.t, 1), "mission_state": mc.state, "uav_gt_local": [round(float(v), 2) for v in (stt.pos_enu - start_enu)]})
            traj.append([s.t, *(stt.pos_enu - start_enu), *(pose[:3, 3] if pose is not None else [np.nan] * 3), c.vx, c.vy, c.vz, c.yaw_rate,
                         st.status.get("state", ""), st.status.get("target", ""), cur.get("front", (0, None))[1], getattr(stt, "yaw", float("nan"))])
            if fc.armed and not fc.landed and (stt.pos_enu[2] - start_enu[2]) > 0.5:      # in flight (skids touching the road at lift-off do not count)
                cps = [cp for cp in s.vehicle.contacts_with_world() if cp[2] != s.world_info["ground_body"]]
                if cps:
                    key = cps[0][2]
                    if key not in contacts_seen:
                        contacts_seen.add(key)
                        collisions += 1
                        collision_log.append({"t": round(s.t, 1), "body": int(key), "name": s.world_info["obstacle_ids"].get(key, "scene object"),
                                              "uav_gt_local": [round(float(v), 2) for v in (stt.pos_enu - start_enu)]})
    wall = time.perf_counter() - t_wall0
    geo = GeoFrame.from_config(cfg)
    gt = {h["id"]: np.array(h["xy_enu"]) - start_enu[:2] for h in s.world_info["humans"]}
    tr = np.array([[r[0], *r[1:7]] for r in traj], float)
    with open(os.path.join(out, "trace.csv"), "w") as f:            # per-nav-tick trace for offline analysis (GT columns are evaluation only)
        f.write("t,gt_x,gt_y,gt_z,slam_x,slam_y,slam_z,cmd_vx,cmd_vy,cmd_vz,cmd_yr,state,target,front_clearance_m,gt_yaw\n")
        for r in traj:
            f.write(",".join(str(v) for v in r) + "\n")
    ok = np.isfinite(tr[:, 4])
    err = np.linalg.norm(tr[ok, 4:7] - tr[ok, 1:4], axis=1) if ok.any() else np.array([np.nan])
    rep = {
        "slam_backend": backend.name, "sim_time_s": s.t, "wall_time_s": wall, "mission_done": mc.done, "final_state": mc.state, "waypoint_index": mc.idx,
        "end_reason": mc.end_reason, "skipped_waypoints": list(mc.skipped), "mission_events": list(mc.events), "fc_final_mode": fc.mode,
        "fc_armed_at_end": bool(fc.armed), "landed_at_end": bool(fc.landed),
        "collisions": collisions, "collision_log": collision_log, "unplanned_touchdowns": touchdowns, "planner_stats": mc.stats,
        "slam_ate_rmse_m": float(np.sqrt(np.nanmean(err ** 2))), "slam_ate_max_m": float(np.nanmax(err)),
        "slam_tracked_fraction": fe.backend.n_ok / max(fe.backend.n_ok + fe.backend.n_lost, 1),
        "survivors": [], "hazards": [{"id": h.hazard_id, "status": h.status, "xy": [round(h.x, 2), round(h.y, 2)]} for h in pipe.map.hazards.items.values()],
        "map_frames": pipe.map.metric.n_frames,
        "scenario": s.world_info.get("scenario", "static") if s.world_info else "none",
        "world_events": list(s.world_info["dynamic"].events) if s.world_info and s.world_info.get("dynamic") else [],
    }
    for t_ in pipe.map.survivors.tracks.values():
        k, d = min(((k, float(np.linalg.norm(np.array([t_.local_x, t_.local_y]) - v))) for k, v in gt.items()), key=lambda kv: kv[1])
        rep["survivors"].append({"id": t_.track_id, "status": t_.status, "obs": t_.observation_count, "nearest_gt": k, "error_m": round(d, 2),
                                 "lat": t_.latitude, "lon": t_.longitude, "first_seen": round(t_.first_seen, 1), "confirmed_at": round(t_.confirmed_at, 1)})
    np.save(os.path.join(out, "trajectory.npy"), tr)
    json.dump(slam_log, open(os.path.join(out, "slam_log.json"), "w"), default=float)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=float)
    # keep the final map for quick inspection
    g = pipe.map.planning_grid(cfg["demo"]["flight_altitude_m"])
    img = np.zeros(g.shape + (3,), np.uint8) + 60
    img[g == 0] = (200, 200, 200)
    img[g == 100] = (40, 40, 230)
    img = np.transpose(img, (1, 0, 2))[::-1]
    cv2.imwrite(os.path.join(out, "final_planning_grid.png"), cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    try:
        backend.close()
    except Exception:
        pass
    s.close()
    if verbose:
        print(json.dumps(rep, indent=1, default=float))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=130.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.seconds, a.out)
