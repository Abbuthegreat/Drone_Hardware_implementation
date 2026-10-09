"""
Command-centre / rescue output: one machine-readable status record per tick and a final rescue report (JSON + Markdown).

Inputs are ONLY what the companion side publishes: the navigation status (/uav/nav/status), the geo-semantic map (/uav/map/semantic),
the MAVLink-bridge telemetry (/uav/telemetry: FC mode, armed, link health, failsafe flag, battery) and the SLAM status. The UAV position
reported here is the SLAM ESTIMATE (geo-referenced through the GPS anchor), never the simulator's pose.
"""
from __future__ import annotations

import math
from typing import Optional

from common.geo import GeoFrame

PAUSE_STATES = ("LOST_HOLD", "MAP_STALE", "NO_ROUTE_HOLD", "VFH_BLOCKED", "AVOID_RETREAT", "LINK_LOST", "FC_NOT_OFFBOARD")
SAFE_EXIT_STATES = ("LANDING", "ABORTED")


def _geo(sem: dict) -> Optional[GeoFrame]:
    a = (sem or {}).get("geo_anchor")
    return None if not a else GeoFrame(a["lat"], a["lon"], a["alt"])


def _r(v, n=2):
    return None if v is None else round(float(v), n)


def status_record(t_wall: float, nav: dict, sem: dict, tel: dict, slam: dict) -> dict:
    nav, sem, tel, slam = nav or {}, sem or {}, tel or {}, slam or {}
    geo = _geo(sem)
    pose = nav.get("pose_est_enu")
    uav = None
    if pose:
        uav = {"x_east_m": pose[0], "y_north_m": pose[1], "altitude_m": pose[2], "heading_enu_deg": _r(math.degrees(pose[3]), 1)}
        if geo is not None:
            lat, lon, alt = geo.enu_to_geodetic([pose[0], pose[1], pose[2]])
            uav.update(lat=round(lat, 7), lon=round(lon, 7), alt_m=round(alt, 2))
    vfh = nav.get("vfh") or {}
    link = tel.get("link") or {}
    state = nav.get("state", "-")
    return {
        "t_wall": round(t_wall, 2), "t_sim": _r(nav.get("t")), "mission_state": state, "current_target": nav.get("target"),
        "uav_estimated": uav,
        "map_age_s": nav.get("grid_age_s"),
        "astar": {"path": nav.get("astar_path", []), "partial": nav.get("path_partial"), "replans": (nav.get("stats") or {}).get("replans")},
        "vfh": {"direction_deg": _r(vfh.get("dir_deg"), 1), "target_direction_deg": _r(vfh.get("target_dir_deg"), 1), "blocked": vfh.get("blocked"),
                "nearest_obstacle_m": _r(vfh.get("nearest_obstacle_m"))},
        "survivors": [{"track_id": s["track_id"], "status": s["status"], "confidence": round(s["confidence"], 3),
                       "local_xyz_m": [round(s["local_x"], 2), round(s["local_y"], 2), round(s["local_z"], 2)],
                       "lat": round(s["latitude"], 7), "lon": round(s["longitude"], 7), "alt_m": round(s["altitude"], 2),
                       "uncertainty_m": round(s["position_uncertainty"], 2), "observations": s["observation_count"],
                       "thermal_support_obs": s.get("thermal_support", 0), "first_seen_s": s["first_seen"], "last_seen_s": s["last_seen"]}
                      for s in sem.get("survivors", [])],
        "hazards": [{"id": h["hazard_id"], "kind": h["kind"], "status": h["status"], "local_xy_m": [round(h["x"], 2), round(h["y"], 2)],
                     "lat": round(h["latitude"], 7), "lon": round(h["longitude"], 7), "max_temp_c": round(h["max_temp_c"], 1)}
                    for h in sem.get("hazards", [])],
        "mavlink": {"link_ok": link.get("ok"), "heartbeat_age_s": _r(link.get("heartbeat_age_s")), "setpoints_sent": tel.get("setpoints_sent"),
                    "cmd_rejected": tel.get("cmd_rejected")},
        "flight_controller": {"mode": tel.get("mode"), "armed": tel.get("armed"), "system_status": tel.get("system_status"),
                              "battery_v": _r(tel.get("battery_v")), "battery_pct": tel.get("battery_pct")},
        "failsafe": {"fc_failsafe": bool(tel.get("failsafe", False)), "mission_safety_state": state if state in PAUSE_STATES + SAFE_EXIT_STATES else None,
                     "last_event": nav.get("last_event")},
        "slam": {"state": slam.get("state"), "backend": slam.get("backend"), "backend_note": slam.get("backend_note") or None},
    }


def one_line(rec: dict) -> str:
    u = rec.get("uav_estimated") or {}
    conf = [s for s in rec["survivors"] if s["status"] == "CONFIRMED"]
    pos = f"({u.get('x_east_m', float('nan')):6.1f},{u.get('y_north_m', float('nan')):6.1f}) alt {u.get('altitude_m', float('nan')):4.1f}m" if u else "(no estimate)"
    return (f"t={rec['t_sim'] or 0:6.1f}s {rec['mission_state']:<14} -> {str(rec['current_target']):<16} UAV{pos} "
            f"VFH {rec['vfh']['direction_deg']}deg  survivors {len(conf)} conf/{len(rec['survivors'])}  hazards {len(rec['hazards'])}  "
            f"FC {rec['flight_controller']['mode']}/{'ARMED' if rec['flight_controller']['armed'] else 'disarmed'}  "
            f"link {'OK' if rec['mavlink']['link_ok'] else 'DOWN'}  failsafe {'YES' if rec['failsafe']['fc_failsafe'] else 'no'}")


def rescue_report(final: dict, events: list, summary: dict) -> tuple:
    """(json_dict, markdown_text) - what a rescue team needs: confirmed survivors with geo-tags, hazards, mission outcome."""
    conf = [s for s in final.get("survivors", []) if s["status"] == "CONFIRMED"]
    cand = [s for s in final.get("survivors", []) if s["status"] != "CONFIRMED"]
    doc = {"mission": summary, "confirmed_survivors": conf, "unconfirmed_candidates": cand, "hazards": final.get("hazards", []),
           "events": events, "final_status": final}
    L = ["# Rescue report (simulation)", "",
         f"Mission end: **{summary.get('reason')}**, final mission state **{summary.get('final_state')}**, simulated {summary.get('sim_s', 0):.1f} s.",
         f"SLAM backend: {summary.get('slam_backend')}. Positions are estimates (SLAM + GPS anchor); uncertainty is the tracker's 1-sigma model.", "",
         "## Confirmed survivors", ""]
    if conf:
        L += ["| # | lat | lon | alt m | local x,y,z m | ± m | confidence | obs | thermal obs | first seen s |", "|---|---|---|---|---|---|---|---|---|---|"]
        L += [f"| {s['track_id']} | {s['lat']:.7f} | {s['lon']:.7f} | {s['alt_m']:.1f} | {s['local_xyz_m']} | {s['uncertainty_m']} | {s['confidence']} | "
              f"{s['observations']} | {s['thermal_support_obs']} | {s['first_seen_s']} |" for s in conf]
    else:
        L.append("none")
    L += ["", f"Unconfirmed candidates: {len(cand)}", "", "## Hazards", ""]
    L += [f"* {h['kind']} #{h['id']} ({h['status']}) at {h['lat']:.7f}, {h['lon']:.7f}, max {h['max_temp_c']} °C (synthetic thermal channel)"
          for h in final.get("hazards", [])] or ["none"]
    L += ["", "## Mission events", ""] + [f"* t={e.get('t')} s {e.get('event')} {', '.join(f'{k}={v}' for k, v in e.items() if k not in ('t', 'event', 'kind', 'state'))}"
                                          for e in events]
    return doc, "\n".join(L) + "\n"
