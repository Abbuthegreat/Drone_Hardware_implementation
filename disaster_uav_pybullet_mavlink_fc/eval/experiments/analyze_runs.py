"""
Offline analysis of repeated missions (ground truth used here only, after the flights).

    python3 eval/experiments/analyze_runs.py logs/experiments/static_base [more dirs ...] [--out logs/experiments/report]

Per run: mission outcome, SLAM accuracy (official numbers from score_report.json), ORB-SLAM3 frame states / map ids / log events, navigation and
safety counters, survivors / hazards, landing, moving-obstacle memory activity, CPU load, and the onset of the localisation error
(first time the error exceeds 0.5 m, with the ground-truth position there) to separate random drift from a location-specific failure.
Per set: N, mean, median, std, min, max, p95, pass/fail counts, P(ATE < 0.8), P(complete & ATE < 0.8 & 0 collisions),
Spearman correlations of ATE with every numeric factor, and an ATE-per-run plot.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

ATE_MAX = 0.8          # existing acceptance threshold (tests/test_ros_demo.py) - not changed


def _csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def error_timeline(run):
    S = np.genfromtxt(os.path.join(run, "slam", "slam_pose.csv"), delimiter=",", names=True, dtype=None, encoding=None)
    G = np.genfromtxt(os.path.join(run, "sim", "ground_truth.csv"), delimiter=",", names=True, dtype=None, encoding=None)
    gt = np.c_[G["pe"], G["pn"], -G["pd"]]
    org = gt[np.searchsorted(G["t"], S["t"][0])]
    idx = np.clip(np.searchsorted(G["t"], S["t"]), 0, len(G) - 1)
    g = gt[idx] - org
    e = np.linalg.norm(np.c_[S["x"], S["y"], S["z"]] - g, axis=1)
    return S["t"], e, g


def analyze(run):
    r = {"run": os.path.relpath(run)}
    sr, ds = os.path.join(run, "score_report.json"), os.path.join(run, "demo_summary.json")
    if not os.path.exists(sr) or not os.path.exists(ds):
        r["status"] = "NO_REPORT"
        return r
    s, sc = json.load(open(ds)), json.load(open(sr))
    sl, fl, ss = sc["slam"], sc["flight"], sc["survivor_summary"]
    nav = s.get("nav_stats", {})
    lx, ly = fl["final_gt_local_xy"]
    conf = [x for x in sc["survivors"] if x["status"] == "CONFIRMED"]
    haz = [h for h in sc["hazards"] if h["status"] == "CONFIRMED"]
    gtr = _csv(os.path.join(run, "sim", "ground_truth.csv"))
    dt_gt = (float(gtr[-1]["t"]) - float(gtr[0]["t"])) / max(len(gtr) - 1, 1) if gtr else 0.0
    r.update(status="OK", backend=s.get("slam_backend"), scenario=s.get("scenario"), reason=s.get("reason"), final_state=s.get("final_state"),
             complete=(s.get("reason") == "mission complete" and s.get("final_state") == "DONE" and not s.get("skipped_waypoints")),
             skipped=len(s.get("skipped_waypoints") or []), return_home=("RETURN home" not in (s.get("skipped_waypoints") or [])) and s.get("final_state") == "DONE",
             sim_s=round(s.get("sim_s", 0), 1), wall_s=round(s.get("wall_s", 0), 1),
             ate=sl["ate_rmse_m"], ate_max=sl["ate_max_m"], final_err=sl["final_error_m"], path_m=sl["gt_path_m"], drift_pct=sl["drift_pct_of_path"], poses=sl["poses"],
             collisions=fl["collision_samples_in_flight"], contact_s=round(fl["collision_samples_in_flight"] * dt_gt, 2), touchdowns=fl["unplanned_touchdown_samples"],
             landed_disarmed=fl["landed_disarmed_at_end"], landing_m=round(float(np.hypot(lx, ly)), 2),
             survivors=",".join(sorted(x["nearest_truth"] for x in conf)), n_survivors=len(conf), s2_found=any(x["nearest_truth"] == "S2" for x in conf),
             dup_nodes=ss["false_or_duplicate_nodes"], fire=len(haz) >= 1, dup_hazards=max(len(haz) - 1, 0),
             s1_obs=sum(x["observations"] for x in conf if x["nearest_truth"] == "S1"), s2_obs=sum(x["observations"] for x in conf if x["nearest_truth"] == "S2"),
             replans=nav.get("replans", 0), vfh_ticks=nav.get("vfh_active_ticks", 0), retreat_ticks=nav.get("retreat_ticks", 0), lost_ticks=nav.get("lost_ticks", 0),
             front_hold=nav.get("front_guard_hold_ticks", 0), front_retreat=nav.get("front_guard_retreat_ticks", 0), failsafe=nav.get("failsafe_events", 0),
             map_stale=nav.get("map_stale_ticks", 0), link_lost=nav.get("link_lost_ticks", 0), setpoints=sc.get("mavlink", {}).get("rx", {}).get("SET_POSITION_TARGET_LOCAL_NED", 0))
    # ORB-SLAM3 frame states (slam_frames.csv, every frame)
    fr = _csv(os.path.join(run, "slam", "slam_frames.csv"))
    if fr:
        st = [x["state"] for x in fr]
        first_ok = next((i for i, x in enumerate(st) if x == "OK"), len(st))
        lost_after = st[first_ok:]
        episodes = sum(1 for a, b in zip(lost_after, lost_after[1:]) if a == "OK" and b != "OK")
        ids = {x["map_id"] for x in fr if x["map_id"] not in ("", "-1")}
        r.update(frames=len(fr), frames_ok=st.count("OK"), frames_lost=sum(1 for x in lost_after if x != "OK"), startup_nonok=first_ok,
                 lost_episodes=episodes, new_map_events=sum(int(x["new_map"] or 0) for x in fr), map_ids=len(ids),
                 runner_crashes=sum(int(x["runner_crashed"] or 0) for x in fr), slam_ms=round(float(np.mean([float(x["latency_ms"]) for x in fr])), 1))
    # ORB-SLAM3's own console (runner stderr -> slam.log)
    lg = os.path.join(run, "slam.log")
    if os.path.exists(lg):
        txt = open(lg, errors="ignore").read()
        r.update(orb_fail_local_map=txt.count("Fail to track local map"), orb_relocalized=len(re.findall(r"[Rr]elocali[sz]ed", txt)),
                 orb_new_map_log=txt.count("Creation of new map with id"), orb_loop=len(re.findall(r"Loop detected|LOOP CLOSURE|\*Loop", txt)),
                 orb_merge=len(re.findall(r"Merge detected|\*Merge", txt)), orb_nan=txt.count("ensure failed"))
    # moving-obstacle memory
    sem = _jsonl(os.path.join(run, "perception", "survivor_map.jsonl"))
    if sem:
        r.update(mem_events=max(x.get("map", {}).get("dynamic_memory_events", 0) for x in sem),
                 mem_cells_max=max(x.get("map", {}).get("dynamic_memory_cells", 0) for x in sem),
                 mem_logged="dynamic_memory_events" in sem[-1].get("map", {}))
    ev = [e for e in _jsonl(os.path.join(run, "nav", "nav_events.jsonl")) if "event" in e]
    for k in ("OBSTACLE_TOO_CLOSE", "FRONT_CLEARANCE_RETREAT", "LOCALIZATION_LOST", "MAP_STALE", "LINK_LOST", "WAYPOINT_SKIPPED", "NO_ROUTE", "VFH_NO_OPENING"):
        r["ev_" + k.lower()] = sum(1 for e in ev if e["event"] == k)
    ld = _csv(os.path.join(run, "load.csv"))
    if ld:
        r.update(load1_mean=round(float(np.mean([float(x["load1"]) for x in ld])), 2), cpu_busy_mean=round(float(np.mean([float(x["cpu_busy_pct"]) for x in ld])), 1))
    # error onset
    try:
        t, e, g = error_timeline(run)
        k = np.where(e > 0.5)[0]
        r.update(err_onset_t=round(float(t[k[0]]), 1) if len(k) else None,
                 err_onset_xy=[round(float(g[k[0], 0]), 1), round(float(g[k[0], 1]), 1)] if len(k) else None,
                 err_at_40s=round(float(e[np.searchsorted(t, 40.0)]), 2) if t[-1] > 40 else None,
                 err_at_80s=round(float(e[np.searchsorted(t, 80.0)]), 2) if t[-1] > 80 else None,
                 err_max_step=round(float(np.max(np.abs(np.diff(e)))), 2),
                 err_max_step_t=round(float(t[1 + int(np.argmax(np.abs(np.diff(e))))]), 1))
    except Exception as ex:      # noqa: BLE001
        r["err_timeline_error"] = str(ex)
    return r


def stats(v):
    v = np.array([x for x in v if x is not None], float)
    if not len(v):
        return {"N": 0}
    return {"N": int(len(v)), "mean": round(float(v.mean()), 3), "median": round(float(np.median(v)), 3), "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 3),
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3), "p95": round(float(np.percentile(v, 95)), 3)}


def summarize(rows):
    ok = [r for r in rows if r.get("status") == "OK"]
    ate = [r["ate"] for r in ok]
    n = len(rows)
    passed = [r for r in ok if r["ate"] < ATE_MAX]
    full = [r for r in ok if r["complete"] and r["ate"] < ATE_MAX and r["collisions"] == 0]
    out = {"runs": n, "runs_with_report": len(ok), "ate": stats(ate), "ate_pass": len(passed), "ate_fail": len(ok) - len(passed),
           "P_ate_lt_0.8": round(len(passed) / max(n, 1), 3), "P_complete_ate_ok_no_collision": round(len(full) / max(n, 1), 3),
           "mission_complete": sum(1 for r in ok if r["complete"]), "collision_runs": sum(1 for r in ok if r["collisions"] > 0),
           "landing_m": stats([r["landing_m"] for r in ok]), "final_err_m": stats([r["final_err"] for r in ok]),
           "s2_found": sum(1 for r in ok if r["s2_found"]), "dup_survivor_runs": sum(1 for r in ok if r["dup_nodes"] > 0),
           "dup_hazard_runs": sum(1 for r in ok if r["dup_hazards"] > 0), "fire_runs": sum(1 for r in ok if r["fire"]),
           "skipped_runs": sum(1 for r in ok if r["skipped"] > 0), "return_home_runs": sum(1 for r in ok if r["return_home"])}
    # Spearman correlation of ATE with every numeric factor
    try:
        from scipy.stats import spearmanr
        cors = {}
        for k in ok[0].keys() if ok else []:
            vals = [r.get(k) for r in ok]
            if k in ("ate", "ate_max", "final_err", "drift_pct", "landing_m") or not all(isinstance(v, (int, float, bool)) and v is not None for v in vals):
                continue
            x = np.array(vals, float)
            if len(set(x)) < 2 or len(x) < 4:
                continue
            rho, p = spearmanr(x, ate)
            cors[k] = {"rho": round(float(rho), 2), "p": round(float(p), 3)}
        out["spearman_vs_ate"] = dict(sorted(cors.items(), key=lambda kv: -abs(kv[1]["rho"])))
    except Exception as ex:      # noqa: BLE001
        out["spearman_error"] = str(ex)
    return out


def plot(sets, path):
    W, H, m = 1000, 420, 60
    img = np.full((H, W, 3), 255, np.uint8)
    allv = [r["ate"] for _, rows in sets for r in rows if r.get("status") == "OK"]
    vmax = max(1.0, max(allv) * 1.1 if allv else 1.0)
    y = lambda v: int(H - m - v / vmax * (H - 2 * m))
    cv2.line(img, (m, y(ATE_MAX)), (W - 20, y(ATE_MAX)), (0, 0, 220), 1)
    cv2.putText(img, "0.8 m threshold", (W - 170, y(ATE_MAX) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    cols = [(200, 120, 0), (0, 150, 0), (150, 0, 150), (0, 120, 200)]
    n_all = sum(len(rows) for _, rows in sets)
    x0, step = m + 10, (W - m - 40) / max(n_all, 1)
    i = 0
    for si, (name, rows) in enumerate(sets):
        c = cols[si % len(cols)]
        cv2.putText(img, name, (m + 10 + si * 230, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        for r in rows:
            if r.get("status") == "OK":
                x = int(x0 + i * step)
                cv2.circle(img, (x, y(r["ate"])), 5, c, -1)
                if r["collisions"] or not r["complete"]:
                    cv2.circle(img, (x, y(r["ate"])), 9, (0, 0, 0), 1)
            i += 1
    cv2.line(img, (m, m), (m, H - m), (0, 0, 0), 1)
    for v in np.arange(0, vmax + 1e-9, 0.25):
        cv2.putText(img, f"{v:.2f}", (5, y(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    cv2.putText(img, "ATE RMSE [m] per run (circled = collision or incomplete mission)", (m, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sets, report = [], {}
    for d in a.dirs:
        runs = sorted(glob.glob(os.path.join(d, "run_*")))
        rows = [analyze(r) for r in runs]
        name = os.path.basename(os.path.normpath(d))
        sets.append((name, rows))
        report[name] = {"summary": summarize(rows), "runs": rows}
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(d, "runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    out = a.out or a.dirs[0]
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, "analysis.json"), "w"), indent=1, default=str)
    plot(sets, os.path.join(out, "ate_per_run.png"))
    for name, rep in report.items():
        print(name, json.dumps({k: v for k, v in rep["summary"].items() if k != "spearman_vs_ate"}, default=str))
        sp = rep["summary"].get("spearman_vs_ate", {})
        print("  strongest Spearman vs ATE:", list(sp.items())[:8])


if __name__ == "__main__":
    main()
