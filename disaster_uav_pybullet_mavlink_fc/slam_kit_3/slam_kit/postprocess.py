#!/usr/bin/env python3
"""
postprocess.py - after the flight: detect humans, place them in the ORB-SLAM3 map, build outputs.

    python3 postprocess.py ~/flights/2026-09-30_22-10-00
    python3 postprocess.py RUN --stride 1 --conf 0.35       # detect on every frame, lower threshold
    python3 postprocess.py RUN --up -z                      # camera points DOWN (see README)

Needs in RUN: associations.txt, meta.json, rgb/, depth/  (from record.py)
and optionally CameraTrajectory.txt (from rgbd_offline). Without a trajectory you still
get the detection video, just no map.

Writes RUN/results/:
    report.html            open this first - map + every human found + snapshots
    map_topdown.png        bird's-eye map, trajectory, red H1/H2.. markers
    map_topdown_height.png same, coloured by height
    map.ply                dense coloured 3D map
    map_with_humans.ply    3D map + red pillars at humans + yellow flight path (MeshLab/CloudCompare)
    detections_video.mp4   camera feed with boxes, human IDs, SLAM status and a minimap
    humans.csv / humans.json   one row per unique human (x, y, z metres, times, #sightings)
    detections.csv         every single detection
    humans/H1.jpg ...      best snapshot of each human
"""
import argparse
import base64
import csv
import json
import os
import sys
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================== I/O
def load_assoc(run):
    frames = []
    with open(os.path.join(run, "associations.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and not line.startswith("#"):
                frames.append((float(p[0]), p[1], p[3]))
    return frames


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def load_trajectory(path):
    """Returns {time_ms_int: (R_wc 3x3, t_wc 3)} . Handles EuRoC (ns) and TUM (s) timestamps."""
    poses = {}
    if not os.path.exists(path):
        return poses
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 8 and not line.startswith("#"):
                rows.append([float(x) for x in p])
    if not rows:
        return poses
    rows = np.array(rows)
    if rows[:, 0].max() > 1e6:          # EuRoC style: 1e9 * seconds
        rows[:, 0] /= 1e9
    for t, x, y, z, qx, qy, qz, qw in rows:
        poses[int(round(t * 1000))] = (quat_to_R(qx, qy, qz, qw), np.array([x, y, z]))
    return poses


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    n = len(xyz)
    data = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(data.tobytes())


def rot_a_to_b(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate about any axis perpendicular to a
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


AXES = {"x": [1, 0, 0], "-x": [-1, 0, 0], "y": [0, 1, 0], "-y": [0, -1, 0], "z": [0, 0, 1], "-z": [0, 0, -1]}


# =============================================================================== main steps
def run_detection(det, run, frames, stride, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get("stride") == stride and cache.get("conf") == det.conf:
            print(f"[post] using cached detections ({cache_path})")
            return {int(k): v for k, v in cache["dets"].items()}
    dets = {}
    idxs = list(range(0, len(frames), stride))
    t0 = time.time()
    for n, i in enumerate(idxs):
        img = cv2.imread(os.path.join(run, frames[i][1]))
        if img is None:
            continue
        d = det(img)
        if d:
            dets[i] = d
        if n % 100 == 0:
            el = time.time() - t0
            fps = (n + 1) / max(el, 1e-6)
            print(f"[post] detect {n}/{len(idxs)}  {fps:.1f} img/s  ETA {(len(idxs) - n) / max(fps, 1e-6):.0f}s  "
                  f"frames with people so far: {len(dets)}", flush=True)
    with open(cache_path, "w") as f:
        json.dump({"stride": stride, "conf": det.conf, "dets": dets}, f)
    return dets


def localize(box, depth, K, depth_scale, max_depth):
    """3D point (camera frame) of a person box using the median depth of the torso region."""
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    u0, u1 = int(x1 + 0.3 * w), int(x2 - 0.3 * w) + 1
    v0, v1 = int(y1 + 0.2 * h), int(y1 + 0.6 * h) + 1
    patch = depth[max(v0, 0):max(v1, 1), max(u0, 0):max(u1, 1)].astype(np.float32) * depth_scale
    vals = patch[(patch > 0.2) & (patch < max_depth)]
    if vals.size < 15:
        return None
    z = float(np.median(vals))
    u, v = (x1 + x2) / 2.0, y1 + 0.4 * h
    fx, fy, cx, cy = K
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def cluster(points, radius, dz_max=2.5):
    """Greedy clustering on the horizontal plane (after alignment z = up)."""
    clusters = []  # list of lists of indices
    cents = []
    for i, p in enumerate(points):
        best, bd = -1, 1e9
        for c, cen in enumerate(cents):
            d = np.linalg.norm(p[:2] - cen[:2])
            if d < radius and abs(p[2] - cen[2]) < dz_max and d < bd:
                best, bd = c, d
        if best < 0:
            clusters.append([i])
            cents.append(p.copy())
        else:
            clusters[best].append(i)
            cents[best] = np.median(points[clusters[best]], axis=0)
    # merge clusters that ended up close
    merged = True
    while merged:
        merged = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if np.linalg.norm(cents[a][:2] - cents[b][:2]) < radius * 0.8:
                    clusters[a] += clusters[b]
                    cents[a] = np.median(points[clusters[a]], axis=0)
                    del clusters[b], cents[b]
                    merged = True
                    break
            if merged:
                break
    return clusters, cents


class VoxelMap:
    def __init__(self, voxel):
        self.v = voxel
        self.keys = np.zeros(0, np.int64)
        self.sum = np.zeros((0, 6), np.float64)   # xyz sum + rgb sum
        self.cnt = np.zeros(0, np.int64)
        self.buf = []

    def add(self, xyz, rgb):
        self.buf.append(np.hstack([xyz, rgb.astype(np.float64)]))
        if sum(len(b) for b in self.buf) > 400000:
            self._flush()

    def _flush(self):
        if not self.buf:
            return
        d = np.vstack(self.buf)
        self.buf = []
        ijk = np.floor(d[:, :3] / self.v).astype(np.int64) + (1 << 20)
        k = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
        keys = np.concatenate([self.keys, k])
        sums = np.vstack([self.sum, d])
        cnts = np.concatenate([self.cnt, np.ones(len(k), np.int64)])
        u, inv = np.unique(keys, return_inverse=True)
        s = np.zeros((len(u), 6))
        for c in range(6):
            s[:, c] = np.bincount(inv, weights=sums[:, c], minlength=len(u))
        self.keys, self.sum, self.cnt = u, s, np.bincount(inv, weights=cnts, minlength=len(u)).astype(np.int64)

    def result(self, min_hits):
        self._flush()
        m = self.cnt >= min_hits
        avg = self.sum[m] / self.cnt[m, None]
        return avg[:, :3], np.clip(avg[:, 3:], 0, 255).astype(np.uint8)


class TopDown:
    """World (aligned, z up) -> pixel mapping for the bird's-eye image.
    The map is rasterised at voxel resolution and then scaled up (nearest) for display."""

    def __init__(self, xy_all, voxel, max_px=1400, margin=1.5):
        lo = np.percentile(xy_all, 0.5, axis=0) - margin
        hi = np.percentile(xy_all, 99.5, axis=0) + margin
        span = np.maximum(hi - lo, 1.0)
        self.base_res = max(voxel, max(span) / max_px)
        self.W0 = int(np.ceil(span[0] / self.base_res)) + 1
        self.H0 = int(np.ceil(span[1] / self.base_res)) + 1
        self.scale = max(1, int(max_px // max(self.W0, self.H0)))
        self.res = self.base_res / self.scale                 # metres per display pixel
        self.lo = lo
        self.W, self.H = self.W0 * self.scale, self.H0 * self.scale

    def px(self, xy, base=False):
        xy = np.atleast_2d(xy)
        r, H = (self.base_res, self.H0) if base else (self.res, self.H)
        u = (xy[:, 0] - self.lo[0]) / r
        v = H - 1 - (xy[:, 1] - self.lo[1]) / r               # north up
        return np.stack([u, v], 1).round().astype(int)


def render_topdown(td, pts, cols, traj, humans, color_mode, zcut):
    img = np.full((td.H0, td.W0, 3), 38, np.uint8)
    if len(pts):
        m = pts[:, 2] < zcut
        p, c = pts[m], cols[m]
        order = np.argsort(p[:, 2])                         # draw low first, high last
        p, c = p[order], c[order]
        uv = td.px(p[:, :2], base=True)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < td.W0) & (uv[:, 1] >= 0) & (uv[:, 1] < td.H0)
        uv, p, c = uv[ok], p[ok], c[ok]
        if color_mode == "rgb":
            img[uv[:, 1], uv[:, 0]] = c[:, ::-1]            # RGB -> BGR
        elif len(p):
            z = p[:, 2]
            z2, z98 = np.percentile(z, [2, 98])
            zn = np.clip((z - z2) / max(z98 - z2, 0.1), 0, 1)
            cm = cv2.applyColorMap((zn * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
            img[uv[:, 1], uv[:, 0]] = cm
    img = cv2.resize(img, (td.W, td.H), interpolation=cv2.INTER_NEAREST)
    # grid every 1 m (5 m bold)
    for axis in (0, 1):
        g0 = np.ceil(td.lo[axis])
        g1 = td.lo[axis] + (td.W if axis == 0 else td.H) * td.res
        for g in np.arange(g0, g1, 1.0):
            if axis == 0:
                u = td.px([[g, td.lo[1]]])[0, 0]
                cv2.line(img, (u, 0), (u, td.H - 1), (70, 70, 70) if g % 5 else (110, 110, 110), 1)
            else:
                v = td.px([[td.lo[0], g]])[0, 1]
                cv2.line(img, (0, v), (td.W - 1, v), (70, 70, 70) if g % 5 else (110, 110, 110), 1)
    # trajectory
    if len(traj) > 1:
        uv = td.px(traj[:, :2])
        cv2.polylines(img, [uv.reshape(-1, 1, 2)], False, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(uv[0]), 7, (0, 200, 0), -1)
        cv2.putText(img, "START", tuple(uv[0] + [9, -9]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(uv[-1]), 7, (255, 255, 255), 2)
        cv2.putText(img, "END", tuple(uv[-1] + [9, -9]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # humans
    for h in humans:
        u, v = td.px(h["xyz"][:2])[0]
        cv2.circle(img, (u, v), 14, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(img, (u, v), 12, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, h["id"], (u + 16, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, h["id"], (u + 16, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    # scale bar + legend
    bar_m = 1 if td.W * td.res < 15 else 5 if td.W * td.res < 60 else 10
    bl = int(bar_m / td.res)
    y0 = td.H - 20
    cv2.rectangle(img, (15, y0 - 4), (15 + bl, y0 + 4), (255, 255, 255), -1)
    cv2.putText(img, f"{bar_m} m", (20 + bl, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"Humans found: {len(humans)}   grid = 1 m", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "yolov8n.onnx"))
    ap.add_argument("--backend", default="auto", choices=["auto", "ultralytics", "onnxruntime", "opencv"])
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--stride", type=int, default=2, help="run detector every Nth frame")
    ap.add_argument("--max-depth", type=float, default=8.0, help="max depth used to place humans (m)")
    ap.add_argument("--map-depth", type=float, default=5.0, help="max depth used to build the map (m)")
    ap.add_argument("--map-stride", type=int, default=6, help="use every Nth tracked frame for the dense map")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=1.5, help="detections closer than this = same person (m)")
    ap.add_argument("--min-obs", type=int, default=3, help="min sightings to confirm a human")
    ap.add_argument("--up", default="auto", help="auto | x | -x | y | -y | z | -z  (world 'up' axis)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    run = os.path.abspath(os.path.expanduser(args.run))
    out = os.path.join(run, "results")
    os.makedirs(os.path.join(out, "humans"), exist_ok=True)

    with open(os.path.join(run, "meta.json")) as f:
        meta = json.load(f)
    ci = meta["color_intrinsics"]
    K = (ci["fx"], ci["fy"], ci["ppx"], ci["ppy"])
    dscale = meta["depth_scale"]
    fps = meta.get("fps", 30)

    if not os.path.exists(os.path.join(run, "associations.txt")):
        sys.path.insert(0, HERE)
        from record import write_associations
        write_associations(run)
    frames = load_assoc(run)
    N = len(frames)
    print(f"[post] {N} frames")

    # ---------------------------------------------------------------- poses
    poses = load_trajectory(os.path.join(run, "CameraTrajectory.txt"))
    frame_pose = [poses.get(int(round(t * 1000))) for t, _, _ in frames]
    n_tracked = sum(p is not None for p in frame_pose)
    has_map = n_tracked > 10
    print(f"[post] SLAM poses for {n_tracked}/{N} frames ({100.0 * n_tracked / max(N, 1):.1f}%)")
    if not has_map:
        print("[post] WARNING: no usable CameraTrajectory.txt -> detection video only, no map.")

    # ---------------------------------------------------------------- gravity alignment
    up_w = None
    first_tracked = next((i for i, p in enumerate(frame_pose) if p is not None), None)
    if args.up != "auto":
        up_w = np.array(AXES[args.up], float)
        why = f"user --up {args.up}"
    elif has_map and meta.get("up_camera_frame") and frames[first_tracked][0] < 5.0:
        # gravity was measured at the start of recording (drone still); map starts right away
        R0 = frame_pose[first_tracked][0]
        up_w = R0 @ np.array(meta["up_camera_frame"], float)
        why = "IMU gravity at start"
    else:
        up_w = np.array([0, -1.0, 0])   # camera y axis points down
        why = "default (camera level at map start; use --up to override)"
    R_al = rot_a_to_b(up_w, np.array([0, 0, 1.0]))
    print(f"[post] up vector (SLAM world) = {np.round(up_w, 3)}  <- {why}")

    def to_world(i, p_cam):
        R, t = frame_pose[i]
        return R_al @ (R @ p_cam + t)

    traj = np.array([R_al @ p[1] for p in frame_pose if p is not None]) if has_map else np.zeros((0, 3))

    # ---------------------------------------------------------------- detection
    sys.path.insert(0, HERE)
    from detector import Detector
    det = Detector(args.model, conf=args.conf, backend=args.backend)
    dets = run_detection(det, run, frames, args.stride, os.path.join(out, "detections_cache.json"))
    n_det = sum(len(v) for v in dets.values())
    print(f"[post] {n_det} person detections in {len(dets)} frames")

    # ---------------------------------------------------------------- localisation
    records = []   # dict per detection
    for i in sorted(dets):
        depth = None
        for d in dets[i]:
            rec = dict(frame=i, t=frames[i][0], box=d[:4], conf=d[4], xyz=None, human=None)
            if has_map and frame_pose[i] is not None:
                if depth is None:
                    depth = cv2.imread(os.path.join(run, frames[i][2]), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    records.append(rec)
                    continue
                pc = localize(d, depth, K, dscale, args.max_depth)
                if pc is not None:
                    rec["xyz"] = to_world(i, pc)
            records.append(rec)
    loc = [r for r in records if r["xyz"] is not None]
    print(f"[post] {len(loc)} detections placed in the map")

    humans = []
    if loc:
        P = np.array([r["xyz"] for r in loc])
        clusters, cents = cluster(P, args.radius)
        good = [(c, cen) for c, cen in zip(clusters, cents) if len(c) >= args.min_obs]
        good.sort(key=lambda cc: min(loc[k]["t"] for k in cc[0]))
        for n, (c, cen) in enumerate(good):
            hid = f"H{n + 1}"
            members = [loc[k] for k in c]
            for m in members:
                m["human"] = hid
            best = max(members, key=lambda m: m["conf"] * min(1.0, (m["box"][3] - m["box"][1]) / 150.0))
            humans.append(dict(id=hid, xyz=np.median(P[c], axis=0), n_obs=len(c),
                               first_t=min(m["t"] for m in members), last_t=max(m["t"] for m in members),
                               max_conf=max(m["conf"] for m in members), best=best))
    print(f"[post] UNIQUE HUMANS: {len(humans)}")
    for h in humans:
        x, y, z = h["xyz"]
        print(f"   {h['id']}: x={x:6.2f} y={y:6.2f} z={z:5.2f} m   seen {h['n_obs']}x  "
              f"t={h['first_t']:.1f}-{h['last_t']:.1f}s  conf={h['max_conf']:.2f}")

    # snapshots
    for h in humans:
        b = h["best"]
        img = cv2.imread(os.path.join(run, frames[b["frame"]][1]))
        x1, y1, x2, y2 = map(int, b["box"])
        pad = 20
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].copy()
        cv2.imwrite(os.path.join(out, "humans", f"{h['id']}.jpg"), crop)
        full = img.copy()
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(full, f"{h['id']} {b['conf']:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out, "humans", f"{h['id']}_frame.jpg"), full)

    # tables
    with open(os.path.join(out, "humans.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_m", "y_m", "z_m", "sightings", "first_seen_s", "last_seen_s", "max_conf", "snapshot"])
        for h in humans:
            w.writerow([h["id"], *[f"{v:.3f}" for v in h["xyz"]], h["n_obs"], f"{h['first_t']:.2f}",
                        f"{h['last_t']:.2f}", f"{h['max_conf']:.3f}", f"humans/{h['id']}.jpg"])
    with open(os.path.join(out, "humans.json"), "w") as f:
        json.dump([dict(id=h["id"], x=float(h["xyz"][0]), y=float(h["xyz"][1]), z=float(h["xyz"][2]),
                        sightings=h["n_obs"], first_seen_s=h["first_t"], last_seen_s=h["last_t"],
                        max_conf=h["max_conf"]) for h in humans], f, indent=2)
    with open(os.path.join(out, "detections.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "conf", "x1", "y1", "x2", "y2", "x_m", "y_m", "z_m", "human"])
        for r in records:
            xyz = r["xyz"] if r["xyz"] is not None else [np.nan] * 3
            w.writerow([r["frame"], f"{r['t']:.3f}", f"{r['conf']:.3f}", *[int(v) for v in r["box"]],
                        *[f"{v:.3f}" for v in xyz], r["human"] or ""])

    # ---------------------------------------------------------------- dense map
    td, topdown = None, None
    if has_map:
        print("[post] building dense map...")
        vm = VoxelMap(args.voxel)
        H, W = ci["height"], ci["width"]
        step = 4
        vs, us = np.mgrid[0:H:step, 0:W:step]
        us, vs = us.ravel(), vs.ravel()
        rays = np.stack([(us - K[2]) / K[0], (vs - K[3]) / K[1], np.ones_like(us, float)], 1)
        tracked = [i for i, p in enumerate(frame_pose) if p is not None][::args.map_stride]
        for n, i in enumerate(tracked):
            depth = cv2.imread(os.path.join(run, frames[i][2]), cv2.IMREAD_UNCHANGED)
            rgb = cv2.imread(os.path.join(run, frames[i][1]))
            if depth is None or rgb is None:
                continue
            z = depth[vs, us].astype(np.float64) * dscale
            m = (z > 0.3) & (z < args.map_depth)
            pc = rays[m] * z[m, None]
            R, t = frame_pose[i]
            pw = (R_al @ (pc @ R.T + t).T).T
            vm.add(pw, rgb[vs[m], us[m]][:, ::-1])
            if n % 100 == 0:
                print(f"[post]   map frame {n}/{len(tracked)}", flush=True)
        mp, mc = vm.result(min_hits=2 if len(tracked) > 20 else 1)
        print(f"[post] map: {len(mp)} points")
        write_ply(os.path.join(out, "map.ply"), mp, mc)

        # map + markers
        extra_p, extra_c = [mp], [mc]
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        for h in humans:                                          # red pillar at each human
            c = h["xyz"]
            zz = np.arange(-0.3, 2.5, 0.02)
            A, Z = np.meshgrid(ang, zz)
            pil = np.stack([c[0] + 0.15 * np.cos(A.ravel()), c[1] + 0.15 * np.sin(A.ravel()), c[2] + Z.ravel()], 1)
            extra_p.append(pil)
            extra_c.append(np.tile([255, 0, 0], (len(pil), 1)))
        if len(traj):                                             # yellow flight path
            extra_p.append(traj)
            extra_c.append(np.tile([255, 220, 0], (len(traj), 1)))
        write_ply(os.path.join(out, "map_with_humans.ply"), np.vstack(extra_p), np.vstack(extra_c))

        # top-down
        xy_all = np.vstack([mp[:, :2], traj[:, :2]] + ([np.array([h["xyz"][:2] for h in humans])] if humans else []))
        td = TopDown(xy_all, args.voxel)
        zcut = (traj[:, 2].max() + 1.0) if len(traj) else 1e9
        topdown = render_topdown(td, mp, mc, traj, humans, "rgb", zcut)
        cv2.imwrite(os.path.join(out, "map_topdown.png"), topdown)
        cv2.imwrite(os.path.join(out, "map_topdown_height.png"),
                    render_topdown(td, mp, mc, traj, humans, "height", zcut))

    # ---------------------------------------------------------------- annotated video
    if not args.no_video:
        print("[post] rendering detection video...")
        H, W = ci["height"], ci["width"]
        vw = cv2.VideoWriter(os.path.join(out, "detections_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        by_frame = {}
        for r in records:
            by_frame.setdefault(r["frame"], []).append(r)
        mini, mini_scale = None, 1.0
        if topdown is not None:
            s = 200.0 / max(topdown.shape[:2])
            mini = cv2.resize(topdown, (int(topdown.shape[1] * s), int(topdown.shape[0] * s)), interpolation=cv2.INTER_AREA)
            mini_scale = s
        det_frames = sorted(by_frame)
        last_recs, last_i = [], -10 ** 9
        seen = set()
        for i, (t, rgbp, _) in enumerate(frames):
            img = cv2.imread(os.path.join(run, rgbp))
            if img is None:
                continue
            if i in by_frame:
                last_recs, last_i = by_frame[i], i
            elif i % args.stride == 0:
                last_recs = []
            recs = last_recs if i - last_i < args.stride else []
            for r in recs:
                x1, y1, x2, y2 = map(int, r["box"])
                col = (0, 0, 255) if r["human"] else (0, 200, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                lab = f"{r['human'] or 'person'} {r['conf']:.2f}"
                if r["xyz"] is not None:
                    lab += f" ({r['xyz'][0]:.1f},{r['xyz'][1]:.1f})m"
                cv2.putText(img, lab, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, lab, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                if r["human"]:
                    seen.add(r["human"])
            status = "SLAM OK" if frame_pose[i] is not None else "SLAM LOST"
            cv2.rectangle(img, (0, 0), (W, 24), (0, 0, 0), -1)
            cv2.putText(img, f"t={t:6.1f}s  frame {i}  {status}  humans so far: {len(seen)}", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if frame_pose[i] is not None else (0, 0, 255), 1,
                        cv2.LINE_AA)
            if mini is not None:
                m = mini.copy()
                if frame_pose[i] is not None:
                    pw = R_al @ frame_pose[i][1]
                    u, v = (td.px(pw[:2])[0] * mini_scale).astype(int)
                    cv2.circle(m, (u, v), 5, (0, 255, 255), -1)
                mh, mw = m.shape[:2]
                if mh < H - 30 and mw < W:
                    img[H - mh - 5:H - 5, W - mw - 5:W - 5] = m
                    cv2.rectangle(img, (W - mw - 6, H - mh - 6), (W - 4, H - 4), (255, 255, 255), 1)
            vw.write(img)
            if i % 500 == 0:
                print(f"[post]   video frame {i}/{N}", flush=True)
        vw.release()

    # ---------------------------------------------------------------- HTML report
    def b64(path):
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    rows = ""
    for h in humans:
        x, y, z = h["xyz"]
        rows += (f"<tr><td><b>{h['id']}</b></td><td>{x:.2f}</td><td>{y:.2f}</td><td>{z:.2f}</td>"
                 f"<td>{h['n_obs']}</td><td>{h['first_t']:.1f} – {h['last_t']:.1f}</td><td>{h['max_conf']:.2f}</td>"
                 f"<td><img src='data:image/jpeg;base64,{b64(os.path.join(out, 'humans', h['id'] + '.jpg'))}'></td>"
                 f"<td><img class='wide' src='data:image/jpeg;base64,{b64(os.path.join(out, 'humans', h['id'] + '_frame.jpg'))}'></td></tr>")
    maptag = (f"<img class='map' src='data:image/png;base64,{b64(os.path.join(out, 'map_topdown.png'))}'>"
              f"<img class='map' src='data:image/png;base64,{b64(os.path.join(out, 'map_topdown_height.png'))}'>"
              if has_map else "<p class='warn'>No SLAM trajectory – map not available.</p>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Flight report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;background:#16181c;color:#e8e8e8;margin:0 auto;padding:16px;max-width:1400px}}
h1{{margin:0 0 4px}} .sub{{color:#9aa}} .stats span{{display:inline-block;background:#23262c;border-radius:8px;padding:8px 12px;margin:4px 6px 4px 0}}
img.map{{max-width:49%;border-radius:8px;margin:4px 0}} table{{border-collapse:collapse;width:100%;margin-top:12px}}
td,th{{border-bottom:1px solid #333;padding:6px;text-align:left;vertical-align:middle}} td img{{max-height:140px;border-radius:6px}}
td img.wide{{max-height:140px}} .warn{{color:#fb4}} @media(max-width:800px){{img.map{{max-width:100%}}}}</style></head><body>
<h1>Flight report</h1><div class="sub">{os.path.basename(run)} · {meta.get('device', '')}</div>
<div class="stats"><span>Frames: {N}</span><span>SLAM tracked: {100.0 * n_tracked / max(N, 1):.0f}%</span>
<span>Person detections: {n_det}</span><span>Placed in map: {len(loc)}</span><span><b>Unique humans: {len(humans)}</b></span></div>
<h2>Map</h2>{maptag}
<p class="sub">Yellow = flight path, green = start, red = humans. Coordinates in metres from the map origin (x/y horizontal, z up). Up axis: {why}.</p>
<h2>Humans</h2><table><tr><th>ID</th><th>x (m)</th><th>y (m)</th><th>z (m)</th><th>Sightings</th><th>Seen (s)</th><th>Conf</th><th>Snapshot</th><th>Frame</th></tr>{rows}</table>
<p class="sub">Files: detections_video.mp4, map.ply, map_with_humans.ply (open in MeshLab/CloudCompare), humans.csv, detections.csv</p>
</body></html>"""
    with open(os.path.join(out, "report.html"), "w") as f:
        f.write(html)
    print(f"\n[post] DONE -> {out}/report.html")


if __name__ == "__main__":
    main()
