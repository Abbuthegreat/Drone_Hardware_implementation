"""
ORB + PnP RGB-D visual odometry with gyro rotation prior.

THIS is the SLAM backend that runs in this repository's Windows environment. It is NOT ORB-SLAM3 (which could not be built
here - see slam/orbslam3_backend.py). It follows the same input/output contract and the same ingredients the physical
navigation stack used (ORB features, depth-lifted 3-D points, PnP, inertial aiding):

  * ORB features (cv2), 3-D points from the registered depth image (metres) using the camera intrinsics
  * frame-to-keyframe tracking: descriptor matching (ratio test) -> solvePnPRansac -> LM refinement on the inliers
  * the gyro (integrated between the two image stamps) provides a rotation prior used to gate implausible PnP results
  * keyframes are inserted on translation / rotation / low overlap; the last N keyframes form the local map
  * relocalisation: when tracking is lost, the incoming frame is matched against the stored keyframe bank
No ground truth is used anywhere. Drift is uncorrected (no loop closure / bundle adjustment) - hence "odometry".
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from slam.backend import SlamBackend, SlamResult


def _rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(tvec).ravel()
    return T


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R.T, -R.T @ t
    return out


def _rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


@dataclass
class Keyframe:
    T_w_c: np.ndarray
    kps: np.ndarray            # (N,2) pixel coords
    desc: np.ndarray
    pts_c: np.ndarray          # (N,3) 3-D points in the keyframe camera frame
    t: float


class OrbPnpVO(SlamBackend):
    name = "orb_pnp_vo"

    def __init__(self, K, cfg):
        self.K = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1.0]])
        self.c = cfg
        self.orb = cv2.ORB_create(nfeatures=int(cfg["n_features"]), scaleFactor=1.2, nlevels=8, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    def reset(self):
        self.kfs: list[Keyframe] = []
        self.ref: Keyframe | None = None
        self.T_w_c = np.eye(4)
        self.state = "INIT"
        self.imu = []                  # (t, gyro_cam)
        self.last_t = None
        self.n_ok = self.n_lost = 0
        self.last_ok_t = None

    # ------------------------------------------------------------------ imu
    def feed_imu(self, t, gyro_cam, accel_cam):
        self.imu.append((t, np.asarray(gyro_cam, float)))
        if len(self.imu) > 4000:
            self.imu = self.imu[-2000:]

    def _gyro_rotation(self, t0, t1):
        """Integrated rotation of the camera between t0 and t1 (R_prev_cur), or None if there is no IMU coverage."""
        seg = [(t, w) for t, w in self.imu if t0 - 0.02 <= t <= t1 + 0.02]
        if len(seg) < 2 or seg[0][0] > t0 + 0.05 or seg[-1][0] < t1 - 0.05:
            return None
        R = np.eye(3)
        for (ta, wa), (tb, wb) in zip(seg[:-1], seg[1:]):
            a, b = max(ta, t0), min(tb, t1)
            if b <= a:
                continue
            dR, _ = cv2.Rodrigues(0.5 * (wa + wb) * (b - a))
            R = R @ dR
        return R

    # ------------------------------------------------------------------ features
    def _features(self, gray, depth):
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 20:
            return None
        uv = np.array([k.pt for k in kps], np.float32)
        u, v = np.clip(uv[:, 0].round().astype(int), 0, depth.shape[1] - 1), np.clip(uv[:, 1].round().astype(int), 0, depth.shape[0] - 1)
        z = depth[v, u]
        ok = np.isfinite(z) & (z > self.c["min_depth_m"]) & (z < self.c["max_depth_m"])
        if ok.sum() < 20:
            return None
        uv, desc, z = uv[ok], desc[ok], z[ok]
        x = (uv[:, 0] - self.K[0, 2]) * z / self.K[0, 0]
        y = (uv[:, 1] - self.K[1, 2]) * z / self.K[1, 1]
        return uv, desc, np.stack([x, y, z], 1)

    def _track_against(self, kf: Keyframe, uv, desc):
        m = self.bf.knnMatch(desc, kf.desc, k=2)
        good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < self.c["ratio_test"] * b.distance]
        if len(good) < self.c["min_inliers"]:
            return None
        obj = np.ascontiguousarray(kf.pts_c[[g.trainIdx for g in good]], np.float64)
        img = np.ascontiguousarray(uv[[g.queryIdx for g in good]], np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, iterationsCount=150,
                                                 reprojectionError=self.c["reproj_error_px"], confidence=0.99,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.c["min_inliers"]:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
        T_c_kf = _rvec_tvec_to_T(rvec, tvec)                       # X_c = T_c_kf * X_kf
        return T_c_kf, len(inl), len(good)

    # ------------------------------------------------------------------ main
    def process(self, t, bgr, depth_m) -> SlamResult:
        t0 = time.perf_counter()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        feats = self._features(gray, depth_m)
        info = {"features": 0 if feats is None else len(feats[0])}
        if feats is None:
            return self._lost(t, info, "feature-poor / invalid depth")
        uv, desc, pts = feats

        if self.ref is None:                                    # first frame: initialise the map at the identity pose
            self.T_w_c = np.eye(4) if not self.kfs else self.T_w_c
            self._add_keyframe(t, uv, desc, pts)
            self.state, self.last_t, self.last_ok_t = "OK", t, t
            info.update(inliers=len(uv), matches=len(uv), keyframes=len(self.kfs))
            return SlamResult(t, "INIT", self.T_w_c.copy(), info)

        res = None
        if self.state == "OK":
            res = self._track_against(self.ref, uv, desc)
        if res is None:                                          # relocalise against the keyframe bank (newest first)
            for kf in reversed(self.kfs[-12:]):
                if kf is self.ref and self.state == "OK":
                    continue
                r = self._track_against(kf, uv, desc)
                if r is not None:
                    res, self.ref = r, kf
                    info["relocalised"] = True
                    break
        if res is None:
            # re-initialise after a hover: while tracking is lost the mission commands zero velocity (the FC holds position), so after
            # `reinit_after_lost_s` the last accepted pose is still a valid anchor. Start a fresh keyframe there (e.g. the scene changed
            # permanently - a collapse - so the old keyframes no longer match). Reported in info["reinitialised"].
            ra = self.c.get("reinit_after_lost_s", 0.0)
            if ra > 0 and self.last_ok_t is not None and t - self.last_ok_t > ra and len(uv) >= self.c.get("reinit_min_features", 150):
                self._add_keyframe(t, uv, desc, pts)
                self.state, self.last_t, self.last_ok_t = "OK", t, t
                self.n_ok += 1
                self.n_reinit = getattr(self, "n_reinit", 0) + 1
                info.update(reinitialised=True, n_reinit=self.n_reinit, inliers=len(uv), keyframes=len(self.kfs))
                return SlamResult(t, "OK", self.T_w_c.copy(), info)
            return self._lost(t, info, "PnP failed")
        T_c_kf, n_inl, n_match = res
        T_new = self.ref.T_w_c @ _inv(T_c_kf)

        # IMU gating: the camera rotation implied by PnP must agree with the integrated gyro
        if self.c.get("imu_rotation_prior", True) and self.last_t is not None and self.state == "OK":
            Rg = self._gyro_rotation(self.last_t, t)
            if Rg is not None:
                dR_vo = self.T_w_c[:3, :3].T @ T_new[:3, :3]
                err = _rot_angle(Rg.T @ dR_vo)
                info["gyro_disagreement_deg"] = round(err, 2)
                if err > 12.0 and n_inl < 80:
                    return self._lost(t, info, f"rotation contradicts gyro ({err:.0f} deg)")

        # motion-consistency gating: a moving object filling the view (e.g. collapsing debris) can make PnP lock onto it and return a
        # translation the vehicle cannot physically have made. Such a pose is rejected (not used, no keyframe), the keyframes created just
        # before (which may contain the moving object) are discarded, and the next frames relocalise against the older keyframe bank.
        if self.last_ok_t is not None and self.ref is not None:
            dt = max(t - self.last_ok_t, 1e-3)
            jump = float(np.linalg.norm(T_new[:3, 3] - self.T_w_c[:3, 3]))
            limit = self.c.get("max_speed_mps", 2.0) * min(dt, 1.0) + self.c.get("jump_margin_m", 0.15)
            info["jump_m"] = round(jump, 3)
            if jump > limit:
                self._discard_recent_keyframes(self.last_ok_t - self.c.get("jump_purge_window_s", 0.5))
                return self._lost(t, info, f"pose jump {jump:.2f} m in {dt:.2f} s exceeds {limit:.2f} m")

        self.T_w_c = T_new
        self.state, self.last_t, self.last_ok_t = "OK", t, t
        self.n_ok += 1
        # keyframe policy
        dT = _inv(self.ref.T_w_c) @ T_new
        trans, rot = float(np.linalg.norm(dT[:3, 3])), _rot_angle(dT[:3, :3])
        ratio = n_inl / max(len(uv), 1)
        if trans > self.c["keyframe_min_translation_m"] or rot > self.c["keyframe_min_rotation_deg"] or ratio < self.c["keyframe_min_inlier_ratio"] * 0.35:
            self._add_keyframe(t, uv, desc, pts)
        info.update(inliers=n_inl, matches=n_match, keyframes=len(self.kfs), dt_ms=round((time.perf_counter() - t0) * 1e3, 1))
        return SlamResult(t, "OK", self.T_w_c.copy(), info)

    def _add_keyframe(self, t, uv, desc, pts):
        kf = Keyframe(self.T_w_c.copy(), uv, desc, pts, t)
        self.kfs.append(kf)
        self.ref = kf
        if len(self.kfs) > 200:
            self.kfs = self.kfs[-200:]

    def _discard_recent_keyframes(self, t_from):
        keep = [kf for kf in self.kfs if kf.t < t_from]
        if keep and len(keep) < len(self.kfs):
            self.kfs = keep
            self.ref = keep[-1]

    def _lost(self, t, info, why):
        self.state = "LOST"
        self.n_lost += 1
        info["reason"] = why
        return SlamResult(t, "LOST", None, info)
