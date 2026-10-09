"""
Simulated IMU, GPS and the (simplified) FC state estimate.

IMU model (per axis, body FRD frame, sampled at `sensors.imu.rate_hz`)
----------------------------------------------------------------------
    gyro_meas  = clip( w_true + b_g + n_g )
    accel_meas = clip( f_true + b_a + n_a )
    f_true     = R_frd_ned ( a_ned_at_IMU - g_ned )         specific force, g_ned = (0,0,+g)
    a_ned_at_IMU = a_cg + alpha x r + w x (w x r)           lever arm r = IMU position - CG
    a_cg, alpha = finite differences of the physics engine's CG velocity / angular velocity
                  over one IMU period
    n_x  ~ N(0, (density^2) * rate)                         discrete white noise from the noise density
    b_x  : b(0) ~ N(0, sigma_init^2),  b[k+1] = b[k] + N(0, rw^2 * dt)   bias random walk
No scale-factor, misalignment, temperature or vibration model.

GPS model
---------
    position error = first-order Gauss-Markov (tau = correlation_time) + white part, per axis
    reported lat/lon/alt = GeoFrame.ned_to_geodetic(true_ned + error)
    velocity error = white noise. Dropouts / degraded mode are configurable (`sensors.gps`).

The estimator/`FcEstimate` gives the FC ground truth + small configured noise (its EKF is NOT modelled).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from common.frames import euler_from_R
from common.geo import GeoFrame

G_NED = None  # set per-config in ImuModel


@dataclass
class ImuSample:
    t: float                # simulation time [s]
    gyro: np.ndarray        # rad/s, FRD
    accel: np.ndarray       # m/s^2, FRD (specific force)


class ImuModel:
    def __init__(self, cfg, model, dt_imu):
        c = cfg["sensors"]["imu"]
        self.c = c
        self.dt = dt_imu
        self.g = float(cfg["vehicle"]["gravity_mps2"])
        self.r = np.array([model.imu_pos_flu[0], -model.imu_pos_flu[1], -model.imu_pos_flu[2]])  # FLU->FRD
        self.rng = np.random.default_rng(c["seed"])
        self.bg = self.rng.normal(0, c["gyro_bias_initial_sigma"], 3)
        self.ba = self.rng.normal(0, c["accel_bias_initial_sigma"], 3)
        self.sg = c["gyro_noise_density"] * math.sqrt(c["rate_hz"])
        self.sa = c["accel_noise_density"] * math.sqrt(c["rate_hz"])
        self.prev_v = None
        self.prev_w = None
        self.last_true_specific_force = np.zeros(3)

    def sample(self, st) -> ImuSample:
        """st: TrueState at the IMU sample time."""
        c, dt = self.c, self.dt
        if self.prev_v is None:
            a_cg = np.zeros(3)
            alpha = np.zeros(3)
        else:
            a_cg = (st.vel_ned - self.prev_v) / dt
            alpha = (st.omega_frd - self.prev_w) / dt
        self.prev_v, self.prev_w = st.vel_ned.copy(), st.omega_frd.copy()
        w = st.omega_frd
        a_body = st.R_ned_frd.T @ a_cg + np.cross(alpha, self.r) + np.cross(w, np.cross(w, self.r))
        f_true = a_body - st.R_ned_frd.T @ np.array([0.0, 0.0, self.g])
        self.last_true_specific_force = f_true
        self.bg += self.rng.normal(0, c["gyro_bias_random_walk"] * math.sqrt(dt), 3)
        self.ba += self.rng.normal(0, c["accel_bias_random_walk"] * math.sqrt(dt), 3)
        gyro = np.clip(w + self.bg + self.rng.normal(0, self.sg, 3), -c["gyro_range_rps"], c["gyro_range_rps"])
        acc = np.clip(f_true + self.ba + self.rng.normal(0, self.sa, 3), -c["accel_range_mps2"], c["accel_range_mps2"])
        return ImuSample(st.t, gyro, acc)


@dataclass
class GpsSample:
    t: float
    fix_type: int           # 0 no fix, 3 3D fix
    lat_deg: float
    lon_deg: float
    alt_m: float
    vel_ned: np.ndarray
    eph_m: float
    epv_m: float
    hdop: float
    satellites: int


class GpsModel:
    def __init__(self, cfg, model, geo: GeoFrame):
        c = cfg["sensors"]["gps"]
        self.c = c
        self.geo = geo
        self.dt = 1.0 / c["rate_hz"]
        self.rng = np.random.default_rng(c["seed"])
        self.err = np.zeros(3)
        self.gps_pos_flu = model.gps_pos_flu
        self.mode = c["mode"]
        self.dropouts = [tuple(d) for d in c["dropouts"]]

    def set_mode(self, mode):
        assert mode in ("healthy", "degraded", "denied")
        self.mode = mode

    def sample(self, st) -> GpsSample:
        c = self.c
        t = st.t
        denied = self.mode == "denied" or any(a <= t <= b for a, b in self.dropouts)
        if denied:
            return GpsSample(t, 0, 0.0, 0.0, 0.0, np.zeros(3), 99.0, 99.0, 99.0, 0)
        scale = c["degraded_sigma_scale"] if self.mode == "degraded" else 1.0
        sats = c["degraded_satellites"] if self.mode == "degraded" else c["satellites"]
        # Gauss-Markov position error (stationary std = sigma), white velocity noise
        a = math.exp(-self.dt / c["correlation_time_s"])
        sig = np.array([c["horizontal_sigma_m"], c["horizontal_sigma_m"], c["vertical_sigma_m"]]) * scale
        self.err = a * self.err + math.sqrt(1 - a * a) * self.rng.normal(0, 1, 3) * sig
        pos_ned = st.pos_ned + st.R_ned_frd @ np.array([self.gps_pos_flu[0], -self.gps_pos_flu[1], -self.gps_pos_flu[2]]) \
            - st.R_ned_frd @ np.zeros(3)
        # antenna offset from CG is applied in the body frame (lever arm), then measurement error
        lat, lon, alt = self.geo.ned_to_geodetic(pos_ned + self.err)
        vel = st.vel_ned + self.rng.normal(0, c["velocity_sigma_mps"] * scale, 3)
        return GpsSample(t, 3, lat, lon, alt, vel, c["horizontal_sigma_m"] * scale, c["vertical_sigma_m"] * scale,
                         c["hdop"] * scale, sats)


class FcEstimator:
    """Ground truth + small configured errors, standing in for the FC's on-board EKF."""

    def __init__(self, cfg, seed=7):
        e = cfg["flight_controller"]["estimator"]
        self.e = e
        self.rng = np.random.default_rng(seed)

    def estimate(self, st, gyro_meas):
        from sim.flight_controller_sim import Estimate
        e = self.e
        R = st.R_ned_frd
        if e["att_sigma_rad"] > 0:
            dth = self.rng.normal(0, e["att_sigma_rad"], 3)
            K = np.array([[0, -dth[2], dth[1]], [dth[2], 0, -dth[0]], [-dth[1], dth[0], 0]])
            R = R @ (np.eye(3) + K)
            u, _, vt = np.linalg.svd(R)
            R = u @ vt
        return Estimate(st.pos_ned + self.rng.normal(0, e["pos_sigma_m"], 3),
                        st.vel_ned + self.rng.normal(0, e["vel_sigma_mps"], 3), R, gyro_meas)
