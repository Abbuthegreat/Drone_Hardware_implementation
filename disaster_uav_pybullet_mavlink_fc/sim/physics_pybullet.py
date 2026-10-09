"""
PyBullet rigid-body vehicle. This module is the ONLY place where PyBullet's ENU/FLU world and the
flight controller's NED/FRD world meet: state is converted to NED/FRD on the way out, and rotor
thrusts (already in newtons per motor) are turned into PyBullet forces/torques on the way in.

The URDF is loaded with URDF_MERGE_FIXED_LINKS (PyBullet then integrates the exact composite body);
its diagonal inertia is overridden from config/drone.yaml because the merged value PyBullet computes
does not match the analytic composite (verified in calibration/verify_inertia.py). The small Ixz
product (~1% cross-coupling) is neglected by the engine.

Forces per step:
  * thrust  T_i  along +z_body (ENU/FLU) applied at rotor i's position,
  * yaw reaction torque  -s_i * kQ w_i^2  about body z (s_i=+1 for a CCW-seen-from-above propeller),
  * body drag (linear+quadratic in air-relative velocity) at the CG, angular drag,
  * gravity, contacts: PyBullet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p

from common.frames import (C_NED_ENU, R_ned_frd_from_enu_flu, enu_to_ned, flu_to_frd, euler_from_R)
from sim.config import inertia_matrix, DEFAULT_URDF
from sim.vehicle_model import VehicleModel


@dataclass
class TrueState:
    """Ground truth. EVALUATION / FC-INTERNAL ONLY. Autonomy must never see this."""
    t: float
    pos_enu: np.ndarray
    vel_enu: np.ndarray
    R_enu_flu: np.ndarray
    omega_flu_body: np.ndarray
    # NED/FRD versions (what the FC works in)
    pos_ned: np.ndarray
    vel_ned: np.ndarray
    R_ned_frd: np.ndarray
    omega_frd: np.ndarray

    @property
    def euler_ned(self):
        return euler_from_R(self.R_ned_frd)


class PyBulletVehicle:
    def __init__(self, client: int, cfg: dict, model: VehicleModel | None = None, urdf: str = DEFAULT_URDF,
                 pos_enu=(0.0, 0.0, 1.0), yaw_ned_deg=0.0):
        self.pb = client
        self.cfg = cfg
        self.model = model or VehicleModel(urdf, cfg)
        aero = cfg["aero"]
        self.aero = aero
        self.wind_enu = enu_to_ned(aero["wind_ned_mps"])          # ned<->enu map is an involution
        self.dt = 1.0 / cfg["sim"]["physics_hz"]
        self.t = 0.0

        yaw_enu = math.pi / 2 - math.radians(yaw_ned_deg)
        orn = p.getQuaternionFromEuler([0, 0, yaw_enu])
        self.body = p.loadURDF(urdf, list(pos_enu), orn, flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MERGE_FIXED_LINKS,
                               physicsClientId=client)
        # inertial frame of the merged base == CG; convert CG pose <-> base_link-axes pose in read_state
        info = p.getDynamicsInfo(self.body, -1, physicsClientId=client)
        self.local_inertial_pos = np.array(info[3])
        self.local_inertial_R = np.array(p.getMatrixFromQuaternion(info[4])).reshape(3, 3)
        I = inertia_matrix(cfg)
        p.changeDynamics(self.body, -1, mass=float(cfg["vehicle"]["mass_kg"]),
                         localInertiaDiagonal=[float(I[0, 0]), float(I[1, 1]), float(I[2, 2])],
                         linearDamping=0.0, angularDamping=0.0, lateralFriction=0.8,
                         restitution=0.0, contactStiffness=8000.0, contactDamping=150.0, physicsClientId=client)
        self.thrusts = np.zeros(4)
        self.torque_reaction = np.zeros(4)

    # ------------------------------------------------------------------ state
    def read_state(self) -> TrueState:
        pos, orn = p.getBasePositionAndOrientation(self.body, physicsClientId=self.pb)
        lin, ang = p.getBaseVelocity(self.body, physicsClientId=self.pb)
        R_in = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        R = R_in @ self.local_inertial_R.T                 # body(base_link)-axes to ENU
        pos = np.array(pos)                                # CG
        vel = np.array(lin)
        w_body = R.T @ np.array(ang)
        return TrueState(self.t, pos, vel, R, w_body,
                         enu_to_ned(pos), enu_to_ned(vel), R_ned_frd_from_enu_flu(R), flu_to_frd(w_body))

    # ------------------------------------------------------------------ forces
    def apply_rotor_forces(self, thrusts, reaction_torques, st: TrueState | None = None):
        """thrusts / reaction_torques: per motor magnitudes [N], [N*m] from the motor model."""
        st = st or self.read_state()
        R = st.R_enu_flu
        z_w = R[:, 2]
        self.thrusts = np.asarray(thrusts, float)
        self.torque_reaction = np.asarray(reaction_torques, float)
        for i in range(4):
            point = st.pos_enu + R @ self.model.rotor_pos_flu[i]
            p.applyExternalForce(self.body, -1, (z_w * self.thrusts[i]).tolist(), point.tolist(), p.WORLD_FRAME,
                                 physicsClientId=self.pb)
        # reaction torque on the airframe is opposite to the propeller spin
        tz = float(np.sum(-self.model.spin * self.torque_reaction))
        # aerodynamic drag
        a = self.aero
        v_air = R.T @ (st.vel_enu - self.wind_enu)
        vxy = np.linalg.norm(v_air[:2])
        F = np.array([-(a["drag_linear_xy"] + a["drag_quad_xy"] * vxy) * v_air[0],
                      -(a["drag_linear_xy"] + a["drag_quad_xy"] * vxy) * v_air[1],
                      -(a["drag_linear_z"] + a["drag_quad_z"] * abs(v_air[2])) * v_air[2]])
        w = st.omega_flu_body
        tau_drag = -(a["ang_drag_linear"] + a["ang_drag_quad"] * np.linalg.norm(w)) * w
        p.applyExternalForce(self.body, -1, (R @ F).tolist(), st.pos_enu.tolist(), p.WORLD_FRAME, physicsClientId=self.pb)
        p.applyExternalTorque(self.body, -1, (R @ (tau_drag + np.array([0.0, 0.0, tz]))).tolist(), p.WORLD_FRAME,
                              physicsClientId=self.pb)

    def step(self):
        p.stepSimulation(physicsClientId=self.pb)
        self.t += self.dt

    def contacts_with_world(self):
        return p.getContactPoints(bodyA=self.body, physicsClientId=self.pb)

    def reset(self, pos_enu, yaw_ned_deg=0.0):
        orn = p.getQuaternionFromEuler([0, 0, math.pi / 2 - math.radians(yaw_ned_deg)])
        p.resetBasePositionAndOrientation(self.body, list(pos_enu), orn, physicsClientId=self.pb)
        p.resetBaseVelocity(self.body, [0, 0, 0], [0, 0, 0], physicsClientId=self.pb)
