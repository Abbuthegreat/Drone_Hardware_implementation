"""
Vehicle geometry parsed from the URDF (single source of truth for all body-frame transforms).

Everything the FC, IMU, GPS and camera need (rotor positions, sensor mounts, CG) is read from the
URDF XML rather than duplicated as constants. All vectors are expressed in the body axes (URDF
base_link axes) with the ORIGIN AT THE COMPOSITE CENTRE OF GRAVITY unless stated otherwise.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from common.frames import R_from_euler, flu_to_frd


def _vec(s, n=3):
    return np.array([float(x) for x in s.split()], dtype=float)


class VehicleModel:
    def __init__(self, urdf_path: str, cfg: dict):
        self.urdf_path = urdf_path
        self.cfg = cfg
        root = ET.parse(urdf_path).getroot()
        self.links = {}
        parent_of = {}
        for l in root.findall("link"):
            inr = l.find("inertial")
            m, I = 0.0, np.zeros((3, 3))
            if inr is not None:
                m = float(inr.find("mass").get("value"))
                a = inr.find("inertia").attrib
                I = np.array([[a["ixx"], a["ixy"], a["ixz"]],
                              [a["ixy"], a["iyy"], a["iyz"]],
                              [a["ixz"], a["iyz"], a["izz"]]], dtype=float)
                org = inr.find("origin")
                if org is not None and np.linalg.norm(_vec(org.get("xyz", "0 0 0"))) > 1e-12:
                    raise ValueError(f"link {l.get('name')}: non-zero inertial origin not supported")
            self.links[l.get("name")] = {"mass": m, "I": I}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = _vec(o.get("xyz", "0 0 0"))
            rpy = _vec(o.get("rpy", "0 0 0"))
            parent_of[j.find("child").get("link")] = (j.find("parent").get("link"), xyz, R_from_euler(*rpy), j.get("type"))
        if any(t != "fixed" for *_, t in parent_of.values()):
            raise ValueError("only fixed joints are supported")
        # pose of every link in the base_link frame
        self.T = {"base_link": (np.zeros(3), np.eye(3))}

        def pose(name):
            if name in self.T:
                return self.T[name]
            par, xyz, R, _ = parent_of[name]
            pp, pR = pose(par)
            self.T[name] = (pp + pR @ xyz, pR @ R)
            return self.T[name]

        for n in self.links:
            pose(n)

        self.total_mass = sum(l["mass"] for l in self.links.values())
        self.cg_base = sum(self.links[n]["mass"] * self.T[n][0] for n in self.links) / self.total_mass
        I = np.zeros((3, 3))
        for n, l in self.links.items():
            p, R = self.T[n]
            r = p - self.cg_base
            I += R @ l["I"] @ R.T + l["mass"] * (r @ r * np.eye(3) - np.outer(r, r))
        self.inertia_cg = I

        # ---- rotors (index 1..4), positions relative to CG, FLU
        self.motor_cfg = sorted(cfg["components"]["motors"], key=lambda m: m["index"])
        self.rotor_pos_flu = np.array([self.T[f"rotor_{m['index']}_link"][0] - self.cg_base for m in self.motor_cfg])
        self.rotor_pos_frd = np.array([flu_to_frd(p) for p in self.rotor_pos_flu])
        self.spin = np.array([1.0 if m["spin"] == "CCW" else -1.0 for m in self.motor_cfg])  # +1 = CCW seen from above

        # ---- sensor mounts relative to CG (FLU) and orientation of their link frame in the body frame
        self.imu_pos_flu = self.T["flight_controller_link"][0] - self.cg_base
        self.gps_pos_flu = self.T["gps_link"][0] - self.cg_base
        self.cam_pos_flu = self.T["rgbd_camera_link"][0] - self.cg_base
        self.cam_R_body_link = self.T["rgbd_camera_link"][1]
        self.cam_optical_pos_flu = self.T["rgbd_camera_optical_link"][0] - self.cg_base
        self.cam_R_body_optical = self.T["rgbd_camera_optical_link"][1]
        if "thermal_camera_link" in self.T:
            self.thermal_link_pos_flu = self.T["thermal_camera_link"][0] - self.cg_base
            from common.frames import R_LINK_OPTICAL
            self.thermal_R_body_optical = self.T["thermal_camera_link"][1] @ R_LINK_OPTICAL
            self.thermal_optical_pos_flu = self.thermal_link_pos_flu

    # ---- convenience for tests / calibration
    def component_positions(self):
        return {n: self.T[n][0] for n in self.links}
