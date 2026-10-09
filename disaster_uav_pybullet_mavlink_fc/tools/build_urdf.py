"""
Generate urdf/disaster_uav.urdf from config/drone.yaml `components`, and write the derived
`mass_properties` block (CG + full inertia tensor) back into drone.yaml.

Inertia method (engineering approach A): every component is a primitive (box / cylinder / thin rod /
thin disc) with its own analytic inertia; the composite tensor about the composite CG is obtained with
the parallel-axis theorem. Nothing is copied from another drone.

    python -m tools.build_urdf            # regenerate URDF and yaml derived block
    python -m tools.build_urdf --check    # exit 1 if the files on disk are stale
"""
from __future__ import annotations

import math
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.frames import R_from_euler, R_LINK_OPTICAL  # noqa: E402

CFG_PATH = os.path.join(ROOT, "config", "drone.yaml")
URDF_PATH = os.path.join(ROOT, "urdf", "disaster_uav.urdf")


# ------------------------------------------------------------------ primitive inertias (about own CoM)
def I_box(m, s):
    x, y, z = s
    return m / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def I_cyl_z(m, r, h):
    return np.diag([m * (3 * r * r + h * h) / 12.0, m * (3 * r * r + h * h) / 12.0, 0.5 * m * r * r])


def I_rod(m, a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    L = np.linalg.norm(d)
    u = d / L
    return m * L * L / 12.0 * (np.eye(3) - np.outer(u, u))


def shift(I, m, r):
    r = np.asarray(r, float)
    return I + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def rpy_of_R(R):
    pitch = -math.asin(max(-1, min(1, R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


# ------------------------------------------------------------------ build
def load_cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parts(cfg, battery_x):
    """Return list of dicts: name, m, xyz, R (link->base), I_link (about own CoM, link axes)."""
    c = cfg["components"]
    parts = []
    # airframe (base_link): one lumped part whose CoM is the mass centroid of primitives
    prim_m, prim_c, prim_I = [], [], []
    for p in c["airframe_primitives"]:
        if p["kind"] == "box":
            xyz = np.array(p["xyz"], float); I = I_box(p["mass_kg"], p["size_m"])
        else:
            a, b = np.array(p["from_xyz"], float), np.array(p["to_xyz"], float)
            xyz = 0.5 * (a + b); I = I_rod(p["mass_kg"], a, b)
        prim_m.append(p["mass_kg"]); prim_c.append(xyz); prim_I.append(I)
    M = sum(prim_m)
    com = sum(m * x for m, x in zip(prim_m, prim_c)) / M
    Ib = sum(shift(I, m, x - com) for I, m, x in zip(prim_I, prim_m, prim_c))
    parts.append(dict(name="airframe_link", m=M, xyz=com, R=np.eye(3), I=Ib))
    bl = c["base_link"]
    parts.append(dict(name="base_link", m=bl["mass_kg"], xyz=np.zeros(3), R=np.eye(3), I=I_box(bl["mass_kg"], bl["size_m"])))

    for name in ("flight_controller_link", "raspberry_pi_link", "esc_link", "battery_link", "rgbd_camera_link", "thermal_camera_link"):
        d = c[name]
        xyz = list(d["xyz"])
        if xyz[0] == "auto":
            xyz[0] = battery_x
        if "rpy_deg" in d:
            R = R_from_euler(*[math.radians(a) for a in d["rpy_deg"]])
        else:
            R = R_from_euler(*d.get("rpy", [0, 0, 0]))
        parts.append(dict(name=name, m=d["mass_kg"], xyz=np.array(xyz, float), R=R,
                          I=I_box(d["mass_kg"], d["size_m"]), size=d["size_m"]))
    g = c["gps_link"]
    parts.append(dict(name="gps_link", m=g["mass_kg"], xyz=np.array(g["xyz"], float), R=np.eye(3),
                      I=I_cyl_z(g["mass_kg"], g["radius_m"], g["length_m"])))
    mo, ro = c["motor"], c["rotor"]
    for mt in c["motors"]:
        i = mt["index"]; xy = mt["xy"]
        parts.append(dict(name=f"motor_{i}_link", m=mo["mass_kg"], xyz=np.array([xy[0], xy[1], 0.0]), R=np.eye(3),
                          I=I_cyl_z(mo["mass_kg"], mo["radius_m"], mo["length_m"])))
        Ir = I_cyl_z(ro["mass_kg"], ro["radius_m"], ro["thickness_m"])
        parts.append(dict(name=f"rotor_{i}_link", m=ro["mass_kg"],
                          xyz=np.array([xy[0], xy[1], ro["z_above_motor_m"]]), R=np.eye(3), I=Ir))
    return parts


def composite(parts):
    M = sum(p["m"] for p in parts)
    cg = sum(p["m"] * p["xyz"] for p in parts) / M
    I = np.zeros((3, 3))
    for p in parts:
        Ib = p["R"] @ p["I"] @ p["R"].T
        I += shift(Ib, p["m"], p["xyz"] - cg)
    return M, cg, I


def solve_battery_x(cfg):
    x = 0.0
    for _ in range(3):  # CG_x is linear in battery x -> converges in one step
        _, cg, _ = composite(build_parts(cfg, x))
        x -= cg[0] * (sum(p["m"] for p in build_parts(cfg, x)) / cfg["components"]["battery_link"]["mass_kg"])
    return round(x, 6)


def f6(v):
    return f"{v:.6f}".rstrip("0").rstrip(".") if abs(v) > 1e-12 else "0"


def inertial_xml(m, I, origin=(0, 0, 0)):
    return (f'    <inertial>\n      <origin xyz="{" ".join(f6(a) for a in origin)}" rpy="0 0 0"/>\n'
            f'      <mass value="{m:.6f}"/>\n'
            f'      <inertia ixx="{I[0,0]:.9f}" ixy="{I[0,1]:.9f}" ixz="{I[0,2]:.9f}" '
            f'iyy="{I[1,1]:.9f}" iyz="{I[1,2]:.9f}" izz="{I[2,2]:.9f}"/>\n    </inertial>\n')


def box_geom(s):
    return f'<geometry><box size="{" ".join(f6(a) for a in s)}"/></geometry>'


def make_urdf(cfg, bx):
    c = cfg["components"]
    parts = {p["name"]: p for p in build_parts(cfg, bx)}
    out = ['<?xml version="1.0"?>\n<robot name="disaster_response_uav">\n',
           "  <!-- GENERATED by tools/build_urdf.py from config/drone.yaml. Do not hand-edit.\n"
           "       Body frame = FLU (x forward, y left, z up). All joints are fixed; transforms are explicit.\n"
           "       Motor numbering (PX4 quad-X): 1 front-right CCW, 2 rear-left CCW, 3 front-left CW, 4 rear-right CW.\n"
           "       Placeholder masses/dimensions are listed in config/drone.yaml. Propellers are visual only. -->\n"]

    def joint(name, parent, child, xyz, rpy=(0, 0, 0)):
        return (f'  <joint name="{name}" type="fixed">\n    <parent link="{parent}"/><child link="{child}"/>\n'
                f'    <origin xyz="{" ".join(f6(a) for a in xyz)}" rpy="{" ".join(f6(a) for a in rpy)}"/>\n  </joint>\n')

    # ---- base link: visuals for all primitives, collisions for plates/gear
    bb = parts["base_link"]
    out.append('  <link name="base_link">\n' + inertial_xml(bb["m"], bb["I"]) + "  </link>\n")
    b = parts["airframe_link"]
    com = b["xyz"]
    s = '  <link name="airframe_link">\n' + inertial_xml(b["m"], b["I"])  # frame IS the CoM (PyBullet merge ignores inertial offsets)
    for p in c["airframe_primitives"]:
        if p["kind"] == "box":
            xyz = np.array(p["xyz"], float) - com
            s += f'    <visual><origin xyz="{" ".join(f6(a) for a in xyz)}"/>{box_geom(p["size_m"])}' \
                 f'<material name="mat_{p["name"]}"><color rgba="{"0.92 0.92 0.92 1" if p["name"].startswith("plate") else "0.10 0.10 0.11 1"}"/></material></visual>\n'
            if p["name"].startswith(("plate", "skid", "strut")):
                s += f'    <collision><origin xyz="{" ".join(f6(a) for a in xyz)}"/>{box_geom(p["size_m"])}</collision>\n'
        else:
            a, e = np.array(p["from_xyz"], float), np.array(p["to_xyz"], float)
            mid, d = 0.5 * (a + e) - com, e - a
            L = np.linalg.norm(d); u = d / L
            z = np.array([0, 0, 1.0]); ax = np.cross(z, u)
            ang = math.acos(max(-1, min(1, float(z @ u))))
            n = np.linalg.norm(ax)
            if n < 1e-9:
                R = np.eye(3)
            else:
                ax /= n; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
            r, pch, y = rpy_of_R(R)
            s += (f'    <visual><origin xyz="{" ".join(f6(v) for v in mid)}" rpy="{f6(r)} {f6(pch)} {f6(y)}"/>'
                  f'<geometry><cylinder radius="0.008" length="{L:.6f}"/></geometry>'
                  f'<material name="mat_{p["name"]}"><color rgba="0.95 0.45 0.05 1"/></material></visual>\n')
    s += "  </link>\n" + joint("airframe_joint", "base_link", "airframe_link", com)
    out.append(s)

    def box_link(name, color):
        p = parts[name]
        return (f'  <link name="{name}">\n' + inertial_xml(p["m"], p["I"]) +
                f'    <visual>{box_geom(p["size"])}<material name="{name}_m"><color rgba="{color}"/></material></visual>\n'
                f'  </link>\n' +
                joint(name.replace("_link", "_joint"), "base_link", name, p["xyz"], rpy_of_R(p["R"])))

    out.append(box_link("flight_controller_link", "0.18 0.25 0.22 1"))
    out.append(box_link("raspberry_pi_link", "0.12 0.32 0.12 1"))
    out.append(box_link("esc_link", "0.20 0.20 0.45 1"))
    out.append(box_link("battery_link", "0.09 0.09 0.09 1"))
    out.append(box_link("rgbd_camera_link", "0.10 0.10 0.10 1"))
    out.append(box_link("thermal_camera_link", "0.55 0.10 0.10 1"))

    # optical frame: z forward, x right, y down (massless frame link)
    ro = rpy_of_R(R_LINK_OPTICAL)
    out.append('  <link name="rgbd_camera_optical_link">\n    <inertial><mass value="0"/>'
               '<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>\n  </link>\n' +
               joint("rgbd_camera_optical_joint", "rgbd_camera_link", "rgbd_camera_optical_link", (0, 0, 0), ro))
    g = c["gps_link"]; p = parts["gps_link"]
    out.append('  <link name="gps_link">\n' + inertial_xml(p["m"], p["I"]) +
               f'    <visual><geometry><cylinder radius="{g["radius_m"]}" length="{g["length_m"]}"/></geometry>'
               '<material name="gps_m"><color rgba="0.12 0.12 0.12 1"/></material></visual>\n  </link>\n' +
               joint("gps_joint", "base_link", "gps_link", p["xyz"]))

    mo, rt = c["motor"], c["rotor"]
    for mt in c["motors"]:
        i = mt["index"]; pm = parts[f"motor_{i}_link"]
        out.append(f'  <link name="motor_{i}_link">\n' + inertial_xml(pm["m"], pm["I"]) +
                   f'    <visual><geometry><cylinder radius="{mo["radius_m"]}" length="{mo["length_m"]}"/></geometry>'
                   f'<material name="motor{i}"><color rgba="0.08 0.08 0.08 1"/></material></visual>\n'
                   f'    <collision><geometry><cylinder radius="{mo["radius_m"]}" length="{mo["length_m"]}"/></geometry></collision>\n'
                   f'  </link>\n' + joint(f"motor_{i}_joint", "base_link", f"motor_{i}_link", pm["xyz"]))
        pr = parts[f"rotor_{i}_link"]
        col = "0.8 0.2 0.2 0.45" if mt["spin"] == "CCW" else "0.2 0.4 0.8 0.45"
        out.append(f'  <link name="rotor_{i}_link"><!-- spin {mt["spin"]} seen from above -->\n' + inertial_xml(pr["m"], pr["I"]) +
                   f'    <visual><geometry><cylinder radius="{rt["radius_m"]}" length="{rt["thickness_m"]}"/></geometry>'
                   f'<material name="rotor{i}"><color rgba="{col}"/></material></visual>\n  </link>\n' +
                   joint(f"rotor_{i}_joint", f"motor_{i}_link", f"rotor_{i}_link", (0, 0, rt["z_above_motor_m"])))
    out.append("</robot>\n")
    return "".join(out)


def sensor_extrinsics(cfg, bx):
    """Static sensor poses in the body frame (FLU, origin at the CG): calibration data the autonomy stack may read."""
    parts = {p["name"]: p for p in build_parts(cfg, bx)}
    _, cg, _ = composite(list(parts.values()))
    out = {}
    def opt(name, R_l):
        R = parts[name]["R"] @ R_l
        return {"xyz_body_flu_m": [round(float(v), 6) for v in (parts[name]["xyz"] - cg)],
                "R_body_from_optical": [[round(float(v), 6) for v in row] for row in R]}
    out["rgbd_optical"] = opt("rgbd_camera_link", R_LINK_OPTICAL)
    out["thermal_optical"] = opt("thermal_camera_link", R_LINK_OPTICAL)
    out["imu"] = {"xyz_body_flu_m": [round(float(v), 6) for v in (parts["flight_controller_link"]["xyz"] - cg)]}
    out["gps"] = {"xyz_body_flu_m": [round(float(v), 6) for v in (parts["gps_link"]["xyz"] - cg)]}
    return out


def _fmt_ext(ext):
    lines = ["sensor_extrinsics:                 # DERIVED from the URDF; body frame FLU, origin at the CG"]
    for k, v in ext.items():
        lines.append(f"  {k}:")
        lines.append(f"    xyz_body_flu_m: {v['xyz_body_flu_m']}")
        if "R_body_from_optical" in v:
            lines.append(f"    R_body_from_optical: {v['R_body_from_optical']}   # optical (z fwd, x right, y down) -> body FLU")
    return "\n".join(lines)


def derived_block(cfg, bx):
    M, cg, I = composite(build_parts(cfg, bx))
    return (
        "# BEGIN DERIVED mass_properties (written by tools/build_urdf.py)\n"
        "mass_properties:\n"
        f"  total_mass_kg: {M:.6f}\n"
        f"  battery_x_solved_m: {bx}      # battery x chosen so the composite CG x = 0\n"
        f"  cg_in_base_link_m: [{cg[0]:.6f}, {cg[1]:.6f}, {cg[2]:.6f}]\n"
        "  inertia_about_cg_kgm2:          # base_link axes (FLU), origin at the CG\n"
        f"    Ixx: {I[0,0]:.6f}\n    Iyy: {I[1,1]:.6f}\n    Izz: {I[2,2]:.6f}\n"
        f"    Ixy: {I[0,1]:.6f}\n    Ixz: {I[0,2]:.6f}\n    Iyz: {I[1,2]:.6f}\n"
        + _fmt_ext(sensor_extrinsics(cfg, bx)) + "\n"
        "# END DERIVED mass_properties")


def main():
    check = "--check" in sys.argv
    cfg = load_cfg()
    bx = solve_battery_x(cfg)
    urdf = make_urdf(cfg, bx)
    block = derived_block(cfg, bx)
    text = open(CFG_PATH, encoding="utf-8").read()
    new = re.sub(r"# BEGIN DERIVED mass_properties.*?# END DERIVED mass_properties", lambda _m: block, text, flags=re.S)
    if check:
        ok = (new == text) and os.path.exists(URDF_PATH) and open(URDF_PATH, encoding="utf-8").read() == urdf
        print("up to date" if ok else "STALE: run python -m tools.build_urdf")
        sys.exit(0 if ok else 1)
    open(URDF_PATH, "w", newline="\n", encoding="utf-8").write(urdf)
    open(CFG_PATH, "w", newline="\n", encoding="utf-8").write(new)
    print(block)
    print(f"wrote {URDF_PATH}")


if __name__ == "__main__":
    main()
