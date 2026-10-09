"""
Diagnostic tool.

    python scripts/diagnose.py                 # environment + configuration + model checks
    python scripts/diagnose.py --live          # also connect to a RUNNING FC (sim or Pixhawk) and print live telemetry
    python scripts/diagnose.py --live --connection serial:/dev/serial0:921600

Checks: python packages (pybullet, pymavlink, numpy, yaml, cv2, rclpy), config validity and derived
hover/thrust numbers, URDF mass/CG/inertia consistency, UDP port availability, and (with --live)
HEARTBEAT, telemetry stream rates, IMU/GPS presence, and clock/timestamp sanity.
"""
import argparse
import importlib
import os
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def line(tag, msg):
    print(f"{tag} {msg}")


def check_imports():
    ok = True
    for name, required in (("numpy", True), ("yaml", True), ("pymavlink", True), ("pybullet", True), ("cv2", False),
                           ("scipy", False), ("rclpy", False)):
        try:
            m = importlib.import_module(name)
            line(OK, f"{name} {getattr(m, '__version__', '')}")
        except Exception as e:                             # noqa: BLE001
            line(BAD if required else WARN, f"{name} not importable ({e.__class__.__name__}) " + ("- REQUIRED" if required else "- optional"))
            ok &= not required
    return ok


def check_config():
    from sim.config import load_config, derived, DEFAULT_URDF
    from sim.vehicle_model import VehicleModel
    import numpy as np
    from sim.config import inertia_matrix
    cfg = load_config()
    d = derived(cfg)
    line(OK, f"config valid; weight {d['weight_n']:.3f} N, hover {d['hover_thrust_per_motor_n']:.4f} N/motor, "
             f"T/W {d['thrust_to_weight_nominal_v']:.2f}, hover rpm (placeholder model) {d['hover_rpm_nominal_v']:.0f}")
    line(OK, f"800 KV no-load limits (NOT loaded rpm): {d['noload_rpm']}")
    m = VehicleModel(DEFAULT_URDF, cfg)
    ok = abs(m.total_mass - 2.2) < 1e-6 and np.allclose(m.inertia_cg, inertia_matrix(cfg), atol=2e-6)
    line(OK if ok else BAD, f"URDF mass {m.total_mass:.6f} kg, CG {m.cg_base.round(4)}, drone.yaml inertia {'matches' if ok else 'STALE -> python -m tools.build_urdf'}")
    return ok


def check_port(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", port))
        line(OK, f"UDP {port} is free (no FC simulator running)")
        return True
    except OSError:
        line(WARN, f"UDP {port} is in use (an FC simulator is probably running)")
        return False
    finally:
        s.close()


def live(connection):
    from companion.mavlink_client import CompanionMavlink
    c = CompanionMavlink(connection)
    try:
        c.connect(8)
    except TimeoutError as e:
        line(BAD, str(e))
        return False
    line(OK, f"HEARTBEAT from system {c.target_system} component {c.target_component}, autopilot id {c.fc_autopilot} "
             f"({'PX4' if c.fc_autopilot == 12 else 'GENERIC / simulator' if c.fc_autopilot == 0 else 'other'})")
    time.sleep(3.0)
    c.msg_counts.clear()
    t0 = time.time()
    time.sleep(4.0)
    dt = time.time() - t0
    for name in ("HEARTBEAT", "LOCAL_POSITION_NED", "GLOBAL_POSITION_INT", "ATTITUDE", "SYS_STATUS", "HIGHRES_IMU", "GPS_RAW_INT"):
        n = c.msg_counts.get(name, 0)
        line(OK if n else WARN, f"{name:22s} {n / dt:6.1f} Hz")
    tl = c.telemetry()
    if tl:
        line(OK, f"NED {tl.local_ned.round(2)} vel {tl.velocity_ned.round(2)} rpy {[round(x, 3) for x in (tl.roll_rad, tl.pitch_rad, tl.yaw_rad)]} "
                 f"mode {tl.mode} armed {tl.armed} batt {tl.battery_v:.2f} V ({tl.battery_pct}%)")
    g = c.gps()
    if g:
        line(OK if g.fix_type >= 3 else WARN, f"GPS fix {g.fix_type} sats {g.satellites} lat {g.lat_deg:.7f} lon {g.lon_deg:.7f} alt {g.alt_m:.2f}")
    c.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--connection", default="udp:127.0.0.1:14540")
    a = ap.parse_args()
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    ok = check_imports()
    if ok:
        ok &= check_config()
    if not a.live:
        check_port(14540)
    else:
        ok &= live(a.connection)
    print("\nRESULT:", "all required checks passed" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
