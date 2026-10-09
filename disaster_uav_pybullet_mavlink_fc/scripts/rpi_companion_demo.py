"""
Minimal companion-computer demo WITHOUT ROS: talks MAVLink directly through CompanionMavlink.

    python scripts/rpi_companion_demo.py                                   # simulation (udp:127.0.0.1:14540)
    python scripts/rpi_companion_demo.py --connection serial:/dev/serial0:921600   # real Pixhawk

Not an autonomous algorithm: a scripted takeoff / square pattern / land, to show the command path.
Commands are NED. Yaw rate is clockwise-positive.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from companion.mavlink_client import CompanionMavlink  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connection", default="udp:127.0.0.1:14540")
    a = ap.parse_args()
    c = CompanionMavlink(a.connection, log_path="logs/rpi_companion_demo.log")
    c.connect(20)
    print("Connected to FC (system %d)." % c.target_system)
    print("arm/mode results:", c.start_offboard())
    t0 = time.time()
    legs = [(8, (0, 0, -0.5)), (8, (1.0, 0, 0)), (8, (0, 1.0, 0)), (8, (-1.0, 0, 0)), (8, (0, -1.0, 0)), (4, (0, 0, 0))]
    for dur, (vn, ve, vd) in legs:
        t_end = time.time() + dur
        while time.time() < t_end:
            c.send_velocity_ned(vn, ve, vd, 0.0)               # keep the setpoint stream alive (>= 2 Hz; we use 20 Hz)
            tl = c.telemetry()
            if tl and int((time.time() - t0) * 2) % 2 == 0:
                print(f"NED {tl.local_ned.round(2)} vel {tl.velocity_ned.round(2)} yaw {tl.yaw_rad:+.2f} mode {tl.mode} "
                      f"GPS {tl.lat_deg:.7f},{tl.lon_deg:.7f} batt {tl.battery_v:.2f} V")
            time.sleep(0.05)
    print("land:", c.land())
    time.sleep(15)
    print("disarm:", c.disarm())
    c.close()


if __name__ == "__main__":
    main()
