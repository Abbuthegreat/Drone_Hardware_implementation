"""Convenience entry point: start the PyBullet simulation + simulated flight controller (MAVLink UDP 14540).

    python run_sim.py            # GUI
    python run_sim.py --headless

Same as `python -m sim.run_fc_sim`. The vehicle is commanded ONLY through MAVLink: start the ROS 2 bridge
(python -m ros2_bridge.uav_mavlink_bridge) or scripts/rpi_companion_demo.py in another terminal.
"""
import sys

from sim.run_fc_sim import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not any(a in ("--gui", "--headless") for a in args):
        args = ["--gui"] + args
    main(args)
