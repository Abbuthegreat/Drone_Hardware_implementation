#!/usr/bin/env bash
# Start PyBullet + simulated flight controller (MAVLink UDP 14540).  ./scripts/start_sim.sh [--gui|--headless] [--no-world]
cd "$(dirname "$0")/.." && exec "${UAV_PYTHON:-python3}" -m sim.run_fc_sim "${@:---headless}"
