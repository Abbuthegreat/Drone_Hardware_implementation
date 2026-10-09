#!/usr/bin/env bash
# Start the ROS 2 <-> MAVLink bridge (source your ROS 2 setup first).  ./scripts/start_bridge.sh [udp:127.0.0.1:14540] [enu|ned|body_flu]
cd "$(dirname "$0")/.." && exec "${UAV_PYTHON:-python3}" -m ros2_bridge.uav_mavlink_bridge --ros-args -p "connection:=${1:-udp:127.0.0.1:14540}" -p "cmd_frame:=${2:-enu}"
