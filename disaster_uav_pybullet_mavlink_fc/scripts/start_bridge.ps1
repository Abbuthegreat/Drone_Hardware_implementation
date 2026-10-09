# Start the ROS 2 <-> MAVLink bridge.  Usage: .\scripts\start_bridge.ps1 [-Connection udp:127.0.0.1:14540] [-CmdFrame enu]
param([string]$Connection = "udp:127.0.0.1:14540", [string]$CmdFrame = "enu")
$py = if ($env:UAV_PYTHON) { $env:UAV_PYTHON } else { "python" }
Set-Location (Split-Path -Parent $PSScriptRoot)
& $py -m ros2_bridge.uav_mavlink_bridge --ros-args -p "connection:=$Connection" -p "cmd_frame:=$CmdFrame"
