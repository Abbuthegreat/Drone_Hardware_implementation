# Dot-source this on Windows if you installed the environments with micromamba (see README "Windows setup"):
#   . .\scripts\env_micromamba.ps1 uavros        # or: uav  (no ROS 2)
# Sets PATH / AMENT_PREFIX_PATH so the conda python finds its DLLs (needed for numpy / pybullet / rclpy) and exports $env:UAV_PYTHON.
param([string]$EnvName = "uavros", [string]$Root = "$env:USERPROFILE\mm\root")
$E = Join-Path $Root "envs\$EnvName"
$env:PATH = "$E;$E\Library\bin;$E\Library\mingw-w64\bin;$E\Scripts;$env:PATH"
$env:CONDA_PREFIX = $E
if (Test-Path "$E\Library\bin\ros2.exe") { $env:AMENT_PREFIX_PATH = "$E\Library" }
$env:UAV_PYTHON = "$E\python.exe"
Write-Host "UAV_PYTHON = $env:UAV_PYTHON"
