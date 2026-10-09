# Run the full validation (pytest + calibration + tuning + ROS 2 communication test).
#   . .\scripts\env_micromamba.ps1 uavros ; .\scripts\run_tests.ps1
$py = if ($env:UAV_PYTHON) { $env:UAV_PYTHON } else { "python" }
Set-Location (Split-Path -Parent $PSScriptRoot)
& $py scripts/run_all_tests.py --ros-python $py
