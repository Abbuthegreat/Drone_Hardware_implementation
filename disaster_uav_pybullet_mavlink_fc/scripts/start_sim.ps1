# Start PyBullet + simulated flight controller (MAVLink UDP 14540).  Usage: .\scripts\start_sim.ps1 [-Gui] [-NoWorld] [-RosSensors]
param([switch]$Gui, [switch]$NoWorld, [switch]$RosSensors)
$py = if ($env:UAV_PYTHON) { $env:UAV_PYTHON } else { "python" }
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$a = @("-m", "sim.run_fc_sim", $(if ($Gui) { "--gui" } else { "--headless" }))
if ($NoWorld) { $a += "--no-world" }
if ($RosSensors) { $a += "--ros-sensors" }
& $py @a
