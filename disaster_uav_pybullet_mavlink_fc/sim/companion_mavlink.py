"""Backward-compatibility shim. The companion-side client now lives in companion/mavlink_client.py
(a package that must not import the simulator)."""
from companion.mavlink_client import CompanionMavlink, Telemetry, hardware_connection  # noqa: F401
