"""
Flight-mode names and their MAVLink HEARTBEAT.custom_mode encoding.

The simulated FC is NOT PX4 and reports MAV_AUTOPILOT_GENERIC. To keep the companion-side code
identical for simulation and hardware, it *reuses the bit layout that PX4 uses* in custom_mode
(main_mode in bits 16-23, sub_mode in bits 24-31). Only the values listed here are implemented
by the simulator. On a real Pixhawk running PX4 the same encoding is what
`CompanionMavlink.set_mode()` sends.
"""
PX4_MAIN_OFFBOARD = 6
PX4_MAIN_AUTO = 4
PX4_SUB_AUTO_LOITER = 3
PX4_SUB_AUTO_LAND = 6


def encode_custom_mode(main: int, sub: int = 0) -> int:
    return (int(main) << 16) | (int(sub) << 24)


def decode_custom_mode(custom_mode: int):
    return (int(custom_mode) >> 16) & 0xFF, (int(custom_mode) >> 24) & 0xFF


# name -> (main, sub)
MODE_TABLE = {
    "HOLD": (PX4_MAIN_AUTO, PX4_SUB_AUTO_LOITER),
    "OFFBOARD": (PX4_MAIN_OFFBOARD, 0),
    "LAND": (PX4_MAIN_AUTO, PX4_SUB_AUTO_LAND),
    # FAILSAFE_HOLD is *reported* as AUTO/LOITER with MAV_STATE_CRITICAL; it cannot be commanded.
    "FAILSAFE_HOLD": (PX4_MAIN_AUTO, PX4_SUB_AUTO_LOITER),
}
COMMANDABLE_MODES = ("HOLD", "OFFBOARD", "LAND")


def mode_to_custom_mode(name: str) -> int:
    main, sub = MODE_TABLE[name]
    return encode_custom_mode(main, sub)


def custom_mode_to_name(custom_mode: int, critical: bool = False) -> str:
    main, sub = decode_custom_mode(custom_mode)
    if main == PX4_MAIN_OFFBOARD:
        return "OFFBOARD"
    if main == PX4_MAIN_AUTO and sub == PX4_SUB_AUTO_LAND:
        return "LAND"
    if main == PX4_MAIN_AUTO and sub == PX4_SUB_AUTO_LOITER:
        return "FAILSAFE_HOLD" if critical else "HOLD"
    return f"UNKNOWN({main},{sub})"
