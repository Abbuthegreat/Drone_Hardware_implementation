"""
Test L - MAVLink communication over real UDP, against the FC simulator running as its own process.

    CompanionMavlink ("udp:127.0.0.1:PORT")  <-- MAVLink 2 / UDP -->  MavlinkFcServer (udpin:0.0.0.0:PORT)

Tests run in file order and share one FC process (state carries over deliberately, like a real session).
Ground truth comes from the evaluation-only UDP JSON stream and is used only to CHECK the telemetry.
"""
import math
import time

import numpy as np
import pytest

from common.geo import GeoFrame
from common.mav import mavlink
from companion.mavlink_client import CompanionMavlink
from eval.ground_truth import EvalListener
from tests.fc_process import FcProcess

pytestmark = pytest.mark.wire
PORT, EVAL_PORT = 14641, 14661


@pytest.fixture(scope="module")
def fc(tmp_path_factory):
    p = FcProcess(PORT, EVAL_PORT, log_dir=str(tmp_path_factory.mktemp("wire")))
    ev = EvalListener(EVAL_PORT)
    p.wait_ready()
    yield p, ev
    ev.close()
    p.stop()


@pytest.fixture(scope="module")
def link(fc):
    c = CompanionMavlink(f"udp:127.0.0.1:{PORT}")
    c.connect(15)
    time.sleep(1.5)
    yield c
    c.close()


def _stream(c, seconds, fn, hz=20):
    t_end = time.time() + seconds
    while time.time() < t_end:
        fn()
        time.sleep(1.0 / hz)


def test_01_heartbeat_fc_to_companion(link):
    hb = link.last_messages["HEARTBEAT"]
    assert hb.type == mavlink.MAV_TYPE_QUADROTOR
    assert hb.autopilot == mavlink.MAV_AUTOPILOT_GENERIC        # it does not pretend to be PX4
    assert (link.target_system, link.target_component) == (1, 1)
    assert not (hb.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def test_02_all_required_telemetry_messages_stream_at_configured_rates(link, cfg):
    link.msg_counts.clear()
    time.sleep(4.0)
    rates = cfg["mavlink"]["rates_hz"]
    for msg, key in (("LOCAL_POSITION_NED", "local_position_ned"), ("ATTITUDE", "attitude"),
                     ("GLOBAL_POSITION_INT", "global_position_int"), ("SYS_STATUS", "sys_status"),
                     ("HIGHRES_IMU", "highres_imu"), ("GPS_RAW_INT", "gps_raw_int")):
        got = link.msg_counts.get(msg, 0) / 4.0
        assert got == pytest.approx(rates[key], rel=0.35), f"{msg}: {got:.1f} Hz vs {rates[key]} Hz"
    assert link.msg_counts.get("HEARTBEAT", 0) / 4.0 == pytest.approx(cfg["mavlink"]["heartbeat_hz"], rel=0.5)


def test_03_timestamps_increase(link):
    ts = []
    for _ in range(10):
        ts.append(link.last_messages["LOCAL_POSITION_NED"].time_boot_ms)
        time.sleep(0.1)
    assert all(b >= a for a, b in zip(ts, ts[1:])) and ts[-1] > ts[0]
    imu = link.pop_imu()
    t = np.array([s.t_s for s in imu])
    assert len(t) > 50 and np.all(np.diff(t) > 0)


def test_04_arm_disarm_command_ack_and_state(link):
    assert link.arm() == mavlink.MAV_RESULT_ACCEPTED
    time.sleep(0.6)
    assert link.telemetry().armed
    assert link.disarm() == mavlink.MAV_RESULT_ACCEPTED       # on the ground: allowed
    time.sleep(0.6)
    assert not link.telemetry().armed
    assert link.arm() == mavlink.MAV_RESULT_ACCEPTED


def test_05_unsupported_command_is_nacked(link):
    r = link.command_long(mavlink.MAV_CMD_DO_SET_HOME, 1)
    assert r == mavlink.MAV_RESULT_UNSUPPORTED


def test_06_offboard_refused_without_setpoint_stream_then_accepted(link):
    assert link.set_mode("OFFBOARD") == mavlink.MAV_RESULT_DENIED
    assert link.telemetry().mode == "HOLD"
    _stream(link, 1.0, lambda: link.send_velocity_ned(0, 0, 0))
    assert link.set_mode("OFFBOARD") == mavlink.MAV_RESULT_ACCEPTED
    _stream(link, 0.6, lambda: link.send_velocity_ned(0, 0, 0))       # keep streaming or the watchdog fires
    tl = link.telemetry()
    assert tl.mode == "OFFBOARD" and tl.armed


def test_07_vehicle_responds_physically_and_telemetry_agrees_with_pybullet(fc, link, cfg):
    _, ev = fc
    geo = GeoFrame.from_config(cfg)
    # climb, then move north+east while yawing slowly
    _stream(link, 5.0, lambda: link.send_velocity_ned(0, 0, -1.0))
    _stream(link, 4.0, lambda: link.send_velocity_ned(1.0, 0.5, 0.0, yaw_rate=0.2))
    tl = link.telemetry()
    truth = ev.nearest(tl.time_boot_ms / 1000.0)
    assert truth is not None and abs(truth["t"] - tl.time_boot_ms / 1000.0) < 0.1
    pos_true, vel_true = np.array(truth["pos_ned"]), np.array(truth["vel_ned"])
    assert -tl.local_ned[2] > 3.0 and tl.local_ned[0] > -2.0            # it climbed and moved north
    assert np.linalg.norm(tl.local_ned - pos_true) < 0.25               # local position vs PyBullet (NED)
    assert np.linalg.norm(tl.velocity_ned - vel_true) < 0.25            # velocity vs PyBullet
    assert abs(math.remainder(tl.yaw_rad - truth["rpy"][2], 2 * math.pi)) < 0.05      # yaw vs PyBullet
    assert tl.yaw_rad > 0.3                                             # positive yaw rate turned clockwise
    ned_from_gps = geo.geodetic_to_ned(tl.lat_deg, tl.lon_deg, tl.alt_m)
    assert np.linalg.norm(ned_from_gps - tl.local_ned) < 0.1            # GPS-frame position follows local position
    # frames: PyBullet ENU truth (+X East, +Y North) vs MAVLink NED
    enu = np.array(truth["pos_enu"])
    assert enu[0] == pytest.approx(pos_true[1], abs=1e-6) and enu[1] == pytest.approx(pos_true[0], abs=1e-6)
    assert enu[2] == pytest.approx(-pos_true[2] + 0.0, abs=0.3)          # up = -down (+ home offset)


def test_08_motor_telemetry_reflects_thrust(fc, link):
    _, ev = fc
    rpm = link.motor_rpm()
    assert rpm is not None and np.all(rpm > 3000)
    truth = ev.latest()
    assert np.allclose(rpm, truth["rpm"], rtol=0.05, atol=50)
    assert 20.0 < sum(truth["thrust_n"]) < 35.0


def test_09_set_mode_message_and_mode_in_heartbeat(link):
    link.set_mode_message("HOLD")
    time.sleep(1.2)
    assert link.telemetry().mode == "HOLD"


def test_10_set_message_interval(link):
    assert link.set_message_interval(30, 5) == mavlink.MAV_RESULT_ACCEPTED   # ATTITUDE -> 5 Hz
    time.sleep(0.5)
    link.msg_counts.clear()
    time.sleep(3.0)
    assert link.msg_counts["ATTITUDE"] / 3.0 == pytest.approx(5, rel=0.4)


def test_11_fc_received_companion_messages(fc, link):
    p, _ = fc
    link.close()
    p.stop()
    s = p.summary()
    assert s is not None
    for name in ("HEARTBEAT", "SET_POSITION_TARGET_LOCAL_NED", "COMMAND_LONG", "SET_MODE"):
        assert s["rx"].get(name, 0) > 0, name
    assert s["tx"].get("COMMAND_ACK", 0) >= 5
