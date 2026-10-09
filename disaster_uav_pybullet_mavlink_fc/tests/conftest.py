import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "wire: starts the FC simulator in a subprocess and talks real MAVLink UDP")
    config.addinivalue_line("markers", "ros: needs rclpy (ROS 2) - skipped otherwise")
    config.addinivalue_line("markers", "slow: full closed-loop mission tests (minutes)")


@pytest.fixture(scope="session")
def cfg():
    from sim.config import load_config
    return load_config()


@pytest.fixture
def make_sim():
    from sim.sim_core import UavSim
    sims = []

    def _make(**kw):
        kw.setdefault("world", False)
        s = UavSim(**kw)
        sims.append(s)
        return s
    yield _make
    for s in sims:
        s.close()
