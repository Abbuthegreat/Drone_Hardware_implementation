"""Dynamic-scenario world events (worlds/dynamic_obstacles.py): trigger logic, no drop onto the UAV, patrol motion, spec validation."""
import copy
import math

import numpy as np
import pybullet as p
import pytest

from worlds.dynamic_obstacles import DynamicObstacles


@pytest.fixture
def pb():
    c = p.connect(p.DIRECT)
    yield c
    p.disconnect(c)


def _maker(pb):
    def make(pos, half, q, tex):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=pb)
        return p.createMultiBody(0, col, -1, pos, q, physicsClientId=pb)
    return make


def _specs(cfg):
    return {s["name"]: s for s in cfg["world"]["dynamic_obstacles"]}


def test_collapse_triggers_in_range_and_never_onto_the_uav(pb, cfg):
    s = _specs(cfg)["aftershock_slab"]
    dyn = DynamicObstacles(pb, [s], _maker(pb))
    body = dyn.items[0]["body"]
    c = np.array(s["xyz"])
    dyn.update(0.0, (c[0] + 20.0, c[1], 2.0))                                   # far away: stays hidden below the ground
    assert p.getBasePositionAndOrientation(body, physicsClientId=pb)[0][2] < -10
    dyn.update(1.0, (c[0] + 1.0, c[1], 2.0))                                    # UAV almost on top of it: must NOT drop
    assert dyn.items[0]["state"] == "waiting"
    r = 0.5 * (s["trigger_radius_m"] + max(s["half"][:2]) + s["min_clearance_m"])  # inside the trigger ring, outside the clearance
    dyn.update(2.0, (c[0] + r, c[1], 2.0))
    assert dyn.items[0]["state"] == "falling"
    for k in range(1, 30):
        dyn.update(2.0 + 0.05 * k, (c[0] + r, c[1], 2.0))
    pos = p.getBasePositionAndOrientation(body, physicsClientId=pb)[0]
    assert dyn.items[0]["state"] == "down" and np.allclose(pos, c, atol=1e-6)
    assert [e[2] for e in dyn.events] == ["collapse triggered", "collapsed"]


def test_patrol_moves_between_its_endpoints(pb, cfg):
    s = _specs(cfg)["sliding_debris"]
    dyn = DynamicObstacles(pb, [s], _maker(pb))
    body = dyn.items[0]["body"]
    a, b = np.array(s["a"]), np.array(s["b"])
    L = float(np.linalg.norm(b - a))
    xs = []
    for t in np.arange(0.0, 2 * (L / s["speed_mps"] + s["dwell_s"]), 0.5):
        dyn.update(float(t), (100.0, 100.0, 2.0))
        xs.append(np.array(p.getBasePositionAndOrientation(body, physicsClientId=pb)[0][:2]))
    d_a = [np.linalg.norm(x - a) for x in xs]
    assert min(d_a) < 0.05 and max(d_a) > L - 0.05                              # reaches both ends
    steps = [np.linalg.norm(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    assert max(steps) <= s["speed_mps"] * 0.5 + 1e-6                            # never faster than configured


def test_impossible_collapse_spec_is_rejected(pb, cfg):
    s = copy.deepcopy(_specs(cfg)["aftershock_slab"])
    s["trigger_radius_m"] = max(s["half"][:2]) + s["min_clearance_m"] - 0.1
    with pytest.raises(ValueError):
        DynamicObstacles(pb, [s], _maker(pb))
