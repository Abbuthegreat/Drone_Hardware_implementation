"""
Dynamic disaster events for the `dynamic` scenario (simulation-world side only; autonomy never sees these specs or body ids).

    collapse : a structure (box) that is not there at the start and falls into place when the UAV comes within `trigger_radius_m`
               of it (secondary collapse / aftershock). It never drops onto the vehicle: the drop waits while the UAV is closer than
               `min_clearance_m` to its footprint. Fall animation `fall_s` from `drop_height_m` above the final pose.
    patrol   : debris sliding back and forth between `a` and `b` (world xy) at `speed_mps`, pausing `dwell_s` at each end.

Bodies are kinematic (mass 0) and moved with resetBasePositionAndOrientation, i.e. the WORLD changes; the UAV is never moved.
The trigger uses the simulated vehicle position because it is the environment reacting, exactly like the rest of the physics engine;
the navigation stack only learns about the change through its RGB-D camera -> map -> A* / VFH+.
"""
from __future__ import annotations

import math

import numpy as np
import pybullet as p


class DynamicObstacles:
    def __init__(self, pb, specs, make_box):
        self.pb = pb
        self.items = []
        self.events = []                                   # evaluation log: (t, name, what)
        for s in specs:
            q = p.getQuaternionFromEuler([math.radians(a) for a in s.get("rpy_deg", [0, 0, 0])])
            if s["kind"] == "collapse":
                if s["trigger_radius_m"] <= max(s["half"][0], s["half"][1]) + s.get("min_clearance_m", 2.0):
                    raise ValueError(f"{s['name']}: trigger_radius_m must exceed footprint + min_clearance_m, or the collapse can never trigger")
                final = np.array(s["xyz"], float)
                hidden = final + np.array([0.0, 0.0, -30.0])        # parked far below the ground (out of every camera's view)
                body = make_box(hidden.tolist(), s["half"], q, s.get("tex", "concrete"))
                self.items.append({"spec": s, "body": body, "q": q, "final": final, "state": "waiting", "t0": None})
            elif s["kind"] == "patrol":
                a, b = np.array(s["a"], float), np.array(s["b"], float)
                z = float(s["half"][2])
                body = make_box([a[0], a[1], z], s["half"], q, s.get("tex", "rust"))
                L = float(np.linalg.norm(b - a))
                self.items.append({"spec": s, "body": body, "q": q, "a": a, "b": b, "z": z, "L": L, "state": "moving"})
            else:
                raise ValueError(f"unknown dynamic obstacle kind {s['kind']}")

    @property
    def body_names(self):
        return {it["body"]: it["spec"]["name"] for it in self.items}

    def describe(self):
        return [dict(it["spec"]) for it in self.items]

    def update(self, t, uav_xyz):
        for it in self.items:
            s = it["spec"]
            if s["kind"] == "collapse":
                self._collapse(it, s, t, uav_xyz)
            else:
                self._patrol(it, s, t)

    def _collapse(self, it, s, t, uav):
        fin = it["final"]
        if it["state"] == "waiting":
            d = math.hypot(uav[0] - fin[0], uav[1] - fin[1])
            footprint = max(s["half"][0], s["half"][1])
            if d < s["trigger_radius_m"] and d - footprint > s.get("min_clearance_m", 2.0):
                it["state"], it["t0"] = "falling", t
                self.events.append((round(t, 2), s["name"], "collapse triggered"))
        if it["state"] == "falling":
            k = min(1.0, (t - it["t0"]) / max(s.get("fall_s", 0.8), 1e-3))
            pos = fin + np.array([0.0, 0.0, s.get("drop_height_m", 3.5) * (1.0 - k * k)])
            p.resetBasePositionAndOrientation(it["body"], pos.tolist(), it["q"], physicsClientId=self.pb)
            if k >= 1.0:
                it["state"] = "down"
                self.events.append((round(t, 2), s["name"], "collapsed"))

    def _patrol(self, it, s, t):
        v, dwell, L = s["speed_mps"], s.get("dwell_s", 0.0), it["L"]
        leg = L / max(v, 1e-6)
        period = 2 * (leg + dwell)
        ph = (t + s.get("phase_s", 0.0)) % period
        if ph < leg:
            u = ph / leg
        elif ph < leg + dwell:
            u = 1.0
        elif ph < 2 * leg + dwell:
            u = 1.0 - (ph - leg - dwell) / leg
        else:
            u = 0.0
        xy = it["a"] + u * (it["b"] - it["a"])
        p.resetBasePositionAndOrientation(it["body"], [xy[0], xy[1], it["z"]], it["q"], physicsClientId=self.pb)
