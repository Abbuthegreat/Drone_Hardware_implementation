"""
UavSim: physics + motors + battery + sensors + simulated FC, stepped at `sim.physics_hz`.

This object contains everything on the "vehicle side" of the MAVLink boundary. The MAVLink server
(sim/mavlink_fc_server.py) is the only thing that talks to `sim.fc` from outside; autonomy code never
imports this module.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pybullet as p
import pybullet_data

from common.frames import euler_from_R
from common.geo import GeoFrame
from sim.battery import BatteryModel
from sim.config import load_config, DEFAULT_URDF
from sim.flight_controller_sim import SimulatedFlightController
from sim.mixer import QuadMixer
from sim.motor_model import MotorModel
from sim.physics_pybullet import PyBulletVehicle
from sim.sensors import ImuModel, GpsModel, FcEstimator
from sim.vehicle_model import VehicleModel


class UavSim:
    def __init__(self, cfg=None, gui=False, world=True, pos_enu=None, yaw_ned_deg=None, connect=True):
        self.cfg = cfg or load_config()
        c = self.cfg
        self.hz = c["sim"]["physics_hz"]
        self.dt = 1.0 / self.hz
        self.ctrl_div = self.hz // c["flight_controller"]["control_hz"]
        self.imu_div = self.hz // c["sensors"]["imu"]["rate_hz"]
        self.gps_div = max(1, round(self.hz / c["sensors"]["gps"]["rate_hz"]))
        self.pb = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.pb)
        p.setGravity(0, 0, -c["vehicle"]["gravity_mps2"], physicsClientId=self.pb)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSolverIterations=100, numSubSteps=1, physicsClientId=self.pb)
        self.world_info = None
        if world:
            from worlds.earthquake_world import build_earthquake_world
            self.world_info = build_earthquake_world(self.pb, c)
        else:
            p.loadURDF("plane.urdf", physicsClientId=self.pb)

        self.model = VehicleModel(DEFAULT_URDF, c)
        sp = c["spawn"]
        if pos_enu is None:
            # base_link origin height so the lowest collision surface (skids) rests on the ground
            z = -self._lowest_z(self.model) + 0.003 + sp["surface_z_m"] + sp["height_above_ground_m"]
            pos_enu = (sp["xy_enu_m"][0], sp["xy_enu_m"][1], z)
        self.vehicle = PyBulletVehicle(self.pb, c, self.model, pos_enu=pos_enu,
                                       yaw_ned_deg=sp["yaw_ned_deg"] if yaw_ned_deg is None else yaw_ned_deg)
        self.motor = MotorModel(c)
        self.battery = BatteryModel(c)
        km = c["propulsion"]["torque_coeff_kQ"] / c["propulsion"]["thrust_coeff_kT"]
        self.mixer = QuadMixer(self.model.rotor_pos_frd, self.model.spin, km)
        self.fc = SimulatedFlightController(c, self.mixer, MotorModel(c))   # FC has its own copy of the thrust-curve knowledge
        self.geo = GeoFrame.from_config(c)
        self.imu = ImuModel(c, self.model, self.imu_div * self.dt)
        self.gps = GpsModel(c, self.model, self.geo)
        self.estimator = FcEstimator(c)

        self.step_count = 0
        self.u = np.zeros(4)
        self.state = self.vehicle.read_state()
        self.imu_sample = self.imu.sample(self.state)
        self.gps_sample = self.gps.sample(self.state)
        self.motor_thrusts = np.zeros(4)
        self.estimate = self.estimator.estimate(self.state, self.imu_sample.gyro)
        self.fc.est = self.estimate
        self.fc.home_down = float(self.estimate.pos_ned[2])
        self.new_imu = False
        self.new_gps = False
        self.hooks = []           # callables(sim) run after each step (MAVLink server, loggers, ...)
        dyn = (self.world_info or {}).get("dynamic")
        if dyn is not None:       # world events of the dynamic scenario, updated at 50 Hz (the world changes; the UAV is never moved)
            div = max(1, self.hz // 50)
            self.hooks.append(lambda sim: dyn.update(sim.t, sim.state.pos_enu) if sim.step_count % div == 0 else None)

    def _lowest_z(self, model):
        """Lowest collision surface (skids/struts) in the base_link frame, from the config primitives."""
        prims = self.cfg["components"]["airframe_primitives"]
        return min(pr["xyz"][2] - pr["size_m"][2] / 2 for pr in prims if pr["name"].startswith(("skid", "strut")))

    @property
    def t(self):
        return self.step_count * self.dt

    def step(self):
        c = self.cfg
        st = self.state = self.vehicle.read_state()
        st.t = self.t
        self.new_imu = self.new_gps = False
        if self.step_count % self.imu_div == 0:
            self.imu_sample = self.imu.sample(st)
            self.new_imu = True
        if self.step_count % self.gps_div == 0:
            self.gps_sample = self.gps.sample(st)
            self.new_gps = True
        if self.step_count % self.ctrl_div == 0:
            self.estimate = self.estimator.estimate(st, self.imu_sample.gyro)
            self.u = self.fc.update(self.t, self.ctrl_div * self.dt, self.estimate,
                                    self.battery.voltage_filtered_v, self.battery.low_battery_failsafe())
        thrust, torque = self.motor.step(self.u, self.battery.voltage_v, self.fc.armed, self.dt)
        self.motor_thrusts = thrust
        self.battery.step(self.motor.electrical_power(c["propulsion"]["avionics_power_w"] if self.fc.armed else 0.0), self.dt)
        self.vehicle.apply_rotor_forces(thrust, torque, st)
        self.vehicle.step()
        self.step_count += 1
        for h in self.hooks:
            h(self)

    def run(self, seconds):
        for _ in range(int(round(seconds * self.hz))):
            self.step()

    def close(self):
        if p.isConnected(self.pb):
            p.disconnect(self.pb)
