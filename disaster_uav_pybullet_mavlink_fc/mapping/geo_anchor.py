"""
Local metric map <-> geographic reference.

While the UAV is still on the ground, GPS fixes are averaged; the antenna lever arm (config `sensor_extrinsics.gps`) is removed using the
start heading, giving the geodetic position of the map origin (UAV body/CG at take-off). Afterwards

    lat, lon, alt = GeoFrame(anchor).enu_to_geodetic(local_xyz)          (x East, y North, z Up)

The SLAM trajectory stays local/relative; GPS only provides the global reference (and its error, ~0.6 m horizontal, is inherited by the
absolute geo-tags but not by relative positions).
"""
from __future__ import annotations

import math

import numpy as np

from common.geo import GeoFrame


class GpsAnchor:
    def __init__(self, antenna_xyz_flu, min_fixes=8):
        self.ant = np.array(antenna_xyz_flu, float)
        self.n_min = min_fixes
        self.samples = []
        self.result = None

    def add(self, lat, lon, alt, fix_type=3):
        if fix_type >= 3 and self.result is None:
            self.samples.append((lat, lon, alt))

    def try_finish(self, heading_enu):
        """Returns (lat0, lon0, alt0) once enough fixes were collected."""
        if self.result is not None:
            return self.result
        if len(self.samples) < self.n_min:
            return None
        s = np.array(self.samples)
        ref = GeoFrame(s[0, 0], s[0, 1], s[0, 2])
        enu = np.array([ref.geodetic_to_ned(*row) for row in s])              # NED offsets from the first fix
        m = enu.mean(axis=0)                                                   # mean NED of the antenna
        # antenna offset from the body origin in ENU (body level at start): rotate FLU by the heading
        c, sn = math.cos(heading_enu), math.sin(heading_enu)
        ant_e = c * self.ant[0] - sn * self.ant[1]
        ant_n = sn * self.ant[0] + c * self.ant[1]
        body_ned = np.array([m[0] - ant_n, m[1] - ant_e, m[2] + self.ant[2]])  # down = -up
        lat, lon, alt = ref.ned_to_geodetic(body_ned)
        self.result = (lat, lon, alt)
        return self.result
