"""
Map frame <-> geographic frame.

The simulation "map" frame is a local tangent-plane NED frame (ENU for PyBullet) whose origin is
the point (lat0, lon0, alt0) on the WGS-84 ellipsoid:

    map origin  = PyBullet world origin (0, 0, ground level)
    map North   = PyBullet +Y
    map East    = PyBullet +X
    map Down    = PyBullet -Z
    alt0        = altitude (metres, WGS-84 ellipsoidal ~ MSL for this purpose) of the ground at the origin

Conversions go through ECEF so they are exact on the ellipsoid (no flat-earth approximation).
The same class is used by the simulated GPS/FC and by the companion side, so a survivor position
expressed in the map frame can be converted to lat/lon with exactly the same reference the FC uses.
"""
from __future__ import annotations

import math
import numpy as np

_A = 6378137.0                      # WGS-84 semi-major axis [m]
_F = 1.0 / 298.257223563            # flattening
_E2 = _F * (2.0 - _F)               # first eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    return np.array([(n + alt_m) * math.cos(lat) * math.cos(lon),
                     (n + alt_m) * math.cos(lat) * math.sin(lon),
                     (n * (1.0 - _E2) + alt_m) * math.sin(lat)])


def ecef_to_geodetic(xyz):
    x, y, z = map(float, xyz)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _E2))
    alt = 0.0
    for _ in range(6):                       # fixed-point iteration converges in 2-3 steps
        n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - _E2 * n / (n + alt)))
    return math.degrees(lat), math.degrees(lon), alt


class GeoFrame:
    """Local tangent plane (NED) anchored at a geodetic origin."""

    def __init__(self, lat0_deg: float, lon0_deg: float, alt0_m: float):
        self.lat0, self.lon0, self.alt0 = float(lat0_deg), float(lon0_deg), float(alt0_m)
        lat, lon = math.radians(self.lat0), math.radians(self.lon0)
        sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
        # rows: North, East, Down unit vectors in ECEF
        self._R_ecef_to_ned = np.array([[-sl * co, -sl * so, cl],
                                        [-so,       co,      0.0],
                                        [-cl * co, -cl * so, -sl]])
        self._ecef0 = geodetic_to_ecef(self.lat0, self.lon0, self.alt0)

    @classmethod
    def from_config(cls, cfg):
        g = cfg["geo"]
        return cls(g["origin_lat_deg"], g["origin_lon_deg"], g["origin_alt_m"])

    def ned_to_geodetic(self, ned):
        ecef = self._ecef0 + self._R_ecef_to_ned.T @ np.asarray(ned, dtype=float)
        return ecef_to_geodetic(ecef)          # (lat_deg, lon_deg, alt_m)

    def geodetic_to_ned(self, lat_deg, lon_deg, alt_m):
        ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        return self._R_ecef_to_ned @ (ecef - self._ecef0)

    def enu_to_geodetic(self, enu):
        from .frames import enu_to_ned
        return self.ned_to_geodetic(enu_to_ned(enu))
