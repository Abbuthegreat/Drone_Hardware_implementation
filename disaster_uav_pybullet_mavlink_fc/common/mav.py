"""
pymavlink helpers shared by the simulated FC and the companion client.

* Forces MAVLink 2 framing (needed for messages with id > 255 such as ESC_STATUS) by setting
  MAVLINK20=1 BEFORE pymavlink is imported. Always import mavutil through this module.
* `open_link()` accepts the connection strings used across the project:
      udp:HOST:PORT      companion-side UDP client (translated to a *client* socket, see below)
      udpout:HOST:PORT   same
      udpin:HOST:PORT    listening socket (this is what the simulated FC uses)
      serial:DEVICE:BAUD or serial:DEVICE   real Pixhawk (e.g. serial:/dev/serial0:921600)

Why not pass "udp:127.0.0.1:14540" straight to pymavlink?  In pymavlink the legacy "udp:" prefix
defaults to *listening* mode, which would clash with the FC that already listens on that port, and
its "udpout:" mode binds the local port number equal to the destination port on Windows, which fails
when the FC runs on the same machine. `MavUdpClient` binds an ephemeral local port instead.
"""
from __future__ import annotations

import os
import socket

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

mavlink = mavutil.mavlink


class MavUdpClient(mavutil.mavudp):
    """UDP client link: sends to (host, port), receives replies on an ephemeral local port."""

    def __init__(self, host, port, source_system=255, source_component=0):
        self.port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_server = False
        self.broadcast = False
        self.destination_addr = (host, int(port))
        self.port.bind(("0.0.0.0", 0))
        self.port.setblocking(False)
        self.last_address = None
        self.timeout = 0
        self.clients = set()
        self.clients_last_alive = {}
        self.resolved_destination_addr = None
        mavutil.mavfile.__init__(self, self.port.fileno(), f"udpout:{host}:{port}", source_system=source_system,
                                 source_component=source_component, input=False, use_native=False)


def open_link(connection: str, baud: int = 115200, source_system: int = 255, source_component: int = 0):
    if connection.startswith(("udp:", "udpout:")):
        hostport = connection.split(":", 1)[1]
        host, port = hostport.rsplit(":", 1)
        return MavUdpClient(host, int(port), source_system, source_component)
    if connection.startswith("udpin:"):
        return mavutil.mavlink_connection(connection, source_system=source_system,
                                          source_component=source_component, dialect="common")
    if connection.startswith("serial:"):
        rest = connection[len("serial:"):]
        dev, _, b = rest.rpartition(":")
        if not dev or not b.isdigit():
            dev, b = rest, str(baud)
        return mavutil.mavlink_connection(dev, baud=int(b), source_system=source_system,
                                          source_component=source_component, dialect="common")
    return mavutil.mavlink_connection(connection, baud=baud, source_system=source_system,
                                      source_component=source_component, dialect="common")
