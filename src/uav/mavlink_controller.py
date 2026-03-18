"""
MAVLink UAV Controller
======================
Wraps pymavlink to provide high-level UAV control primitives:
  - connect / disconnect
  - arm / disarm
  - takeoff / land
  - move_to (lat, lon, alt)
  - set_airspeed
  - get_telemetry

The controller is designed to work both against a real flight controller
(via serial/UDP) and against ArduPilot/PX4 SITL via MAVProxy.

Design notes
------------
MAVLink uses a request–response model.  Each command is sent as a
MAV_CMD encoded inside a COMMAND_LONG or COMMAND_INT message, and the
flight controller replies with a COMMAND_ACK.  Position targets are
sent via SET_POSITION_TARGET_GLOBAL_INT or via MISSION_ITEM messages.

Coordinate frame used throughout: MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
  * latitude / longitude in 1e-7 degrees (integer)
  * altitude in metres above home (relative)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import ardupilotmega as mavlink
    _MAVLINK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MAVLINK_AVAILABLE = False
    logger.warning(
        "pymavlink is not installed – MAVLinkController will run in stub mode."
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Telemetry:
    """Snapshot of key UAV telemetry values."""
    latitude: float = 0.0       # degrees
    longitude: float = 0.0      # degrees
    altitude: float = 0.0       # metres above home (relative)
    heading: float = 0.0        # degrees (0–360)
    airspeed: float = 0.0       # m/s
    groundspeed: float = 0.0    # m/s
    battery_voltage: float = 0.0  # volts
    battery_remaining: int = -1   # percent (-1 = unknown)
    armed: bool = False
    mode: str = "UNKNOWN"
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MAVLinkController:
    """High-level UAV controller built on top of pymavlink.

    Parameters
    ----------
    connection_string:
        pymavlink connection string, e.g.::

            "udp:127.0.0.1:14550"   # SITL via MAVProxy (default)
            "tcp:127.0.0.1:5760"    # SITL direct TCP
            "serial:/dev/ttyACM0:57600"  # real hardware

    timeout:
        Seconds to wait for the initial heartbeat from the autopilot.
    source_system:
        MAVLink system ID to use for the GCS (ground station).
    """

    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14550",
        timeout: int = 30,
        source_system: int = 255,
    ) -> None:
        self.connection_string = connection_string
        self.timeout = timeout
        self.source_system = source_system
        self._connection = None
        self._target_system: int = 1
        self._target_component: int = 1

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the MAVLink connection and wait for a heartbeat."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] MAVLink connect: %s", self.connection_string)
            return

        logger.info("Connecting to %s …", self.connection_string)
        self._connection = mavutil.mavlink_connection(
            self.connection_string,
            source_system=self.source_system,
        )
        msg = self._connection.wait_heartbeat(timeout=self.timeout)
        if msg is None:
            raise ConnectionError(
                f"No heartbeat received from {self.connection_string} "
                f"within {self.timeout} s"
            )
        self._target_system = self._connection.target_system
        self._target_component = self._connection.target_component
        logger.info(
            "Connected – system %d, component %d",
            self._target_system,
            self._target_component,
        )

    def disconnect(self) -> None:
        """Close the MAVLink connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Disconnected from %s", self.connection_string)

    def is_connected(self) -> bool:
        return self._connection is not None or not _MAVLINK_AVAILABLE

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Change the flight mode (e.g. 'GUIDED', 'LOITER', 'RTL')."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] set_mode: %s", mode)
            return

        mode_id = self._connection.mode_mapping().get(mode)
        if mode_id is None:
            raise ValueError(f"Unknown mode: {mode!r}")
        self._connection.mav.set_mode_send(
            self._target_system,
            mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        logger.info("Mode changed to %s", mode)

    # ------------------------------------------------------------------
    # Arm / Disarm
    # ------------------------------------------------------------------

    def arm(self, timeout: int = 15) -> None:
        """Arm the vehicle.  Switches to GUIDED mode first."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] arm")
            return

        self.set_mode("GUIDED")
        self._connection.mav.command_long_send(
            self._target_system,
            self._target_component,
            mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,      # confirmation
            1,      # arm
            0, 0, 0, 0, 0, 0,
        )
        start = time.time()
        while time.time() - start < timeout:
            msg = self._connection.recv_match(
                type="HEARTBEAT", blocking=True, timeout=1
            )
            if msg and (msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                logger.info("Vehicle armed.")
                return
        raise TimeoutError("Vehicle did not arm within %d s" % timeout)

    def disarm(self) -> None:
        """Disarm the vehicle."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] disarm")
            return

        self._connection.mav.command_long_send(
            self._target_system,
            self._target_component,
            mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,      # disarm
            0, 0, 0, 0, 0, 0,
        )
        logger.info("Disarm command sent.")

    # ------------------------------------------------------------------
    # Takeoff / Land
    # ------------------------------------------------------------------

    def takeoff(self, altitude: float = 5.0) -> None:
        """Command the vehicle to take off to *altitude* metres (relative)."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] takeoff to %.1f m", altitude)
            return

        self._connection.mav.command_long_send(
            self._target_system,
            self._target_component,
            mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,   # pitch, empty, empty, yaw
            0, 0,          # lat, lon (0 = current)
            altitude,
        )
        logger.info("Takeoff to %.1f m commanded.", altitude)

    def land(self) -> None:
        """Command the vehicle to land at the current position."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] land")
            return

        self._connection.mav.command_long_send(
            self._target_system,
            self._target_component,
            mavlink.MAV_CMD_NAV_LAND,
            0,
            0, 0, 0, 0,
            0, 0, 0,
        )
        logger.info("Land command sent.")

    def return_to_launch(self) -> None:
        """Command the vehicle to return to the launch point."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] return_to_launch")
            return

        self.set_mode("RTL")
        logger.info("RTL mode engaged.")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def move_to(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> None:
        """Fly to a global GPS waypoint.

        Parameters
        ----------
        latitude, longitude:
            Target position in decimal degrees.
        altitude:
            Target altitude in metres above home (relative).
        """
        if not _MAVLINK_AVAILABLE:
            logger.info(
                "[STUB] move_to lat=%.6f lon=%.6f alt=%.1f",
                latitude, longitude, altitude,
            )
            return

        self._connection.mav.set_position_target_global_int_send(
            0,                          # time_boot_ms (ignored)
            self._target_system,
            self._target_component,
            mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            # type_mask: only position is used (ignore vel, acc, yaw)
            0b110111111000,
            int(latitude * 1e7),
            int(longitude * 1e7),
            altitude,
            0, 0, 0,   # vx, vy, vz
            0, 0, 0,   # afx, afy, afz
            0, 0,      # yaw, yaw_rate
        )
        logger.info(
            "Moving to lat=%.6f lon=%.6f alt=%.1f m",
            latitude, longitude, altitude,
        )

    def set_airspeed(self, airspeed: float) -> None:
        """Set the target airspeed in m/s."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] set_airspeed %.1f m/s", airspeed)
            return

        self._connection.mav.command_long_send(
            self._target_system,
            self._target_component,
            mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0,
            0,          # speed type: 0 = airspeed
            airspeed,
            -1,         # throttle (-1 = no change)
            0, 0, 0, 0,
        )
        logger.info("Airspeed set to %.1f m/s", airspeed)

    def hover(self, duration: float = 3.0) -> None:
        """Hold position for *duration* seconds (LOITER mode)."""
        if not _MAVLINK_AVAILABLE:
            logger.info("[STUB] hover for %.1f s", duration)
            time.sleep(duration)
            return

        self.set_mode("LOITER")
        time.sleep(duration)
        self.set_mode("GUIDED")

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_telemetry(self) -> Telemetry:
        """Return a :class:`Telemetry` snapshot from the latest MAVLink messages."""
        telem = Telemetry()

        if not _MAVLINK_AVAILABLE or self._connection is None:
            logger.debug("[STUB] get_telemetry returning default values")
            return telem

        # GPS position
        gps_msg = self._connection.recv_match(
            type="GLOBAL_POSITION_INT", blocking=True, timeout=2
        )
        if gps_msg:
            telem.latitude = gps_msg.lat / 1e7
            telem.longitude = gps_msg.lon / 1e7
            telem.altitude = gps_msg.relative_alt / 1000.0  # mm → m
            telem.heading = gps_msg.hdg / 100.0

        # Airspeed / groundspeed
        vfr_msg = self._connection.recv_match(
            type="VFR_HUD", blocking=True, timeout=2
        )
        if vfr_msg:
            telem.airspeed = vfr_msg.airspeed
            telem.groundspeed = vfr_msg.groundspeed

        # Battery
        bat_msg = self._connection.recv_match(
            type="BATTERY_STATUS", blocking=True, timeout=2
        )
        if bat_msg:
            if bat_msg.voltages[0] != 65535:
                telem.battery_voltage = bat_msg.voltages[0] / 1000.0
            telem.battery_remaining = bat_msg.battery_remaining

        # Armed status & mode
        hb_msg = self._connection.recv_match(
            type="HEARTBEAT", blocking=True, timeout=2
        )
        if hb_msg:
            telem.armed = bool(
                hb_msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            telem.mode = mavutil.mode_string_v10(hb_msg)

        return telem

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MAVLinkController":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()
