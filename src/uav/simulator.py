"""
UAV Simulator
=============
A lightweight in-process UAV simulator that mirrors the
:class:`~src.uav.mavlink_controller.MAVLinkController` API.

The simulator is used in unit-tests and offline demos where a real flight
controller or SITL is unavailable.  It tracks virtual state (position,
altitude, mode, armed status …) and responds to commands instantly so
that higher-level code can be tested without any hardware dependency.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from src.uav.mavlink_controller import Telemetry

logger = logging.getLogger(__name__)

# Earth radius used for simple flat-earth displacements (metres per degree)
_METRES_PER_DEG_LAT = 111_320.0


@dataclass
class SimulatorState:
    """Mutable state of the virtual UAV."""
    latitude: float = -35.3632621    # ArduPilot SITL default home (Canberra)
    longitude: float = 149.1652374
    altitude: float = 0.0
    heading: float = 0.0
    airspeed: float = 0.0
    groundspeed: float = 0.0
    battery_voltage: float = 12.6
    battery_remaining: int = 100
    armed: bool = False
    mode: str = "STABILIZE"
    home_latitude: float = -35.3632621
    home_longitude: float = 149.1652374


class UAVSimulator:
    """In-process UAV simulator that mirrors :class:`MAVLinkController`.

    All movement commands update the internal :attr:`state` immediately (no
    physics integration).  A simple drain on battery is applied each time
    :meth:`get_telemetry` is called so that battery-related code can be
    tested.

    Parameters
    ----------
    initial_state:
        Optional :class:`SimulatorState` to seed the simulator with a
        specific starting position.
    battery_drain_per_second:
        Battery percentage drained per second of simulated flight time.
        Defaults to 0 (no drain) so tests are deterministic.
    """

    def __init__(
        self,
        initial_state: Optional[SimulatorState] = None,
        battery_drain_per_second: float = 0.0,
    ) -> None:
        self.state = initial_state or SimulatorState()
        self._battery_drain = battery_drain_per_second
        self._last_telem_time: float = time.time()
        self._connected: bool = False
        self._command_log: list[tuple[str, tuple]] = []

    # ------------------------------------------------------------------
    # Connection management (mirrors MAVLinkController API)
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True
        logger.info("[SIM] Connected")

    def disconnect(self) -> None:
        self._connected = False
        logger.info("[SIM] Disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self.state.mode = mode
        self._log("set_mode", mode)
        logger.info("[SIM] Mode → %s", mode)

    # ------------------------------------------------------------------
    # Arm / Disarm
    # ------------------------------------------------------------------

    def arm(self, timeout: int = 15) -> None:  # noqa: ARG002
        self.state.armed = True
        self.state.mode = "GUIDED"
        self._log("arm")
        logger.info("[SIM] Armed")

    def disarm(self) -> None:
        self.state.armed = False
        self._log("disarm")
        logger.info("[SIM] Disarmed")

    # ------------------------------------------------------------------
    # Takeoff / Land
    # ------------------------------------------------------------------

    def takeoff(self, altitude: float = 5.0) -> None:
        if not self.state.armed:
            raise RuntimeError("Cannot take off: vehicle is not armed.")
        self.state.altitude = float(altitude)
        self.state.airspeed = 3.0
        self.state.groundspeed = 0.0
        self._log("takeoff", altitude)
        logger.info("[SIM] Took off to %.1f m", altitude)

    def land(self) -> None:
        self.state.altitude = 0.0
        self.state.airspeed = 0.0
        self.state.groundspeed = 0.0
        self.state.armed = False
        self._log("land")
        logger.info("[SIM] Landed")

    def return_to_launch(self) -> None:
        self.state.latitude = self.state.home_latitude
        self.state.longitude = self.state.home_longitude
        self.state.altitude = 0.0
        self.state.armed = False
        self.state.mode = "RTL"
        self._log("return_to_launch")
        logger.info("[SIM] Returned to launch")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def move_to(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> None:
        """Instantly teleport the simulator to the given GPS waypoint."""
        # Update heading to face the target
        dlat = latitude - self.state.latitude
        dlon = longitude - self.state.longitude
        if dlat != 0 or dlon != 0:
            self.state.heading = (
                math.degrees(math.atan2(dlon, dlat)) % 360
            )
        self.state.latitude = float(latitude)
        self.state.longitude = float(longitude)
        self.state.altitude = float(altitude)
        self.state.groundspeed = self.state.airspeed
        self._log("move_to", latitude, longitude, altitude)
        logger.info(
            "[SIM] Moved to lat=%.6f lon=%.6f alt=%.1f m",
            latitude, longitude, altitude,
        )

    def set_airspeed(self, airspeed: float) -> None:
        self.state.airspeed = float(airspeed)
        self._log("set_airspeed", airspeed)
        logger.info("[SIM] Airspeed → %.1f m/s", airspeed)

    def hover(self, duration: float = 3.0) -> None:
        self.state.groundspeed = 0.0
        self._log("hover", duration)
        logger.info("[SIM] Hovering for %.1f s (simulated)", duration)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_telemetry(self) -> Telemetry:
        now = time.time()
        elapsed = now - self._last_telem_time
        self._last_telem_time = now

        # Apply battery drain
        if self.state.armed and self._battery_drain > 0:
            drain = self._battery_drain * elapsed
            self.state.battery_remaining = max(
                0, int(self.state.battery_remaining - drain)
            )

        return Telemetry(
            latitude=self.state.latitude,
            longitude=self.state.longitude,
            altitude=self.state.altitude,
            heading=self.state.heading,
            airspeed=self.state.airspeed,
            groundspeed=self.state.groundspeed,
            battery_voltage=self.state.battery_voltage,
            battery_remaining=self.state.battery_remaining,
            armed=self.state.armed,
            mode=self.state.mode,
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # Inspection helpers (for testing)
    # ------------------------------------------------------------------

    @property
    def command_log(self) -> list[tuple[str, tuple]]:
        """Return a copy of all commands issued since construction."""
        return list(self._command_log)

    def clear_command_log(self) -> None:
        self._command_log.clear()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "UAVSimulator":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, command: str, *args) -> None:
        self._command_log.append((command, args))
