"""Tests for the UAV Simulator."""

import pytest
from src.uav.simulator import UAVSimulator, SimulatorState
from src.uav.mavlink_controller import Telemetry


class TestUAVSimulatorConnection:
    def test_connect_and_disconnect(self):
        sim = UAVSimulator()
        assert not sim.is_connected()
        sim.connect()
        assert sim.is_connected()
        sim.disconnect()
        assert not sim.is_connected()

    def test_context_manager(self):
        with UAVSimulator() as sim:
            assert sim.is_connected()
        assert not sim.is_connected()


class TestUAVSimulatorArming:
    def test_arm(self):
        sim = UAVSimulator()
        sim.connect()
        sim.arm()
        assert sim.state.armed is True
        assert sim.state.mode == "GUIDED"

    def test_disarm(self):
        sim = UAVSimulator()
        sim.connect()
        sim.arm()
        sim.disarm()
        assert sim.state.armed is False

    def test_arm_sets_guided_mode(self):
        sim = UAVSimulator()
        sim.connect()
        sim.arm()
        assert sim.state.mode == "GUIDED"


class TestUAVSimulatorFlight:
    def setup_method(self):
        self.sim = UAVSimulator()
        self.sim.connect()
        self.sim.arm()

    def test_takeoff_updates_altitude(self):
        self.sim.takeoff(altitude=10.0)
        assert self.sim.state.altitude == pytest.approx(10.0)

    def test_takeoff_requires_armed(self):
        sim = UAVSimulator()
        sim.connect()
        with pytest.raises(RuntimeError, match="not armed"):
            sim.takeoff()

    def test_land_resets_altitude(self):
        self.sim.takeoff(10.0)
        self.sim.land()
        assert self.sim.state.altitude == pytest.approx(0.0)
        assert self.sim.state.armed is False

    def test_move_to_updates_position(self):
        self.sim.takeoff(5.0)
        self.sim.move_to(37.7749, -122.4194, 15.0)
        assert self.sim.state.latitude == pytest.approx(37.7749)
        assert self.sim.state.longitude == pytest.approx(-122.4194)
        assert self.sim.state.altitude == pytest.approx(15.0)

    def test_set_airspeed(self):
        self.sim.set_airspeed(5.0)
        assert self.sim.state.airspeed == pytest.approx(5.0)

    def test_return_to_launch_resets_position(self):
        home_lat = self.sim.state.home_latitude
        home_lon = self.sim.state.home_longitude
        self.sim.takeoff(10.0)
        self.sim.move_to(37.7749, -122.4194, 10.0)
        self.sim.return_to_launch()
        assert self.sim.state.latitude == pytest.approx(home_lat)
        assert self.sim.state.longitude == pytest.approx(home_lon)
        assert self.sim.state.altitude == pytest.approx(0.0)


class TestUAVSimulatorTelemetry:
    def test_get_telemetry_returns_telemetry(self):
        with UAVSimulator() as sim:
            t = sim.get_telemetry()
        assert isinstance(t, Telemetry)

    def test_telemetry_reflects_state(self):
        sim = UAVSimulator()
        sim.connect()
        sim.arm()
        sim.takeoff(7.5)
        t = sim.get_telemetry()
        assert t.altitude == pytest.approx(7.5)
        assert t.armed is True

    def test_battery_drain(self):
        import time
        sim = UAVSimulator(battery_drain_per_second=10.0)
        sim.connect()
        sim.arm()
        initial_battery = sim.state.battery_remaining
        # Simulate elapsed time by manually updating _last_telem_time
        sim._last_telem_time -= 5  # pretend 5 seconds have passed
        t = sim.get_telemetry()
        assert t.battery_remaining < initial_battery


class TestUAVSimulatorCommandLog:
    def test_command_log_records_actions(self):
        sim = UAVSimulator()
        sim.connect()
        sim.arm()
        sim.takeoff(5.0)
        sim.land()
        commands = [entry[0] for entry in sim.command_log]
        assert "arm" in commands
        assert "takeoff" in commands
        assert "land" in commands

    def test_clear_command_log(self):
        sim = UAVSimulator()
        sim.connect()
        sim.arm()
        sim.clear_command_log()
        assert sim.command_log == []


class TestSimulatorState:
    def test_custom_initial_state(self):
        state = SimulatorState(latitude=51.5, longitude=-0.1, altitude=100.0)
        sim = UAVSimulator(initial_state=state)
        assert sim.state.latitude == pytest.approx(51.5)
        assert sim.state.longitude == pytest.approx(-0.1)
        assert sim.state.altitude == pytest.approx(100.0)
