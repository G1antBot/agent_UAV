"""Tests for the closed-loop Orchestrator."""

import pytest
from unittest.mock import MagicMock, patch
from src.orchestrator import Orchestrator
from src.uav.simulator import UAVSimulator
from src.llm.volcano_engine import VolcanoLLMClient
from src.perception.grounding_dino import GroundingDINODetector


class TestOrchestratorInitialisation:
    def test_default_uses_simulator(self):
        orch = Orchestrator(use_simulator=True)
        assert isinstance(orch.uav, UAVSimulator)

    def test_llm_is_volcano_client(self):
        orch = Orchestrator(use_simulator=True)
        assert isinstance(orch.llm, VolcanoLLMClient)

    def test_detector_is_grounding_dino(self):
        orch = Orchestrator(use_simulator=True)
        assert isinstance(orch.detector, GroundingDINODetector)

    def test_load_config_missing_file(self):
        # Should not raise; uses defaults
        orch = Orchestrator(config_path="/nonexistent/path.yaml", use_simulator=True)
        assert orch.uav is not None

    def test_context_manager(self):
        with Orchestrator(use_simulator=True) as orch:
            assert orch.uav is not None
        assert not orch.uav.is_connected()


class TestOrchestratorMission:
    def test_run_mission_connects_and_disconnects(self):
        orch = Orchestrator(use_simulator=True)

        connected_during: list[bool] = []

        original_run = orch.agent.run

        def patched_run(task: str) -> str:
            connected_during.append(orch.uav.is_connected())
            return "Mission complete"

        orch.agent.run = patched_run

        orch.run_mission("起飞到5米")
        # Was connected during the mission
        assert connected_during == [True]
        # Disconnected after
        assert not orch.uav.is_connected()

    def test_run_mission_returns_string(self):
        orch = Orchestrator(use_simulator=True)
        orch.agent.run = lambda task: "任务完成"
        result = orch.run_mission("测试任务")
        assert isinstance(result, str)

    def test_run_mission_disconnects_on_exception(self):
        orch = Orchestrator(use_simulator=True)
        orch.agent.run = MagicMock(side_effect=RuntimeError("Agent failed"))
        with pytest.raises(RuntimeError, match="Agent failed"):
            orch.run_mission("故障任务")
        assert not orch.uav.is_connected()


class TestOrchestratorConfigLoading:
    def test_load_config_valid_file(self, tmp_path):
        config_content = """\
llm:
  api_key: "test-key"
  endpoint_id: "ep-test"
mavlink:
  connection_string: "udp:127.0.0.1:14550"
grounding_dino:
  box_threshold: 0.4
agent:
  max_steps: 10
  verbose: false
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(config_content)
        orch = Orchestrator(config_path=str(cfg_file), use_simulator=True)
        # Verify config values were applied
        assert orch.detector.box_threshold == pytest.approx(0.4)

    def test_env_var_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "secret-key-123")
        config_content = """\
llm:
  api_key: "${TEST_API_KEY}"
  endpoint_id: "ep-0001"
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(config_content)
        orch = Orchestrator(config_path=str(cfg_file), use_simulator=True)
        assert orch.llm._api_key == "secret-key-123"
