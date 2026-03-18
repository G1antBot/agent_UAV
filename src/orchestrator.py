"""
Closed-Loop Orchestrator
========================
Ties together all subsystems into the "natural language → LLM decision →
UAV control → environment perception" closed loop described in the
project README.

Loop structure
--------------
::

    ┌────────────────────────────────────────────┐
    │  User / mission planner                    │
    │  (natural language task string)            │
    └────────────────┬───────────────────────────┘
                     │  task
                     ▼
    ┌────────────────────────────────────────────┐
    │  VolcanoLLMClient                          │
    │  Interprets intent, plans sub-goals        │
    └────────────────┬───────────────────────────┘
                     │  plan
                     ▼
    ┌────────────────────────────────────────────┐
    │  UAVAgent (SmolAgents CodeAgent)           │
    │  Translates plan → Python tool calls       │
    └──────────┬──────────────────┬──────────────┘
               │ control          │ detect
               ▼                  ▼
    ┌──────────────────┐  ┌──────────────────────┐
    │  MAVLinkControl  │  │  GroundingDINO        │
    │  (or Simulator)  │  │  (perception)         │
    └──────────┬───────┘  └──────────┬────────────┘
               │ telemetry            │ detections
               └──────────┬──────────┘
                           │  feedback
                           ▼
    ┌────────────────────────────────────────────┐
    │  VolcanoLLMClient                          │
    │  Re-evaluates and issues next sub-goal     │
    └────────────────────────────────────────────┘

The loop runs until:
  * the agent reports task completion, **or**
  * ``max_iterations`` is reached, **or**
  * a safety condition is triggered (e.g. low battery).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import yaml

from src.uav.mavlink_controller import MAVLinkController
from src.uav.simulator import UAVSimulator
from src.perception.grounding_dino import GroundingDINODetector
from src.llm.volcano_engine import VolcanoLLMClient
from src.agent.uav_agent import UAVAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """Closed-loop mission orchestrator.

    Parameters
    ----------
    config_path:
        Path to ``configs/config.yaml``.  When ``None``, all sub-systems
        are constructed with default / stub parameters.
    use_simulator:
        If ``True`` (default), uses :class:`UAVSimulator` instead of a
        real MAVLink connection.  Set to ``False`` for hardware flights.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_simulator: bool = True,
    ) -> None:
        self._cfg = self._load_config(config_path)
        self._use_simulator = use_simulator

        # Build sub-systems
        self.uav = self._build_uav()
        self.detector = self._build_detector()
        self.llm = self._build_llm()
        self.agent = self._build_agent()

        logger.info(
            "Orchestrator initialised (simulator=%s)", use_simulator
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Optional[str]) -> dict:
        if path is None:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("Config file not found: %s – using defaults.", path)
            return {}

    def _build_uav(self) -> Any:
        if self._use_simulator:
            logger.info("Using UAVSimulator (no hardware required).")
            return UAVSimulator()

        mav_cfg = self._cfg.get("mavlink", {})
        return MAVLinkController(
            connection_string=mav_cfg.get(
                "connection_string", "udp:127.0.0.1:14550"
            ),
            timeout=mav_cfg.get("timeout", 30),
        )

    def _build_detector(self) -> GroundingDINODetector:
        dino_cfg = self._cfg.get("grounding_dino", {})
        detector = GroundingDINODetector(
            config_path=dino_cfg.get(
                "config_path",
                "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ),
            weights_path=dino_cfg.get(
                "weights_path", "weights/groundingdino_swint_ogc.pth"
            ),
            box_threshold=dino_cfg.get("box_threshold", 0.35),
            text_threshold=dino_cfg.get("text_threshold", 0.25),
            device=dino_cfg.get("device", "cpu"),
        )
        # Attempt to load model; failures are non-fatal (stub mode)
        try:
            detector.load()
        except Exception as exc:
            logger.warning("Could not load Grounding DINO: %s", exc)
        return detector

    def _build_llm(self) -> VolcanoLLMClient:
        llm_cfg = self._cfg.get("llm", {})

        def _resolve(value: str) -> str:
            """Expand ${ENV_VAR} placeholders in config values."""
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                return os.environ.get(value[2:-1], "")
            return value or ""

        return VolcanoLLMClient(
            api_key=_resolve(llm_cfg.get("api_key", "")),
            endpoint_id=_resolve(llm_cfg.get("endpoint_id", "")),
            base_url=llm_cfg.get(
                "base_url", "https://ark.cn-beijing.volces.com/api/v3"
            ),
            max_tokens=llm_cfg.get("max_tokens", 2048),
            temperature=llm_cfg.get("temperature", 0.2),
        )

    def _build_agent(self) -> UAVAgent:
        agent_cfg = self._cfg.get("agent", {})
        return UAVAgent(
            uav_controller=self.uav,
            detector=self.detector,
            llm_client=self.llm,
            max_steps=agent_cfg.get("max_steps", 20),
            verbose=agent_cfg.get("verbose", True),
        )

    # ------------------------------------------------------------------
    # Mission execution
    # ------------------------------------------------------------------

    def run_mission(self, task: str) -> str:
        """Execute a full closed-loop mission from a natural-language task.

        Parameters
        ----------
        task:
            Natural-language mission description, e.g.
            ``"起飞到10米，前往坐标(37.7749,-122.4194)，检测红色车辆，返航"``.

        Returns
        -------
        str
            Mission report / final agent output.
        """
        logger.info("=== Mission START ===  task: %s", task)

        # Connect UAV (simulator or real hardware)
        self.uav.connect()

        try:
            report = self.agent.run(task)
        finally:
            self.uav.disconnect()

        logger.info("=== Mission END ===  report: %s", report)
        return report

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Orchestrator":
        return self

    def __exit__(self, *_) -> None:
        if self.uav.is_connected():
            self.uav.disconnect()
