"""
UAV SmolAgent
==============
Integrates Hugging Face **SmolAgents** with the UAV control stack and the
Volcano Engine LLM.

SmolAgents (https://huggingface.co/docs/smolagents) is a lightweight
code-agent framework that:

1. Receives a natural-language task description.
2. Uses a code-generation LLM to produce Python action code.
3. Executes the code in a sandboxed environment.
4. Iterates until the task is complete or ``max_steps`` is exhausted.

Here we register UAV-specific *tools* so the agent can call them:

+------------------------+--------------------------------------------+
| Tool name              | Description                                |
+========================+============================================+
| ``arm``                | Arm the UAV motors                         |
| ``disarm``             | Disarm the UAV motors                      |
| ``takeoff``            | Take off to given altitude (m)             |
| ``land``               | Land at current position                   |
| ``return_to_launch``   | Fly back to the home point                 |
| ``move_to``            | Fly to a GPS waypoint                      |
| ``set_airspeed``       | Set airspeed in m/s                        |
| ``hover``              | Hold position for N seconds                |
| ``detect_objects``     | Run Grounding DINO on the latest frame     |
| ``get_telemetry``      | Read current UAV state                     |
+------------------------+--------------------------------------------+

The LLM backend used by SmolAgents is a thin adapter that forwards
requests to :class:`~src.llm.volcano_engine.VolcanoLLMClient`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from smolagents import CodeAgent, tool, LiteLLMModel
    _SMOLAGENTS_AVAILABLE = True
except ImportError:
    _SMOLAGENTS_AVAILABLE = False
    logger.warning(
        "smolagents not installed – UAVAgent will use a fallback implementation."
    )


# ---------------------------------------------------------------------------
# Volcano Engine adapter for smolagents
# ---------------------------------------------------------------------------

class VolcanoSmolagentsModel:
    """Adapter that makes :class:`VolcanoLLMClient` look like a smolagents
    ``Model`` so it can be plugged into ``CodeAgent``.

    smolagents expects a callable ``Model`` object that accepts a list of
    messages and returns a string.
    """

    def __init__(self, volcano_client: Any) -> None:
        self._client = volcano_client

    def __call__(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        # Extract the last user message
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        prompt = user_msgs[-1] if user_msgs else ""
        return self._client.chat(prompt)

    # smolagents checks for this attribute
    @property
    def last_input_token_count(self) -> int:
        return 0

    @property
    def last_output_token_count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# UAVAgent
# ---------------------------------------------------------------------------

class UAVAgent:
    """SmolAgents-based UAV task agent.

    Parameters
    ----------
    uav_controller:
        A :class:`~src.uav.mavlink_controller.MAVLinkController` or
        :class:`~src.uav.simulator.UAVSimulator` instance.
    detector:
        A :class:`~src.perception.grounding_dino.GroundingDINODetector`
        instance (may be ``None`` if detection is not needed).
    llm_client:
        A :class:`~src.llm.volcano_engine.VolcanoLLMClient` instance.
    max_steps:
        Maximum number of agent reasoning steps per task.
    verbose:
        If ``True``, print intermediate agent output to stdout.
    """

    def __init__(
        self,
        uav_controller: Any,
        detector: Any,
        llm_client: Any,
        max_steps: int = 20,
        verbose: bool = True,
    ) -> None:
        self._uav = uav_controller
        self._detector = detector
        self._llm = llm_client
        self._max_steps = max_steps
        self._verbose = verbose
        self._agent = None
        self._build_agent()

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _build_agent(self) -> None:
        """Create the SmolAgents ``CodeAgent`` with all UAV tools registered."""
        if not _SMOLAGENTS_AVAILABLE:
            logger.warning("[STUB] SmolAgents not available; agent in stub mode.")
            return

        tools = self._create_tools()
        model = VolcanoSmolagentsModel(self._llm)
        self._agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=self._max_steps,
            verbosity_level=1 if self._verbose else 0,
        )
        logger.info("UAVAgent built with %d tools.", len(tools))

    def _create_tools(self) -> list:
        """Return a list of smolagents Tool objects for UAV actions."""
        uav = self._uav
        detector = self._detector

        @tool
        def arm() -> str:
            """Arm the UAV motors. Must be called before takeoff."""
            uav.arm()
            return "Motors armed."

        @tool
        def disarm() -> str:
            """Disarm the UAV motors."""
            uav.disarm()
            return "Motors disarmed."

        @tool
        def takeoff(altitude: float) -> str:
            """Take off to the given altitude in metres above home."""
            uav.takeoff(altitude=altitude)
            return f"Taking off to {altitude} m."

        @tool
        def land() -> str:
            """Land the UAV at its current position."""
            uav.land()
            return "Landing initiated."

        @tool
        def return_to_launch() -> str:
            """Return the UAV to its launch point."""
            uav.return_to_launch()
            return "Returning to launch."

        @tool
        def move_to(latitude: float, longitude: float, altitude: float) -> str:
            """Fly to a GPS waypoint.

            Args:
                latitude: Target latitude in decimal degrees.
                longitude: Target longitude in decimal degrees.
                altitude: Target altitude in metres above home.
            """
            uav.move_to(latitude=latitude, longitude=longitude, altitude=altitude)
            return f"Moving to ({latitude:.6f}, {longitude:.6f}) at {altitude} m."

        @tool
        def set_airspeed(speed: float) -> str:
            """Set the UAV airspeed in m/s.

            Args:
                speed: Target airspeed in metres per second.
            """
            uav.set_airspeed(airspeed=speed)
            return f"Airspeed set to {speed} m/s."

        @tool
        def hover(duration: float) -> str:
            """Hold the UAV position for the given number of seconds.

            Args:
                duration: Hover duration in seconds.
            """
            uav.hover(duration=duration)
            return f"Hovering for {duration} s."

        @tool
        def detect_objects(prompt: str) -> str:
            """Run open-vocabulary object detection on the current camera frame.

            Args:
                prompt: Dot-separated list of categories to search for,
                        e.g. "person . red car . building".

            Returns a human-readable summary of detected objects.
            """
            if detector is None:
                return "Detector not available."
            telem = uav.get_telemetry()
            # In a real system the frame would come from a live camera feed.
            # Here we use a synthetic image path derived from the UAV position
            # which can be patched in tests.
            frame_path = getattr(detector, "_test_frame", None)
            if frame_path is None:
                return "No camera frame available."
            results = detector.detect(frame_path, prompt)
            return detector.summarise(results)

        @tool
        def get_telemetry() -> str:
            """Read the current UAV telemetry (position, altitude, speed, battery)."""
            t = uav.get_telemetry()
            return (
                f"lat={t.latitude:.6f}, lon={t.longitude:.6f}, "
                f"alt={t.altitude:.1f}m, heading={t.heading:.0f}°, "
                f"airspeed={t.airspeed:.1f}m/s, armed={t.armed}, "
                f"mode={t.mode}, battery={t.battery_remaining}%"
            )

        return [
            arm, disarm, takeoff, land, return_to_launch,
            move_to, set_airspeed, hover, detect_objects, get_telemetry,
        ]

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def run(self, task: str) -> str:
        """Execute a natural-language UAV task.

        Parameters
        ----------
        task:
            Free-form task description, e.g.
            ``"起飞到10米高度，向北飞行100米，检测是否有人，然后返航"``.

        Returns
        -------
        str
            Final agent output / mission report.
        """
        logger.info("UAVAgent.run: %s", task)

        if self._agent is None:
            # Fallback: ask the LLM directly and try to execute key phrases
            return self._fallback_run(task)

        result = self._agent.run(task)
        return str(result)

    def _fallback_run(self, task: str) -> str:
        """Simple LLM-only fallback when SmolAgents is not available."""
        logger.info("[FALLBACK] Running task via LLM only: %s", task)
        reply = self._llm.chat(
            f"请分析以下无人机任务并给出控制指令建议：\n{task}"
        )
        return reply
