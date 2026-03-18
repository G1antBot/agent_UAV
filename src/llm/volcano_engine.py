"""
Volcano Engine LLM Client (火山方舟 Ark)
==========================================
Thin wrapper around the Volcano Engine Ark API, which exposes an
OpenAI-compatible chat-completions endpoint.

Typical usage::

    client = VolcanoLLMClient(api_key="...", endpoint_id="ep-xxxx-yyyy")
    response = client.chat("请分析当前无人机状态并给出下一步飞行指令")
    print(response)

The client maintains a conversation history so that multi-turn interactions
(e.g. follow-up clarifications or iterative mission planning) work
correctly out of the box.

Endpoint configuration
----------------------
After registering on the Volcano Engine console (https://www.volcengine.com/),
create a "推理接入点" (inference endpoint) for a Doubao model and copy the
endpoint ID (format: ``ep-<digits>-<letters>``).  Set it as the
``VOLC_ENDPOINT_ID`` environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from volcenginesdkarkruntime import Ark
    _ARK_AVAILABLE = True
except ImportError:
    _ARK_AVAILABLE = False
    logger.warning(
        "volcengine-python-sdk not installed – VolcanoLLMClient runs in stub mode."
    )


_SYSTEM_PROMPT = """你是一个无人机任务规划助手。
你的职责是：
1. 理解用户的自然语言飞行任务需求；
2. 将任务分解为具体的无人机控制指令序列；
3. 结合环境感知信息（目标检测结果）做出决策；
4. 以JSON格式输出结构化的飞行计划或直接给出自然语言分析。

可用的无人机控制动作：
- arm()                          – 解锁电机
- disarm()                       – 锁定电机
- takeoff(altitude)              – 起飞到指定高度（米）
- land()                         – 就地降落
- return_to_launch()             – 返回起飞点
- move_to(lat, lon, alt)         – 飞往GPS坐标
- set_airspeed(speed)            – 设置飞行速度（m/s）
- hover(duration)                – 悬停指定秒数
- detect_objects(prompt)         – 使用Grounding DINO检测指定目标
- get_telemetry()                – 获取当前无人机状态

请始终优先考虑飞行安全，若任务存在安全隐患请明确说明。
"""


class VolcanoLLMClient:
    """Chat client for Volcano Engine Ark (火山方舟) LLM.

    Parameters
    ----------
    api_key:
        Volcano Engine API key.  Falls back to the ``VOLC_ACCESSKEY``
        environment variable when not provided.
    endpoint_id:
        Ark inference endpoint ID (e.g. ``"ep-20240601123456-abcde"``).
        Falls back to ``VOLC_ENDPOINT_ID``.
    base_url:
        Custom base URL for the Ark API.
    max_tokens:
        Maximum number of tokens in each completion response.
    temperature:
        Sampling temperature (0 = deterministic, 1 = creative).
    system_prompt:
        System-level instruction for the assistant.  Defaults to the
        built-in UAV planning prompt.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self._api_key = api_key or os.environ.get("VOLC_ACCESSKEY", "")
        self._endpoint_id = endpoint_id or os.environ.get("VOLC_ENDPOINT_ID", "")
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt

        # Conversation history (list of {"role": ..., "content": ...})
        self._history: list[dict[str, str]] = []

        if _ARK_AVAILABLE and self._api_key:
            self._client = Ark(api_key=self._api_key, base_url=self._base_url)
        else:
            self._client = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send *user_message* and return the assistant's reply.

        The full conversation history (including the system prompt) is
        sent with each request so the model has context.
        """
        self._history.append({"role": "user", "content": user_message})

        if self._client is None or not self._endpoint_id:
            reply = self._stub_reply(user_message)
        else:
            reply = self._call_api()

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset_history(self) -> None:
        """Clear the conversation history."""
        self._history = []

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict[str, str]]:
        """Combine system prompt with conversation history."""
        return [
            {"role": "system", "content": self._system_prompt},
            *self._history,
        ]

    def _call_api(self) -> str:
        """Call the Ark API and return the assistant message content."""
        messages = self._build_messages()
        logger.debug("Calling Ark endpoint %s", self._endpoint_id)
        try:
            response = self._client.chat.completions.create(
                model=self._endpoint_id,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("Ark API call failed: %s", exc)
            raise

    def _stub_reply(self, message: str) -> str:
        """Return a canned stub reply when the SDK is unavailable.

        This ensures the rest of the pipeline can run end-to-end in
        environments where no LLM credentials are configured.
        """
        logger.info("[STUB] LLM received: %s", message)
        return (
            "[STUB LLM] 收到任务请求。"
            "在真实环境中，此处将返回火山方舟大模型的决策输出。"
        )
