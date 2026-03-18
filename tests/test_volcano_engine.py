"""Tests for the Volcano Engine LLM client."""

import pytest
from unittest.mock import MagicMock, patch
from src.llm.volcano_engine import VolcanoLLMClient


class TestVolcanoLLMClientStub:
    """Tests that run without any real API credentials."""

    def test_instantiation_no_creds(self):
        client = VolcanoLLMClient(api_key="", endpoint_id="")
        assert client is not None

    def test_chat_returns_stub_reply(self):
        client = VolcanoLLMClient(api_key="", endpoint_id="")
        reply = client.chat("起飞到10米")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_history_accumulates(self):
        client = VolcanoLLMClient(api_key="", endpoint_id="")
        client.chat("请起飞")
        client.chat("悬停3秒")
        history = client.history
        assert len(history) == 4  # 2 user + 2 assistant
        roles = [m["role"] for m in history]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_reset_history(self):
        client = VolcanoLLMClient(api_key="", endpoint_id="")
        client.chat("测试")
        client.reset_history()
        assert client.history == []

    def test_history_is_copy(self):
        client = VolcanoLLMClient(api_key="", endpoint_id="")
        client.chat("测试")
        h1 = client.history
        h2 = client.history
        assert h1 == h2
        assert h1 is not h2  # must be independent copies

    def test_chat_stores_user_message(self):
        client = VolcanoLLMClient(api_key="", endpoint_id="")
        client.chat("任务：搜索目标")
        assert any(
            m["role"] == "user" and "任务：搜索目标" in m["content"]
            for m in client.history
        )

    def test_chat_stores_assistant_reply(self):
        client = VolcanoLLMClient(api_key="", endpoint_id="")
        reply = client.chat("任务：搜索目标")
        assert any(
            m["role"] == "assistant" and m["content"] == reply
            for m in client.history
        )


class TestVolcanoLLMClientWithMockedAPI:
    """Tests that exercise the API path via mocking."""

    def _make_mock_response(self, content: str):
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_chat_calls_api_when_configured(self):
        client = VolcanoLLMClient(api_key="test-key", endpoint_id="ep-test-0001")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_mock_response("飞往目标坐标")
        )
        # Directly inject the mock API client and enable API path
        client._client = mock_client
        with patch("src.llm.volcano_engine._ARK_AVAILABLE", True):
            reply = client.chat("请规划飞行任务")
        assert reply == "飞往目标坐标"
        mock_client.chat.completions.create.assert_called_once()

    def test_build_messages_includes_system_prompt(self):
        client = VolcanoLLMClient(api_key="", endpoint_id="")
        client._history = [{"role": "user", "content": "你好"}]
        messages = client._build_messages()
        assert messages[0]["role"] == "system"
        assert len(messages) == 2

    def test_api_error_propagates(self):
        client = VolcanoLLMClient(api_key="test-key", endpoint_id="ep-test-0001")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        client._client = mock_client
        with patch("src.llm.volcano_engine._ARK_AVAILABLE", True):
            with pytest.raises(RuntimeError, match="API error"):
                client.chat("test")
