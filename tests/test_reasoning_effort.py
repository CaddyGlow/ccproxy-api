"""Tests for OpenAI reasoning effort and developer role support."""

import pytest

from ccproxy.models.openai import OpenAIChatCompletionRequest, OpenAIMessage
from ccproxy.services.translator import OpenAITranslator


@pytest.mark.unit
class TestReasoningEffort:
    """Test reasoning effort parameter handling."""

    def test_reasoning_effort_low(self):
        """Test low reasoning effort converts to thinking tokens."""
        translator = OpenAITranslator()

        request = {
            "model": "o1-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "low",
        }

        result = translator.openai_to_anthropic_request(request)

        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 1000

    def test_reasoning_effort_medium(self):
        """Test medium reasoning effort converts to thinking tokens."""
        translator = OpenAITranslator()

        request = {
            "model": "o1-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "medium",
        }

        result = translator.openai_to_anthropic_request(request)

        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 5000

    def test_reasoning_effort_high(self):
        """Test high reasoning effort converts to thinking tokens."""
        translator = OpenAITranslator()

        request = {
            "model": "o1-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "high",
        }

        result = translator.openai_to_anthropic_request(request)

        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 10000

    def test_no_reasoning_effort(self):
        """Test that no reasoning effort means no thinking configuration."""
        translator = OpenAITranslator()

        request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = translator.openai_to_anthropic_request(request)

        assert "thinking" not in result


@pytest.mark.unit
class TestDeveloperRole:
    """Test developer role message handling."""

    def test_developer_role_becomes_system(self):
        """Test developer role messages become system prompts."""
        translator = OpenAITranslator()

        request = {
            "model": "o1-mini",
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        }

        result = translator.openai_to_anthropic_request(request)

        assert result["system"] == "You are a helpful assistant."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_developer_and_system_roles_combined(self):
        """Test developer and system roles are combined."""
        translator = OpenAITranslator()

        request = {
            "model": "o1-mini",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "developer", "content": "Always be concise."},
                {"role": "user", "content": "Hello"},
            ],
        }

        result = translator.openai_to_anthropic_request(request)

        assert result["system"] == "You are helpful.\nAlways be concise."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_multiple_developer_messages(self):
        """Test multiple developer messages are concatenated."""
        translator = OpenAITranslator()

        request = {
            "model": "o1-mini",
            "messages": [
                {"role": "developer", "content": "Rule 1: Be helpful."},
                {"role": "developer", "content": "Rule 2: Be concise."},
                {"role": "user", "content": "Hello"},
            ],
        }

        result = translator.openai_to_anthropic_request(request)

        assert result["system"] == "Rule 1: Be helpful.\nRule 2: Be concise."
        assert len(result["messages"]) == 1

    def test_openai_message_model_accepts_developer_role(self):
        """Test that OpenAIMessage model accepts developer role."""
        message = OpenAIMessage(
            role="developer", content="You are a helpful assistant."
        )

        assert message.role == "developer"
        assert message.content == "You are a helpful assistant."

    def test_openai_request_with_developer_role(self):
        """Test OpenAIChatCompletionRequest accepts developer role."""
        request = OpenAIChatCompletionRequest(
            model="o1-mini",
            messages=[
                OpenAIMessage(role="developer", content="Be helpful"),
                OpenAIMessage(role="user", content="Hello"),
            ],
        )

        assert request.messages[0].role == "developer"
        assert request.messages[1].role == "user"
