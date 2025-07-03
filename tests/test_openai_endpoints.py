"""Tests for OpenAI-compatible endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from claude_proxy.main import app
from claude_proxy.models.openai_models import (
    OpenAIChatCompletionRequest,
    OpenAIMessage,
    OpenAIModelsResponse,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestOpenAIModelsEndpoint:
    """Test OpenAI models endpoint."""

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

        # Check first model structure
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert "created" in model
        assert "owned_by" in model
        assert model["object"] == "model"


class TestOpenAIChatCompletionsEndpoint:
    """Test OpenAI chat completions endpoint."""

    @pytest.fixture
    def mock_claude_client(self):
        """Mock Claude client."""
        with patch("claude_proxy.api.openai.chat.ClaudeClient") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def sample_request(self):
        """Sample OpenAI chat completion request."""
        return {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 100,
            "temperature": 0.7,
        }

    @pytest.fixture
    def sample_claude_response(self):
        """Sample Claude response."""
        return {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello! I'm doing well, thank you for asking."}
            ],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }

    def test_chat_completion_non_streaming(
        self, client, mock_claude_client, sample_request, sample_claude_response
    ):
        """Test non-streaming chat completion."""
        mock_claude_client.complete = AsyncMock(return_value=sample_claude_response)

        response = client.post("/v1/chat/completions", json=sample_request)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

        # Check choices structure
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert (
            choice["message"]["content"]
            == "Hello! I'm doing well, thank you for asking."
        )
        assert choice["finish_reason"] == "stop"

        # Check usage
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 15
        assert data["usage"]["total_tokens"] == 25

    def test_chat_completion_streaming(
        self, client, mock_claude_client, sample_request
    ):
        """Test streaming chat completion."""

        # Mock streaming response
        async def mock_stream():
            yield {"type": "message_start", "message": {"role": "assistant"}}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello! "},
            }
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "I'm doing well."},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 10, "output_tokens": 15},
            }

        mock_claude_client.stream_completion = AsyncMock(return_value=mock_stream())

        request_data = {**sample_request, "stream": True}
        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_completion_with_tools(
        self, client, mock_claude_client, sample_claude_response
    ):
        """Test chat completion with tool calls."""
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather like?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
        }

        # Mock response with tool use
        tool_response = {
            **sample_claude_response,
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "New York"},
                }
            ],
            "stop_reason": "tool_use",
        }

        mock_claude_client.complete = AsyncMock(return_value=tool_response)

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"] is not None
        assert len(choice["message"]["tool_calls"]) == 1

        tool_call = choice["message"]["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"

    def test_chat_completion_invalid_request(self, client):
        """Test chat completion with invalid request."""
        invalid_request = {
            "model": "gpt-4",
            "messages": [],  # Empty messages should fail validation
        }

        response = client.post("/v1/chat/completions", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_chat_completion_client_error(
        self, client, mock_claude_client, sample_request
    ):
        """Test chat completion with Claude client error."""
        from claude_proxy.exceptions import ClaudeProxyError

        mock_claude_client.complete = AsyncMock(
            side_effect=ClaudeProxyError("API rate limit exceeded")
        )

        response = client.post("/v1/chat/completions", json=sample_request)
        assert response.status_code == 500

        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["type"] == "api_error"

    def test_model_passthrough(
        self, client, mock_claude_client, sample_claude_response
    ):
        """Test that model names are passed through as-is."""
        mock_claude_client.complete = AsyncMock(return_value=sample_claude_response)

        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        # Verify the call was made with the same model name
        args, kwargs = mock_claude_client.complete.call_args
        assert kwargs["model"] == "claude-3-opus-20240229"

        # Verify response contains the same model name
        data = response.json()
        assert data["model"] == "claude-3-opus-20240229"


class TestOpenAIRequestValidation:
    """Test OpenAI request validation."""

    def test_openai_message_validation(self):
        """Test OpenAI message validation."""
        # Valid message
        valid_message = OpenAIMessage(role="user", content="Hello, world!")
        assert valid_message.role == "user"
        assert valid_message.content == "Hello, world!"

        # Message with tool calls
        tool_message = OpenAIMessage(
            role="assistant",
            content="I'll help you with that.",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "New York"}',
                    },
                }
            ],
        )
        assert tool_message.tool_calls is not None
        assert len(tool_message.tool_calls) == 1

    def test_openai_request_validation(self):
        """Test OpenAI request validation."""
        # Valid request
        valid_request = OpenAIChatCompletionRequest(
            model="gpt-4", messages=[OpenAIMessage(role="user", content="Hello")]
        )
        assert valid_request.model == "gpt-4"
        assert len(valid_request.messages) == 1

        # Request with parameters
        param_request = OpenAIChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[OpenAIMessage(role="user", content="Hello")],
            max_tokens=150,
            temperature=0.8,
            stream=True,
        )
        assert param_request.max_tokens == 150
        assert param_request.temperature == 0.8
        assert param_request.stream is True

    def test_model_validation(self):
        """Test model validation - should pass through as-is."""
        # Test Claude model passes through unchanged
        claude_request = OpenAIChatCompletionRequest(
            model="claude-3-sonnet-20240229",
            messages=[OpenAIMessage(role="user", content="Hello")],
        )
        assert claude_request.model == "claude-3-sonnet-20240229"

        # Test any model name passes through as-is
        any_request = OpenAIChatCompletionRequest(
            model="any-model-name",
            messages=[OpenAIMessage(role="user", content="Hello")],
        )
        assert any_request.model == "any-model-name"


class TestOpenAIResponseGeneration:
    """Test OpenAI response generation."""

    def test_models_response_generation(self):
        """Test models response generation."""
        response = OpenAIModelsResponse.create_default()

        assert response.object == "list"
        assert isinstance(response.data, list)
        assert len(response.data) > 0

        # Check model structure
        for model in response.data:
            assert hasattr(model, "id")
            assert hasattr(model, "object")
            assert hasattr(model, "created")
            assert hasattr(model, "owned_by")
            assert model.object == "model"
            assert model.owned_by == "anthropic"
