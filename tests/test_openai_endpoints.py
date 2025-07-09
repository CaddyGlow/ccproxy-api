"""Tests for OpenAI-compatible endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ccproxy.formatters.translator import map_openai_model_to_claude

# Removed import of global app - using test_client fixture instead
from ccproxy.models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIMessage,
    OpenAIModelsResponse,
)


# Removed custom client fixture - using test_client from conftest.py instead


@pytest.mark.integration
class TestOpenAIModelsEndpoint:
    """Test OpenAI models endpoint."""

    def test_list_models(self, test_client):
        """Test listing available models."""
        response = test_client.get("/cc/openai/v1/models")
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


@pytest.mark.integration
class TestOpenAIChatCompletionsEndpoint:
    """Test OpenAI chat completions endpoint."""

    @pytest.fixture
    def mock_claude_client(self):
        """Mock Claude client directly."""
        with patch(
            "ccproxy.routers.claudecode.openai.ClaudeClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def sample_request(self):
        """Sample OpenAI chat completion request."""
        return {
            "model": "claude-opus-4-20250514",
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
            "model": "claude-opus-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }

    def test_chat_completion_non_streaming(
        self, test_client, mock_claude_client, sample_request, sample_claude_response
    ):
        """Test non-streaming chat completion."""
        mock_claude_client.create_completion = AsyncMock(
            return_value=sample_claude_response
        )

        response = test_client.post(
            "/cc/openai/v1/chat/completions", json=sample_request
        )
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
        self, test_client, mock_claude_client, sample_request
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

        mock_claude_client.create_completion = AsyncMock(return_value=mock_stream())

        request_data = {**sample_request, "stream": True}
        response = test_client.post("/cc/openai/v1/chat/completions", json=request_data)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_completion_with_tools(
        self, test_client, mock_claude_client, sample_claude_response
    ):
        """Test chat completion with tool calls."""
        request_data = {
            "model": "claude-opus-4-20250514",
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

        mock_claude_client.create_completion = AsyncMock(return_value=tool_response)

        response = test_client.post("/cc/openai/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        data = response.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"] is not None
        assert len(choice["message"]["tool_calls"]) == 1

        tool_call = choice["message"]["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"

    def test_chat_completion_invalid_request(self, test_client):
        """Test chat completion with invalid request."""
        invalid_request = {
            "model": "claude-opus-4-20250514",
            "messages": [],  # Empty messages should fail validation
        }

        response = test_client.post(
            "/cc/openai/v1/chat/completions", json=invalid_request
        )
        assert response.status_code == 422  # Validation error

    def test_chat_completion_client_error(
        self, test_client, mock_claude_client, sample_request
    ):
        """Test chat completion with Claude client error."""
        from ccproxy.exceptions import ClaudeProxyError

        mock_claude_client.create_completion = AsyncMock(
            side_effect=ClaudeProxyError("API rate limit exceeded")
        )

        response = test_client.post(
            "/cc/openai/v1/chat/completions", json=sample_request
        )
        assert response.status_code == 500

        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["type"] == "api_error"

    def test_model_passthrough(
        self, test_client, mock_claude_client, sample_claude_response
    ):
        """Test that model names are passed through as-is."""
        mock_claude_client.create_completion = AsyncMock(
            return_value=sample_claude_response
        )

        request_data = {
            "model": "claude-opus-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = test_client.post("/cc/openai/v1/chat/completions", json=request_data)
        assert response.status_code == 200

        # Verify the call was made with the same model name
        args, kwargs = mock_claude_client.create_completion.call_args
        assert args[1].model == "claude-opus-4-20250514"

        # Verify response contains the same model name
        data = response.json()
        assert data["model"] == "claude-opus-4-20250514"

    def test_openai_model_translation(
        self, test_client, mock_claude_client, sample_claude_response
    ):
        """Test that OpenAI model names are translated to Claude models."""
        # Test cases for model translation
        test_cases = [
            ("o3-mini", "claude-opus-4-20250514"),
            ("o1-mini", "claude-sonnet-4-20250514"),
            ("gpt-4o-mini", "claude-3-5-haiku-latest"),
            ("gpt-4o", "claude-3-7-sonnet-20250219"),
            ("claude-3-5-sonnet-latest", "claude-3-5-sonnet-latest"),  # Pass through
        ]

        for openai_model, expected_claude_model in test_cases:
            # Adjust response to match the model
            adjusted_response = {
                **sample_claude_response,
                "model": expected_claude_model,
            }
            mock_claude_client.create_completion = AsyncMock(
                return_value=adjusted_response
            )

            request_data = {
                "model": openai_model,
                "messages": [{"role": "user", "content": "Hello"}],
            }

            response = test_client.post(
                "/cc/openai/v1/chat/completions", json=request_data
            )
            assert response.status_code == 200

            # Verify the Claude client was called with the translated model
            args, kwargs = mock_claude_client.create_completion.call_args
            assert args[1].model == expected_claude_model

            # Verify response contains the original OpenAI model name
            data = response.json()
            assert data["model"] == openai_model

    def test_streaming_format_openai_endpoint(self, test_client, mock_claude_client):
        """Test that /cc/openai/v1 endpoint returns OpenAI SSE format."""

        # Mock streaming response with Anthropic format
        async def mock_stream():
            yield {"type": "message_start", "message": {"role": "assistant"}}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
            }

        mock_claude_client.create_completion = AsyncMock(return_value=mock_stream())

        request_data = {
            "model": "o3-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        # Send request to OpenAI endpoint
        with test_client.stream(
            "POST", "/cc/openai/v1/chat/completions", json=request_data
        ) as response:
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )

            # Collect all chunks
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)

            # Verify OpenAI SSE format
            assert len(chunks) > 0

            # First chunk should have role
            first_chunk_data = json.loads(chunks[0][6:])  # Remove "data: " prefix
            assert first_chunk_data["object"] == "chat.completion.chunk"
            assert "choices" in first_chunk_data
            assert first_chunk_data["choices"][0]["delta"].get("role") == "assistant"

            # Content chunks should have OpenAI format
            for chunk_line in chunks[1:-1]:  # Skip first and last
                if chunk_line == "data: [DONE]":
                    continue
                chunk_data = json.loads(chunk_line[6:])
                assert chunk_data["object"] == "chat.completion.chunk"
                assert "choices" in chunk_data
                assert "delta" in chunk_data["choices"][0]

            # Last chunk should be [DONE]
            assert chunks[-1] == "data: [DONE]"

    def test_streaming_with_usage_option(self, test_client, mock_claude_client):
        """Test streaming with stream_options.include_usage."""

        # Mock streaming response with usage
        async def mock_stream():
            yield {"type": "message_start", "message": {"role": "assistant"}}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello there!"},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 25, "output_tokens": 50},
            }

        mock_claude_client.create_completion = AsyncMock(return_value=mock_stream())

        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Send request to OpenAI endpoint
        with test_client.stream(
            "POST", "/cc/openai/v1/chat/completions", json=request_data
        ) as response:
            assert response.status_code == 200

            # Collect all chunks
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)

            # Find the final chunk before [DONE]
            final_chunk_data = None
            for i in range(len(chunks) - 1, -1, -1):
                if chunks[i] != "data: [DONE]":
                    final_chunk_data = json.loads(chunks[i][6:])
                    break

            # Verify usage is included in final chunk
            assert final_chunk_data is not None
            assert "usage" in final_chunk_data
            assert final_chunk_data["usage"]["prompt_tokens"] == 25
            assert final_chunk_data["usage"]["completion_tokens"] == 50
            assert final_chunk_data["usage"]["total_tokens"] == 75

    def test_streaming_without_usage_option(self, test_client, mock_claude_client):
        """Test streaming without stream_options.include_usage."""

        # Mock streaming response with usage
        async def mock_stream():
            yield {"type": "message_start", "message": {"role": "assistant"}}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello!"},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 25, "output_tokens": 50},
            }

        mock_claude_client.create_completion = AsyncMock(return_value=mock_stream())

        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            # No stream_options or include_usage=false
        }

        # Send request to OpenAI endpoint
        with test_client.stream(
            "POST", "/cc/openai/v1/chat/completions", json=request_data
        ) as response:
            assert response.status_code == 200

            # Collect all chunks
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)

            # Find the final chunk before [DONE]
            final_chunk_data = None
            for i in range(len(chunks) - 1, -1, -1):
                if chunks[i] != "data: [DONE]":
                    final_chunk_data = json.loads(chunks[i][6:])
                    break

            # Verify usage is NOT included (default behavior)
            assert final_chunk_data is not None
            assert "usage" not in final_chunk_data


@pytest.mark.unit
class TestOpenAIRequestValidation:
    """Test OpenAI request validation."""

    def test_openai_message_validation(self):
        """Test OpenAI message validation."""
        # Valid message
        valid_message = OpenAIMessage(
            role="user",
            content="Hello, world!",
            name=None,
            tool_calls=None,
            tool_call_id=None,
        )
        assert valid_message.role == "user"
        assert valid_message.content == "Hello, world!"

        # Message with tool calls
        tool_message = OpenAIMessage(
            role="assistant",
            content="I'll help you with that.",
            name=None,
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
            tool_call_id=None,
        )
        assert tool_message.tool_calls is not None
        assert len(tool_message.tool_calls) == 1

    def test_openai_request_validation(self):
        """Test OpenAI request validation."""
        # Valid request
        valid_request = OpenAIChatCompletionRequest(
            model="claude-opus-4-20250514",
            messages=[
                OpenAIMessage(
                    role="user",
                    content="Hello",
                    name=None,
                    tool_calls=None,
                    tool_call_id=None,
                )
            ],
            max_tokens=None,
            temperature=None,
            top_p=None,
            n=1,
            stream=False,
            stream_options=None,
            stop=None,
            presence_penalty=None,
            frequency_penalty=None,
            logit_bias=None,
            user=None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=True,
            response_format=None,
            seed=None,
            logprobs=None,
            top_logprobs=None,
        )
        assert valid_request.model == "claude-opus-4-20250514"
        assert len(valid_request.messages) == 1

        # Request with parameters
        param_request = OpenAIChatCompletionRequest(
            model="claude-3-7-sonnet-20250219",
            messages=[
                OpenAIMessage(
                    role="user",
                    content="Hello",
                    name=None,
                    tool_calls=None,
                    tool_call_id=None,
                )
            ],
            max_tokens=150,
            temperature=0.8,
            top_p=None,
            n=1,
            stream=True,
            stream_options=None,
            stop=None,
            presence_penalty=None,
            frequency_penalty=None,
            logit_bias=None,
            user=None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=True,
            response_format=None,
            seed=None,
            logprobs=None,
            top_logprobs=None,
        )
        assert param_request.max_tokens == 150
        assert param_request.temperature == 0.8
        assert param_request.stream is True

    def test_model_validation(self):
        """Test model validation - should pass through as-is."""
        # Test Claude model passes through unchanged
        claude_request = OpenAIChatCompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[
                OpenAIMessage(
                    role="user",
                    content="Hello",
                    name=None,
                    tool_calls=None,
                    tool_call_id=None,
                )
            ],
            max_tokens=None,
            temperature=None,
            top_p=None,
            n=1,
            stream=False,
            stream_options=None,
            stop=None,
            presence_penalty=None,
            frequency_penalty=None,
            logit_bias=None,
            user=None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=True,
            response_format=None,
            seed=None,
            logprobs=None,
            top_logprobs=None,
        )
        assert claude_request.model == "claude-sonnet-4-20250514"

        # Test any model name passes through as-is
        any_request = OpenAIChatCompletionRequest(
            model="any-model-name",
            messages=[
                OpenAIMessage(
                    role="user",
                    content="Hello",
                    name=None,
                    tool_calls=None,
                    tool_call_id=None,
                )
            ],
            max_tokens=None,
            temperature=None,
            top_p=None,
            n=1,
            stream=False,
            stream_options=None,
            stop=None,
            presence_penalty=None,
            frequency_penalty=None,
            logit_bias=None,
            user=None,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=True,
            response_format=None,
            seed=None,
            logprobs=None,
            top_logprobs=None,
        )
        assert any_request.model == "any-model-name"


@pytest.mark.integration
class TestOpenAIToolsValidation:
    """Test OpenAI tools validation functionality."""

    @pytest.fixture
    def mock_claude_client(self):
        """Mock Claude client directly."""
        with patch(
            "ccproxy.routers.claudecode.openai.ClaudeClient"
        ) as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def sample_request_with_tools(self):
        """Sample OpenAI request with tools."""
        return {
            "model": "claude-opus-4-20250514",
            "messages": [{"role": "user", "content": "What's the weather like?"}],
            "max_tokens": 100,
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

    @patch("ccproxy.routers.claudecode.openai.get_settings")
    def test_tools_validation_error_mode(
        self, mock_get_settings, test_client, sample_request_with_tools
    ):
        """Test that tools validation returns error in error mode."""
        # Mock settings to return error mode
        mock_settings = MagicMock()
        mock_settings.api_tools_handling = "error"
        mock_get_settings.return_value = mock_settings

        response = test_client.post(
            "/cc/openai/v1/chat/completions", json=sample_request_with_tools
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert (
            "Tools definitions are not supported" in data["detail"]["error"]["message"]
        )
        assert data["detail"]["error"]["type"] == "unsupported_parameter"

    @patch("ccproxy.routers.claudecode.openai.get_settings")
    def test_tools_validation_warning_mode(
        self,
        mock_get_settings,
        test_client,
        mock_claude_client,
        sample_request_with_tools,
    ):
        """Test that tools validation logs warning in warning mode."""
        # Mock settings to return warning mode
        mock_settings = MagicMock()
        mock_settings.api_tools_handling = "warning"
        mock_get_settings.return_value = mock_settings

        # Mock Claude client response
        mock_claude_client.create_completion = AsyncMock(
            return_value={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "I can help you with that!"}],
                "model": "claude-opus-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 8},
            }
        )

        with patch("ccproxy.routers.claudecode.openai.logger") as mock_logger:
            response = test_client.post(
                "/cc/openai/v1/chat/completions", json=sample_request_with_tools
            )

            assert response.status_code == 200
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Tools ignored" in warning_call
            assert "1 tools" in warning_call

    @patch("ccproxy.routers.claudecode.openai.get_settings")
    def test_tools_validation_ignore_mode(
        self,
        mock_get_settings,
        test_client,
        mock_claude_client,
        sample_request_with_tools,
    ):
        """Test that tools validation ignores tools in ignore mode."""
        # Mock settings to return ignore mode
        mock_settings = MagicMock()
        mock_settings.tools_handling = "ignore"
        mock_get_settings.return_value = mock_settings

        # Mock Claude client response
        mock_claude_client.create_completion = AsyncMock(
            return_value={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "I can help you with that!"}],
                "model": "claude-opus-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 8},
            }
        )

        with patch("ccproxy.routers.claudecode.openai.logger") as mock_logger:
            response = test_client.post(
                "/cc/openai/v1/chat/completions", json=sample_request_with_tools
            )

            assert response.status_code == 200
            # Should not log warning or error
            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()

    @patch("ccproxy.routers.claudecode.openai.get_settings")
    def test_request_without_tools_continues_normally(
        self, mock_get_settings, test_client, mock_claude_client
    ):
        """Test that requests without tools continue normally regardless of settings."""
        # Mock settings (tools_handling doesn't matter for this test)
        mock_settings = MagicMock()
        mock_settings.api_tools_handling = "error"
        mock_get_settings.return_value = mock_settings

        sample_request = {
            "model": "claude-opus-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        # Mock Claude client response
        mock_claude_client.create_completion = AsyncMock(
            return_value={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help you?"}],
                "model": "claude-opus-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 8},
            }
        )

        response = test_client.post(
            "/cc/openai/v1/chat/completions", json=sample_request
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"


@pytest.mark.unit
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
