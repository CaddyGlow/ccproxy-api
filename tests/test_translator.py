"""Tests for OpenAI to Anthropic translator."""

import json
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.services.translator import (
    OPENAI_TO_CLAUDE_MODEL_MAPPING,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIRequest,
    OpenAIResponse,
    OpenAIStreamChoice,
    OpenAIStreamResponse,
    OpenAITranslator,
    OpenAIUsage,
    map_openai_model_to_claude,
)


@pytest.mark.unit
class TestModelMapping:
    """Test OpenAI to Claude model mapping functionality."""

    def test_map_openai_model_to_claude_exact_matches(self) -> None:
        """Test exact model name matches."""
        for openai_model, expected_claude in OPENAI_TO_CLAUDE_MODEL_MAPPING.items():
            result = map_openai_model_to_claude(openai_model)
            assert result == expected_claude

    def test_map_openai_model_to_claude_startswith_matches(self) -> None:
        """Test startswith model name matches."""
        test_cases = [
            ("gpt-4o-mini-2024-07-18", "claude-3-5-haiku-latest"),
            ("o3-mini-preview", "claude-opus-4-20250514"),
            ("o1-mini-2024", "claude-sonnet-4-20250514"),
            ("gpt-4o-2024-05-13", "claude-3-7-sonnet-20250219"),
        ]

        for input_model, expected_output in test_cases:
            result = map_openai_model_to_claude(input_model)
            assert result == expected_output

    def test_map_claude_models_pass_through(self) -> None:
        """Test that Claude models pass through without mapping."""
        claude_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
        ]

        for model in claude_models:
            result = map_openai_model_to_claude(model)
            assert result == model

    def test_map_unknown_models_pass_through(self) -> None:
        """Test that unknown models pass through unchanged."""
        unknown_models = [
            "unknown-model",
            "custom-model-v1",
            "my-fine-tuned-model",
            "text-davinci-003",  # OpenAI model not in mapping
        ]

        for model in unknown_models:
            result = map_openai_model_to_claude(model)
            assert result == model

    def test_exact_match_precedence_over_startswith(self) -> None:
        """Test that exact matches take precedence over startswith matches."""
        # gpt-4o should match exactly, not as startswith for gpt-4o-mini
        result = map_openai_model_to_claude("gpt-4o")
        assert result == "claude-3-7-sonnet-20250219"

    def test_empty_string_model(self) -> None:
        """Test behavior with empty string model."""
        result = map_openai_model_to_claude("")
        assert result == ""


@pytest.mark.unit
class TestPydanticModels:
    """Test Pydantic model validation and creation."""

    def test_openai_message_creation(self) -> None:
        """Test OpenAI message model creation."""
        # Basic user message
        msg = OpenAIMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

        # Assistant message with tool calls
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"},
            }
        ]
        msg = OpenAIMessage(
            role="assistant", content="Using tool", tool_calls=tool_calls
        )
        assert msg.role == "assistant"
        assert msg.tool_calls == tool_calls

        # Tool message
        msg = OpenAIMessage(role="tool", content="Tool result", tool_call_id="call_123")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"

    def test_openai_request_creation(self) -> None:
        """Test OpenAI request model creation."""
        messages = [OpenAIMessage(role="user", content="Hello")]
        req = OpenAIRequest(model="gpt-4o", messages=messages)

        assert req.model == "gpt-4o"
        assert len(req.messages) == 1
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.stream is False  # default value

        # With optional parameters
        req = OpenAIRequest(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            stop=["STOP"],
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )
        assert req.max_tokens == 100
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.stream is True
        assert req.stop == ["STOP"]

    def test_openai_choice_creation(self) -> None:
        """Test OpenAI choice model creation."""
        message = OpenAIMessage(role="assistant", content="Hello")
        choice = OpenAIChoice(index=0, message=message, finish_reason="stop")

        assert choice.index == 0
        assert choice.message == message
        assert choice.finish_reason == "stop"

    def test_openai_usage_creation(self) -> None:
        """Test OpenAI usage model creation."""
        usage = OpenAIUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_openai_response_creation(self) -> None:
        """Test OpenAI response model creation."""
        message = OpenAIMessage(role="assistant", content="Hello")
        choice = OpenAIChoice(index=0, message=message, finish_reason="stop")
        usage = OpenAIUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        response = OpenAIResponse(
            id="chatcmpl-123",
            created=1234567890,
            model="gpt-4o",
            choices=[choice],
            usage=usage,
        )

        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert response.created == 1234567890
        assert response.model == "gpt-4o"
        assert len(response.choices) == 1
        assert response.usage == usage

    def test_openai_stream_choice_creation(self) -> None:
        """Test OpenAI stream choice model creation."""
        choice = OpenAIStreamChoice(
            index=0, delta={"content": "Hello"}, finish_reason=None
        )

        assert choice.index == 0
        assert choice.delta == {"content": "Hello"}
        assert choice.finish_reason is None

    def test_openai_stream_response_creation(self) -> None:
        """Test OpenAI stream response model creation."""
        choice = OpenAIStreamChoice(index=0, delta={"content": "Hello"})

        response = OpenAIStreamResponse(
            id="chatcmpl-123", created=1234567890, model="gpt-4o", choices=[choice]
        )

        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion.chunk"
        assert response.created == 1234567890
        assert response.model == "gpt-4o"
        assert len(response.choices) == 1
        assert response.usage is None


@pytest.mark.unit
class TestOpenAITranslator:
    """Test OpenAI to Anthropic translator functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.translator = OpenAITranslator()

    def test_translator_initialization(self) -> None:
        """Test translator initialization."""
        translator = OpenAITranslator()
        assert translator is not None

    def test_openai_to_anthropic_request_basic(self) -> None:
        """Test basic OpenAI to Anthropic request conversion."""
        openai_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 100,
        }

        result = self.translator.openai_to_anthropic_request(openai_request)

        assert result["model"] == "claude-3-7-sonnet-20250219"  # mapped model
        assert result["max_tokens"] == 100
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, how are you?"

    def test_openai_to_anthropic_request_with_system_prompt(self) -> None:
        """Test request conversion with system prompt."""
        openai_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        }

        result = self.translator.openai_to_anthropic_request(openai_request)

        assert "system" in result
        assert result["system"] == "You are a helpful assistant."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_openai_to_anthropic_request_with_optional_params(self) -> None:
        """Test request conversion with optional parameters."""
        openai_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "stop": ["STOP", "END"],
        }

        result = self.translator.openai_to_anthropic_request(openai_request)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stream"] is True
        assert result["stop_sequences"] == ["STOP", "END"]

    def test_openai_to_anthropic_request_stop_string(self) -> None:
        """Test request conversion with single stop string."""
        openai_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "stop": "STOP",
        }

        result = self.translator.openai_to_anthropic_request(openai_request)

        assert result["stop_sequences"] == ["STOP"]

    def test_openai_to_anthropic_request_with_tools(self) -> None:
        """Test request conversion with tools."""
        openai_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Get weather"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        result = self.translator.openai_to_anthropic_request(openai_request)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["description"] == "Get current weather"
        assert result["tools"][0]["input_schema"] == {
            "type": "object",
            "properties": {},
        }

    def test_openai_to_anthropic_request_with_deprecated_functions(self) -> None:
        """Test request conversion with deprecated functions parameter."""
        openai_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Get weather"}],
            "functions": [
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }

        result = self.translator.openai_to_anthropic_request(openai_request)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"

    def test_openai_to_anthropic_request_with_tool_choice(self) -> None:
        """Test request conversion with tool_choice."""
        test_cases = [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ("required", {"type": "any"}),
            (
                {"type": "function", "function": {"name": "test_func"}},
                {"type": "tool", "name": "test_func"},
            ),
        ]

        for tool_choice, expected in test_cases:
            openai_request = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "tool_choice": tool_choice,
            }

            result = self.translator.openai_to_anthropic_request(openai_request)
            assert result["tool_choice"] == expected

    def test_openai_to_anthropic_request_with_deprecated_function_call(self) -> None:
        """Test request conversion with deprecated function_call parameter."""
        test_cases = [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ({"name": "test_func"}, {"type": "tool", "name": "test_func"}),
        ]

        for function_call, expected in test_cases:
            openai_request = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "function_call": function_call,
            }

            result = self.translator.openai_to_anthropic_request(openai_request)
            assert result["tool_choice"] == expected

    def test_anthropic_to_openai_response_basic(self) -> None:
        """Test basic Anthropic to OpenAI response conversion."""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        with patch("time.time", return_value=1234567890):
            result = self.translator.anthropic_to_openai_response(
                anthropic_response, "gpt-4o", "test-request-id"
            )

        assert result["id"] == "test-request-id"
        assert result["object"] == "chat.completion"
        assert result["created"] == 1234567890
        assert result["model"] == "gpt-4o"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert (
            result["choices"][0]["message"]["content"] == "Hello! How can I help you?"
        )
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30

    def test_anthropic_to_openai_response_with_tool_use(self) -> None:
        """Test response conversion with tool use."""
        anthropic_response = {
            "content": [
                {"type": "text", "text": "I'll help you with that."},
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"location": "New York"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 15, "output_tokens": 10},
        }

        result = self.translator.anthropic_to_openai_response(
            anthropic_response, "gpt-4o"
        )

        assert result["choices"][0]["message"]["content"] == "I'll help you with that."
        assert "tool_calls" in result["choices"][0]["message"]
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        # The actual implementation uses json.dumps which formats with double quotes
        assert tool_calls[0]["function"]["arguments"] == '{"location": "New York"}'
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_anthropic_to_openai_response_auto_generated_id(self) -> None:
        """Test response conversion with auto-generated request ID."""
        anthropic_response = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "abcdef123456789012345678901234567890"
            result = self.translator.anthropic_to_openai_response(
                anthropic_response, "gpt-4o"
            )

        assert result["id"] == "chatcmpl-abcdef12345678901234567890123"

    def test_anthropic_to_openai_response_empty_content(self) -> None:
        """Test response conversion with empty content."""
        anthropic_response = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }

        result = self.translator.anthropic_to_openai_response(
            anthropic_response, "gpt-4o"
        )

        assert result["choices"][0]["message"]["content"] is None

    @pytest.mark.asyncio
    async def test_anthropic_to_openai_stream_basic(self) -> None:
        """Test basic Anthropic to OpenAI streaming conversion."""

        async def mock_stream():
            chunks = [
                {
                    "type": "message_start",
                    "message": {"id": "msg_123", "role": "assistant", "content": []},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": " world"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                },
            ]
            for chunk in chunks:
                yield chunk

        with patch("time.time", return_value=1234567890):
            stream = self.translator.anthropic_to_openai_stream(
                mock_stream(), "gpt-4o", "test-req-id"
            )

            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

        # Check initial chunk
        assert chunks[0]["id"] == "test-req-id"
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

        # Check content chunks
        assert chunks[1]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[2]["choices"][0]["delta"]["content"] == " world"

        # Check final chunk
        final_chunk = chunks[-1]
        assert final_chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" in final_chunk
        assert final_chunk["usage"]["prompt_tokens"] == 5
        assert final_chunk["usage"]["completion_tokens"] == 10

    @pytest.mark.asyncio
    async def test_anthropic_to_openai_stream_with_tool_use(self) -> None:
        """Test streaming conversion with tool use."""

        async def mock_stream():
            chunks = [
                {
                    "type": "message_start",
                    "message": {"id": "msg_123", "role": "assistant"},
                },
                {
                    "type": "content_block_start",
                    "content_block": {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "get_weather",
                    },
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": '{"loc'},
                },
                {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": 'ation": "NY"}',
                    },
                },
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            ]
            for chunk in chunks:
                yield chunk

        stream = self.translator.anthropic_to_openai_stream(mock_stream(), "gpt-4o")

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        # Find tool call chunks
        tool_chunks = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
        assert len(tool_chunks) >= 1

        # Check tool call structure
        first_tool_chunk = tool_chunks[0]
        tool_call = first_tool_chunk["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["id"] == "call_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_anthropic_to_openai_stream_auto_generated_id(self) -> None:
        """Test streaming conversion with auto-generated request ID."""

        async def mock_stream():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.return_value.hex = "abcdef123456789012345678901234567890"
            stream = self.translator.anthropic_to_openai_stream(mock_stream(), "gpt-4o")

            chunk = await stream.__anext__()
            assert chunk["id"] == "chatcmpl-abcdef12345678901234567890123"

    def test_convert_messages_to_anthropic_basic(self) -> None:
        """Test basic message conversion."""
        messages = [
            OpenAIMessage(role="user", content="Hello"),
            OpenAIMessage(role="assistant", content="Hi there!"),
        ]

        result_messages, system_prompt = self.translator._convert_messages_to_anthropic(
            messages
        )

        assert system_prompt is None
        assert len(result_messages) == 2
        assert result_messages[0]["role"] == "user"
        assert result_messages[0]["content"] == "Hello"
        assert result_messages[1]["role"] == "assistant"
        assert result_messages[1]["content"] == "Hi there!"

    def test_convert_messages_to_anthropic_with_system(self) -> None:
        """Test message conversion with system message."""
        messages = [
            OpenAIMessage(role="system", content="You are helpful."),
            OpenAIMessage(role="user", content="Hello"),
        ]

        result_messages, system_prompt = self.translator._convert_messages_to_anthropic(
            messages
        )

        assert system_prompt == "You are helpful."
        assert len(result_messages) == 1
        assert result_messages[0]["role"] == "user"

    def test_convert_messages_to_anthropic_with_system_content_blocks(self) -> None:
        """Test system message conversion with content blocks."""
        messages = [
            OpenAIMessage(
                role="system",
                content=[
                    {"type": "text", "text": "You are"},
                    {"type": "text", "text": "helpful."},
                ],
            ),
            OpenAIMessage(role="user", content="Hello"),
        ]

        result_messages, system_prompt = self.translator._convert_messages_to_anthropic(
            messages
        )

        assert system_prompt == "You are helpful."
        assert len(result_messages) == 1

    def test_convert_messages_to_anthropic_with_tool_calls(self) -> None:
        """Test message conversion with tool calls."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NY"}'},
            }
        ]
        messages = [
            OpenAIMessage(
                role="assistant", content="Let me check", tool_calls=tool_calls
            )
        ]

        result_messages, _ = self.translator._convert_messages_to_anthropic(messages)

        assert len(result_messages) == 1
        content = result_messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # text + tool_use
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Let me check"
        assert content[1]["type"] == "tool_use"
        assert content[1]["id"] == "call_123"
        assert content[1]["name"] == "get_weather"

    def test_convert_messages_to_anthropic_with_tool_results(self) -> None:
        """Test message conversion with tool results."""
        messages = [
            OpenAIMessage(
                role="tool", content="Weather: Sunny", tool_call_id="call_123"
            )
        ]

        result_messages, _ = self.translator._convert_messages_to_anthropic(messages)

        assert len(result_messages) == 1
        assert result_messages[0]["role"] == "user"
        content = result_messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "call_123"
        assert content[0]["content"] == "Weather: Sunny"

    def test_convert_content_to_anthropic_string(self) -> None:
        """Test content conversion with string input."""
        result = self.translator._convert_content_to_anthropic("Hello world")
        assert result == "Hello world"

    def test_convert_content_to_anthropic_none(self) -> None:
        """Test content conversion with None input."""
        result = self.translator._convert_content_to_anthropic(None)
        assert result == ""

    def test_convert_content_to_anthropic_with_text_blocks(self) -> None:
        """Test content conversion with text blocks."""
        content = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]

        result = self.translator._convert_content_to_anthropic(content)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Hello"
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "World"

    def test_convert_content_to_anthropic_with_base64_image(self) -> None:
        """Test content conversion with base64 image."""
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/"
                },
            }
        ]

        result = self.translator._convert_content_to_anthropic(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["media_type"] == "image/jpeg"
        assert result[0]["source"]["data"] == "/9j/4AAQSkZJRgABAQEAYABgAAD/"

    def test_convert_content_to_anthropic_with_invalid_base64_image(self) -> None:
        """Test content conversion with invalid base64 image URL."""
        content = [{"type": "image_url", "image_url": {"url": "data:invalid-format"}}]

        with patch("ccproxy.services.translator.logger") as mock_logger:
            result = self.translator._convert_content_to_anthropic(content)
            mock_logger.warning.assert_called_once()

        # Should return empty content since invalid image was skipped
        assert result == ""

    def test_convert_content_to_anthropic_with_url_image(self) -> None:
        """Test content conversion with URL-based image."""
        content = [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]

        result = self.translator._convert_content_to_anthropic(content)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "[Image: https://example.com/image.jpg]"

    def test_convert_tools_to_anthropic(self) -> None:
        """Test tools conversion."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = self.translator._convert_tools_to_anthropic(tools)

        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get current weather"
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_convert_functions_to_anthropic(self) -> None:
        """Test deprecated functions conversion."""
        functions = [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

        result = self.translator._convert_functions_to_anthropic(functions)

        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get current weather"
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_convert_tool_choice_to_anthropic_string_values(self) -> None:
        """Test tool choice conversion with string values."""
        test_cases = [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ("required", {"type": "any"}),
            ("unknown", {"type": "auto"}),  # fallback
        ]

        for input_choice, expected in test_cases:
            result = self.translator._convert_tool_choice_to_anthropic(input_choice)
            assert result == expected

    def test_convert_tool_choice_to_anthropic_dict_values(self) -> None:
        """Test tool choice conversion with dict values."""
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        result = self.translator._convert_tool_choice_to_anthropic(tool_choice)

        assert result == {"type": "tool", "name": "get_weather"}

    def test_convert_function_call_to_anthropic_string_values(self) -> None:
        """Test deprecated function_call conversion with string values."""
        test_cases = [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ("unknown", {"type": "auto"}),  # fallback
        ]

        for input_call, expected in test_cases:
            result = self.translator._convert_function_call_to_anthropic(input_call)
            assert result == expected

    def test_convert_function_call_to_anthropic_dict_values(self) -> None:
        """Test deprecated function_call conversion with dict values."""
        function_call = {"name": "get_weather"}

        result = self.translator._convert_function_call_to_anthropic(function_call)

        assert result == {"type": "tool", "name": "get_weather"}

    def test_convert_tool_call_to_anthropic(self) -> None:
        """Test tool call conversion."""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"location": "NY"}'},
        }

        result = self.translator._convert_tool_call_to_anthropic(tool_call)

        assert result == {
            "type": "tool_use",
            "id": "call_123",
            "name": "get_weather",
            "input": {"location": "NY"},
        }

    def test_convert_tool_use_to_openai(self) -> None:
        """Test tool use conversion."""
        tool_use = {
            "id": "call_123",
            "name": "get_weather",
            "input": {"location": "NY"},
        }

        result = self.translator._convert_tool_use_to_openai(tool_use)

        assert (
            result
            == {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "NY"}',  # json.dumps converts dict to JSON format
                },
            }
        )

    def test_convert_stop_reason_to_openai(self) -> None:
        """Test stop reason conversion."""
        test_cases = [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
            ("tool_use", "tool_calls"),
            ("unknown_reason", "stop"),  # fallback
            (None, None),
        ]

        for input_reason, expected in test_cases:
            result = self.translator._convert_stop_reason_to_openai(input_reason)
            assert result == expected

    def test_convert_messages_with_empty_tool_calls_content(self) -> None:
        """Test message conversion when assistant has tool calls but empty content."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ]
        messages = [OpenAIMessage(role="assistant", content="", tool_calls=tool_calls)]

        result_messages, _ = self.translator._convert_messages_to_anthropic(messages)

        assert len(result_messages) == 1
        content = result_messages[0]["content"]
        assert isinstance(content, list)
        # The implementation always creates a text block, even for empty content
        assert len(content) == 2  # text block (even if empty) + tool_use
        assert content[0]["type"] == "text"
        assert content[0]["text"] == ""
        assert content[1]["type"] == "tool_use"

    def test_convert_messages_with_tool_results_append_to_previous_user(self) -> None:
        """Test tool result appending to previous user message."""
        messages = [
            OpenAIMessage(role="user", content="Get weather"),
            OpenAIMessage(role="tool", content="Sunny", tool_call_id="call_123"),
        ]

        result_messages, _ = self.translator._convert_messages_to_anthropic(messages)

        assert len(result_messages) == 1
        assert result_messages[0]["role"] == "user"
        content = result_messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # original text + tool_result
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Get weather"
        assert content[1]["type"] == "tool_result"
        assert content[1]["tool_use_id"] == "call_123"
        assert content[1]["content"] == "Sunny"

    def test_convert_empty_content_list(self) -> None:
        """Test content conversion with empty list."""
        result = self.translator._convert_content_to_anthropic([])
        assert result == ""

    def test_tools_with_missing_function_data(self) -> None:
        """Test tools conversion with missing function data."""
        tools = [
            {
                "type": "function",
                "function": {},  # Missing required fields
            }
        ]

        result = self.translator._convert_tools_to_anthropic(tools)

        assert len(result) == 1
        assert result[0]["name"] == ""
        assert result[0]["description"] == ""
        assert result[0]["input_schema"] == {}

    def test_tool_call_with_missing_data(self) -> None:
        """Test tool call conversion with missing data."""
        tool_call = {
            "id": "",
            "function": {},  # Missing name and arguments
        }

        result = self.translator._convert_tool_call_to_anthropic(tool_call)

        assert result["type"] == "tool_use"
        assert result["id"] == ""
        assert result["name"] == ""
        assert result["input"] == {}

    def test_tool_use_with_missing_data(self) -> None:
        """Test tool use conversion with missing data."""
        tool_use: dict[str, Any] = {}  # Missing all fields

        result = self.translator._convert_tool_use_to_openai(tool_use)

        assert result["id"] == ""
        assert result["type"] == "function"
        assert result["function"]["name"] == ""
        assert result["function"]["arguments"] == "{}"

    def test_convert_messages_with_tool_calls_edge_case_coverage(self) -> None:
        """Test edge case to cover line 496 - defensive code path."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ]

        messages = [
            OpenAIMessage(role="assistant", content="test", tool_calls=tool_calls)
        ]

        # Mock _convert_content_to_anthropic to return something that's not a list or string
        with patch.object(
            self.translator, "_convert_content_to_anthropic", return_value=None
        ):
            result_messages, _ = self.translator._convert_messages_to_anthropic(
                messages
            )

        assert len(result_messages) == 1
        content = result_messages[0]["content"]
        assert isinstance(content, list)
        # Should have tool_use only since content was None -> []
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"

    def test_convert_tool_choice_to_anthropic_unknown_dict_type(self) -> None:
        """Test tool choice conversion with unknown dict type."""
        tool_choice = {"type": "unknown", "some_field": "value"}

        result = self.translator._convert_tool_choice_to_anthropic(tool_choice)

        # Should fall back to auto for unknown dict types
        assert result == {"type": "auto"}
