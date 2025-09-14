import pytest

from ccproxy.llms.adapters.anthropic_messages_to_openai_chatcompletions import (
    AnthropicToOpenAIChatCompletionsAdapter,
)
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.openai import models as openai_models


@pytest.fixture
def adapter() -> AnthropicToOpenAIChatCompletionsAdapter:
    return AnthropicToOpenAIChatCompletionsAdapter()


class TestAnthropicToOpenAIChatCompletionsAdapter:
    @pytest.mark.asyncio
    async def test_adapt_request_simple_text(
        self, adapter: AnthropicToOpenAIChatCompletionsAdapter
    ) -> None:
        # Arrange
        anthropic_request = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 1024,
        }

        # Act
        openai_request = await adapter.adapt_request(anthropic_request)

        # Assert
        assert openai_request["model"] == "claude-3-opus-20240229"
        assert openai_request["max_completion_tokens"] == 1024
        assert len(openai_request["messages"]) == 1
        assert openai_request["messages"][0]["role"] == "user"
        assert openai_request["messages"][0]["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_adapt_request_with_system_prompt(
        self, adapter: AnthropicToOpenAIChatCompletionsAdapter
    ) -> None:
        # Arrange
        anthropic_request = {
            "model": "claude-3-opus-20240229",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 1024,
        }

        # Act
        openai_request = await adapter.adapt_request(anthropic_request)

        # Assert
        assert len(openai_request["messages"]) == 2
        assert openai_request["messages"][0]["role"] == "system"
        assert (
            openai_request["messages"][0]["content"] == "You are a helpful assistant."
        )
        assert openai_request["messages"][1]["role"] == "user"
        assert openai_request["messages"][1]["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_adapt_request_with_tools(
        self, adapter: AnthropicToOpenAIChatCompletionsAdapter
    ) -> None:
        # Arrange
        anthropic_request = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": "What's the weather in San Francisco?"}
            ],
            "max_tokens": 1024,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                }
            ],
        }

        # Act
        openai_request = await adapter.adapt_request(anthropic_request)

        # Assert
        assert len(openai_request["tools"]) == 1
        tool = openai_request["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert (
            tool["function"]["description"]
            == "Get the current weather in a given location"
        )
        assert tool["function"]["parameters"]["type"] == "object"

    @pytest.mark.asyncio
    async def test_adapt_request_with_tool_choice_and_parallel(
        self, adapter: AnthropicToOpenAIChatCompletionsAdapter
    ) -> None:
        anthropic_request = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200,
            "tool_choice": {
                "type": "tool",
                "name": "search",
                "disable_parallel_tool_use": True,
            },
        }
        openai_request = await adapter.adapt_request(anthropic_request)
        assert openai_request["tool_choice"]["type"] == "function"
        assert openai_request["tool_choice"]["function"]["name"] == "search"
        assert openai_request["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_adapt_request_with_image_block(
        self, adapter: AnthropicToOpenAIChatCompletionsAdapter
    ) -> None:
        anthropic_request = {
            "model": "claude-3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "caption"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AAA...",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 200,
        }
        openai_request = await adapter.adapt_request(anthropic_request)
        parts = openai_request["messages"][0]["content"]
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_adapt_request_with_tool_use_and_result(
        self, adapter: AnthropicToOpenAIChatCompletionsAdapter
    ) -> None:
        # Arrange
        anthropic_request = {
            "model": "claude-3",
            "messages": [
                {"role": "user", "content": "What is the weather in SF?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"location": "San Francisco"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "72 and sunny",
                        }
                    ],
                },
            ],
            "max_tokens": 200,
        }

        # Act
        openai_request = await adapter.adapt_request(anthropic_request)

        # Assert
        messages = openai_request["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["id"] == "toolu_1"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "toolu_1"
        assert messages[2]["content"] == "72 and sunny"

    @pytest.mark.asyncio
    async def test_adapt_stream(
        self, adapter: AnthropicToOpenAIChatCompletionsAdapter
    ) -> None:
        # Arrange
        async def anthropic_stream():
            yield anthropic_models.MessageStartEvent(
                type="message_start",
                message=anthropic_models.MessageResponse(
                    id="msg_1",
                    type="message",
                    role="assistant",
                    model="claude-3",
                    content=[],
                    stop_reason=None,
                    stop_sequence=None,
                    usage=anthropic_models.Usage(input_tokens=10, output_tokens=0),
                ),
            ).model_dump()
            yield anthropic_models.ContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=anthropic_models.TextBlock(type="text", text="Hello"),
            ).model_dump()
            yield anthropic_models.ContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=anthropic_models.TextBlock(type="text", text=" world"),
            ).model_dump()
            yield anthropic_models.MessageDeltaEvent(
                type="message_delta",
                delta=anthropic_models.MessageDelta(stop_reason="end_turn"),
                usage=anthropic_models.Usage(input_tokens=10, output_tokens=5),
            ).model_dump()
            yield anthropic_models.MessageStopEvent(type="message_stop").model_dump()

        # Act
        stream = adapter.adapt_stream(anthropic_stream())
        chunks = [chunk async for chunk in stream]

        # Assert
        assert len(chunks) == 3

        # First chunk
        chunk1 = openai_models.ChatCompletionChunk.model_validate(chunks[0])
        assert chunk1.choices[0].delta.content == "Hello"
        assert chunk1.model == "claude-3"

        # Second chunk
        chunk2 = openai_models.ChatCompletionChunk.model_validate(chunks[1])
        assert chunk2.choices[0].delta.content == " world"

        # Final chunk
        chunk3 = openai_models.ChatCompletionChunk.model_validate(chunks[2])
        assert chunk3.choices[0].delta.content is None
        assert chunk3.choices[0].finish_reason == "stop"
        assert chunk3.usage is not None
        assert chunk3.usage.prompt_tokens == 10
        assert chunk3.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_adapt_error(
        self, adapter: AnthropicToOpenAIChatCompletionsAdapter
    ) -> None:
        # Arrange
        anthropic_error = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid request",
            },
        }

        # Act
        openai_error = await adapter.adapt_error(anthropic_error)

        # Assert
        error_detail = openai_error.get("error")
        assert error_detail is not None
        assert error_detail["type"] == "invalid_request_error"
        assert error_detail["message"] == "Invalid request"
        assert error_detail["param"] is None
        assert error_detail["code"] is None
