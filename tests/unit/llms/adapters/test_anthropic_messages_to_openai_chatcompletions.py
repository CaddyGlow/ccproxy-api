import pytest

from ccproxy.llms.adapters.anthropic_messages_to_openai_chatcompletions import (
    AnthropicToOpenAIChatCompletionsAdapter,
)


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
