"""Configuration and static data for endpoint test execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ccproxy.llms.models.anthropic import MessageResponse, MessageStartEvent
from ccproxy.llms.models.openai import (
    BaseStreamEvent,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ResponseObject,
)
from ccproxy.llms.streaming.accumulators import (
    ClaudeAccumulator,
    OpenAIAccumulator,
    ResponsesAccumulator,
    StreamAccumulator,
)
from ccproxy.plugins.claude_api import factory as claude_api_factory
from ccproxy.plugins.claude_sdk.plugin import factory as claude_sdk_factory
from ccproxy.plugins.codex import factory as codex_factory
from ccproxy.plugins.copilot import factory as copilot_factory

from .models import EndpointTest
from .tools import ANTHROPIC_TOOLS, CODEX_TOOLS, OPENAI_TOOLS


# Centralized message payloads per provider
MESSAGE_PAYLOADS: dict[str, Any] = {
    "openai": [{"role": "user", "content": "Hello"}],
    "anthropic": [{"role": "user", "content": "Hello"}],
    "response_api": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }
    ],
    "response_api_structured": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is 2+2? Answer in one word."}
            ],
        }
    ],
    # Tool testing payloads
    "openai_tools": [
        {
            "role": "user",
            "content": "What's the weather like in New York, and how far is it from Los Angeles?",
        }
    ],
    "anthropic_tools": [
        {
            "role": "user",
            "content": "What's the weather like in New York, and how far is it from Los Angeles?",
        }
    ],
    "responses_tools": [
        {
            "role": "user",
            "content": "What's the weather like in New York, and how far is it from Los Angeles?",
        }
    ],
    # Thinking mode payloads
    "openai_thinking": [
        {
            "role": "user",
            "content": "I need to calculate the factorial of 5. Can you help me think through this step by step?",
        }
    ],
    "responses_thinking": [
        {
            "role": "user",
            "content": "I need to calculate the factorial of 5. Can you help me think through this step by step?",
        }
    ],
    # Using messages format with tools
    # Structured output payloads
    "openai_structured": [
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ],
}


OPENAI_STRUCTURED_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "structured_math_response",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Single-word response to the math question.",
                },
                "answer_type": {
                    "type": "string",
                    "description": "Classification of the answer, e.g. number or word.",
                },
            },
            "required": ["answer", "answer_type"],
            "additionalProperties": False,
        },
    },
}


RESPONSES_STRUCTURED_TEXT_FORMAT: dict[str, Any] = {
    "format": {
        "type": "json_schema",
        "name": OPENAI_STRUCTURED_RESPONSE_FORMAT["json_schema"].get(
            "name", "structured_response"
        ),
        "schema": OPENAI_STRUCTURED_RESPONSE_FORMAT["json_schema"].get("schema", {}),
    }
}


# Request payload templates with model_class for validation
REQUEST_DATA: dict[str, dict[str, Any]] = {
    "openai_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai"],
        "max_tokens": 100,
        "stream": True,
        "model_class": ChatCompletionResponse,
        "chunk_model_class": ChatCompletionChunk,
        "api_format": "openai",
    },
    "openai_non_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai"],
        "max_tokens": 100,
        "stream": False,
        "model_class": ChatCompletionResponse,
        "api_format": "openai",
    },
    "response_api_stream": {
        "model": "{model}",
        "stream": True,
        "max_completion_tokens": 1000,
        "input": MESSAGE_PAYLOADS["response_api"],
        "model_class": ResponseObject,
        "chunk_model_class": BaseStreamEvent,
        "api_format": "responses",
    },
    "response_api_non_stream": {
        "model": "{model}",
        "stream": False,
        "max_completion_tokens": 1000,
        "input": MESSAGE_PAYLOADS["response_api"],
        "model_class": ResponseObject,
        "api_format": "responses",
    },
    "responses_structured_stream": {
        "model": "{model}",
        "max_completion_tokens": 1000,
        "stream": True,
        "input": MESSAGE_PAYLOADS["response_api_structured"],
        "text": RESPONSES_STRUCTURED_TEXT_FORMAT,
        "model_class": ResponseObject,
        "chunk_model_class": BaseStreamEvent,
        "accumulator_class": ResponsesAccumulator,
        "api_format": "responses",
    },
    "responses_structured_non_stream": {
        "model": "{model}",
        "max_completion_tokens": 1000,
        "stream": False,
        "input": MESSAGE_PAYLOADS["response_api_structured"],
        "text": RESPONSES_STRUCTURED_TEXT_FORMAT,
        "model_class": ResponseObject,
        "accumulator_class": ResponsesAccumulator,
        "api_format": "responses",
    },
    "anthropic_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": True,
        "messages": MESSAGE_PAYLOADS["anthropic"],
        "model_class": MessageResponse,
        "chunk_model_class": MessageStartEvent,
        "api_format": "anthropic",
    },
    "anthropic_non_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": False,
        "messages": MESSAGE_PAYLOADS["anthropic"],
        "model_class": MessageResponse,
        "api_format": "anthropic",
    },
    # Tool-enhanced requests
    "responses_tools_stream": {
        "model": "{model}",
        "max_completion_tokens": 1000,
        "stream": True,
        "tools": CODEX_TOOLS,
        "input": MESSAGE_PAYLOADS["responses_tools"],
        "model_class": ResponseObject,
        "chunk_model_class": BaseStreamEvent,
        "accumulator_class": ResponsesAccumulator,
        "api_format": "responses",
    },
    "responses_tools_non_stream": {
        "model": "{model}",
        "max_completion_tokens": 1000,
        "stream": False,
        "tools": CODEX_TOOLS,
        "input": MESSAGE_PAYLOADS["responses_tools"],
        "model_class": ResponseObject,
        "accumulator_class": ResponsesAccumulator,
        "api_format": "responses",
    },
    "responses_thinking_stream": {
        "model": "{model}",
        "max_completion_tokens": 1000,
        "stream": True,
        "input": MESSAGE_PAYLOADS["responses_thinking"],
        "reasoning": {"effort": "high", "summary": "auto"},
        "model_class": ResponseObject,
        "chunk_model_class": BaseStreamEvent,
        "accumulator_class": ResponsesAccumulator,
        "api_format": "responses",
    },
    "responses_thinking_non_stream": {
        "model": "{model}",
        "max_completion_tokens": 1000,
        "stream": False,
        "input": MESSAGE_PAYLOADS["responses_thinking"],
        "reasoning": {"effort": "high", "summary": "auto"},
        "model_class": ResponseObject,
        "accumulator_class": ResponsesAccumulator,
        "api_format": "responses",
    },
    "openai_tools_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai_tools"],
        "max_tokens": 1000,
        "stream": True,
        "tools": OPENAI_TOOLS,
        "model_class": ChatCompletionResponse,
        "chunk_model_class": ChatCompletionChunk,
        "accumulator_class": OpenAIAccumulator,
        "api_format": "openai",
    },
    "openai_tools_non_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai_tools"],
        "max_tokens": 1000,
        "stream": False,
        "tools": OPENAI_TOOLS,
        "model_class": ChatCompletionResponse,
        "api_format": "openai",
    },
    "anthropic_tools_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": True,
        "messages": MESSAGE_PAYLOADS["anthropic_tools"],
        "tools": ANTHROPIC_TOOLS,
        "model_class": MessageResponse,
        "chunk_model_class": MessageStartEvent,
        "accumulator_class": ClaudeAccumulator,
        "api_format": "anthropic",
    },
    "anthropic_tools_non_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": False,
        "messages": MESSAGE_PAYLOADS["anthropic_tools"],
        "tools": ANTHROPIC_TOOLS,
        "model_class": MessageResponse,
        "api_format": "anthropic",
    },
    "messages_tools_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": True,
        "messages": MESSAGE_PAYLOADS["anthropic_tools"],
        "tools": CODEX_TOOLS,
        "model_class": MessageResponse,
        "chunk_model_class": MessageStartEvent,
        "accumulator_class": ClaudeAccumulator,
        "api_format": "responses",
    },
    "messages_tools_non_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": False,
        "messages": MESSAGE_PAYLOADS["anthropic_tools"],
        "tools": CODEX_TOOLS,
        "model_class": MessageResponse,
        "api_format": "responses",
    },
    # Thinking mode requests (OpenAI only)
    "openai_thinking_stream": {
        "model": "o3-mini",
        "messages": MESSAGE_PAYLOADS["openai_thinking"],
        "stream": True,
        "temperature": 1.0,
        "model_class": ChatCompletionResponse,
        "chunk_model_class": ChatCompletionChunk,
        "api_format": "openai",
    },
    "openai_thinking_non_stream": {
        "model": "o3-mini",
        "messages": MESSAGE_PAYLOADS["openai_thinking"],
        "stream": False,
        "temperature": 1.0,
        "model_class": ChatCompletionResponse,
        "api_format": "openai",
    },
    # Structured output requests
    "openai_structured_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai_structured"],
        "max_tokens": 100,
        "stream": True,
        "temperature": 0.7,
        "response_format": OPENAI_STRUCTURED_RESPONSE_FORMAT,
        "model_class": ChatCompletionResponse,
        "chunk_model_class": ChatCompletionChunk,
        "api_format": "openai",
    },
    "openai_structured_non_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai_structured"],
        "max_tokens": 100,
        "stream": False,
        "temperature": 0.7,
        "response_format": OPENAI_STRUCTURED_RESPONSE_FORMAT,
        "model_class": ChatCompletionResponse,
        "api_format": "openai",
    },
}


class APIFormatTools(Protocol):
    """Protocol for API format-specific tool result handling."""

    def build_continuation_request(
        self,
        initial_request: dict[str, Any],
        original_response: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a continuation request with tool results for this API format."""


class OpenAIFormatTools:
    """Handle tool result continuation for the OpenAI API format."""

    def build_continuation_request(
        self,
        initial_request: dict[str, Any],
        original_response: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        continuation_request = initial_request.copy()

        original_messages = initial_request.get("messages", [])

        tool_calls = []
        if "choices" in original_response:
            for choice in original_response.get("choices", []):
                message = choice.get("message", {})
                if message.get("tool_calls"):
                    tool_calls.extend(message["tool_calls"])

        if not tool_calls and original_response.get("tool_calls"):
            tool_calls.extend(original_response["tool_calls"])

        assistant_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }

        tool_messages = []
        for result in tool_results:
            tool_call = result["tool_call"]
            tool_result = result["result"]
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "content": json.dumps(tool_result),
                }
            )

        continuation_request["messages"] = (
            original_messages + [assistant_message] + tool_messages
        )

        if "tools" in continuation_request:
            del continuation_request["tools"]

        return continuation_request


class AnthropicFormatTools:
    """Handle tool result continuation for the Anthropic API format."""

    def build_continuation_request(
        self,
        initial_request: dict[str, Any],
        original_response: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        continuation_request = initial_request.copy()

        original_messages = initial_request.get("messages", [])

        assistant_content = []
        for result in tool_results:
            tool_call = result["tool_call"]
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "input": result["tool_input"],
                }
            )

        assistant_message = {
            "role": "assistant",
            "content": assistant_content,
        }

        user_content = []
        summary_parts: list[str] = []
        for result in tool_results:
            tool_call = result["tool_call"]
            tool_result = result["result"]
            user_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.get("id"),
                    "content": str(tool_result),
                }
            )
            summary_parts.append(
                f"Tool {tool_call.get('name', 'unknown')} returned: {json.dumps(tool_result)}"
            )

        if summary_parts:
            user_content.append(
                {
                    "type": "text",
                    "text": " ".join(summary_parts),
                }
            )

        continuation_messages = original_messages + [assistant_message]
        if user_content:
            continuation_messages.append({"role": "user", "content": user_content})

        continuation_request["messages"] = continuation_messages

        if "tools" in continuation_request:
            del continuation_request["tools"]

        return continuation_request


class ResponsesFormatTools:
    """Handle tool result continuation for the Responses API format."""

    def build_continuation_request(
        self,
        initial_request: dict[str, Any],
        original_response: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        continuation_request = initial_request.copy()

        original_input = initial_request.get("input", [])
        original_messages = initial_request.get("messages", [])

        if not original_input and original_messages:
            original_input = []
            for message in original_messages:
                content = message.get("content")
                if isinstance(content, str):
                    original_input.append(
                        {
                            "type": "message",
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "input_text"
                                    if message["role"] == "user"
                                    else "output_text",
                                    "text": content,
                                }
                            ],
                        }
                    )
                elif isinstance(content, list):
                    blocks = []
                    for block in content:
                        if block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})
                            blocks.append(
                                {
                                    "type": "output_text",
                                    "text": f"Called {tool_name} with {tool_input}",
                                }
                            )
                        elif block.get("type") == "tool_result":
                            tool_id = block.get("tool_use_id", "")
                            result_content = block.get("content", "")
                            blocks.append(
                                {
                                    "type": "input_text",
                                    "text": f"Tool {tool_id} returned: {result_content}",
                                }
                            )
                        else:
                            blocks.append(
                                {
                                    "type": "input_text"
                                    if message["role"] == "user"
                                    else "output_text",
                                    "text": str(block),
                                }
                            )

                    if blocks:
                        original_input.append(
                            {
                                "type": "message",
                                "role": message["role"],
                                "content": blocks,
                            }
                        )

        continuation_input = original_input.copy()

        assistant_text_parts = []
        for result in tool_results:
            tool_name = result["tool_name"]
            tool_input = result["tool_input"]
            assistant_text_parts.append(
                f"I called {tool_name} with arguments: {json.dumps(tool_input)}"
            )

        if assistant_text_parts:
            continuation_input.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": " ".join(assistant_text_parts),
                        }
                    ],
                }
            )

            user_text_parts = []
            for result in tool_results:
                tool_name = result["tool_name"]
                tool_result = result["result"]
                user_text_parts.append(
                    f"The {tool_name} function returned: {json.dumps(tool_result)}"
                )

            continuation_input.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": " ".join(user_text_parts)
                            + " Please provide a summary of this information.",
                        }
                    ],
                }
            )

        continuation_request["input"] = continuation_input

        if "messages" in continuation_request:
            del continuation_request["messages"]

        if "tools" in continuation_request:
            del continuation_request["tools"]

        return continuation_request


FORMAT_TOOLS: dict[str, APIFormatTools] = {
    "openai": OpenAIFormatTools(),
    "anthropic": AnthropicFormatTools(),
    "responses": ResponsesFormatTools(),
}


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a provider's endpoints and capabilities."""

    name: str
    base_path: str
    model: str
    supported_formats: list[str]
    description_prefix: str


@dataclass(frozen=True)
class FormatConfig:
    """Configuration mapping API format to request types and endpoint paths."""

    name: str
    endpoint_path: str
    request_type_base: str
    description: str


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "copilot": ProviderConfig(
        name="copilot",
        base_path="/copilot/v1",
        model="gpt-4o",
        supported_formats=[
            "chat_completions",
            "responses",
            "messages",
            "chat_completions_tools",
            "messages_tools",
            "chat_completions_thinking",
            "chat_completions_structured",
            "responses_structured",
        ],
        description_prefix="Copilot",
    ),
    "claude": ProviderConfig(
        name="claude",
        base_path="/claude/v1",
        model="claude-sonnet-4-20250514",
        supported_formats=[
            "chat_completions",
            "responses",
            "messages",
            "chat_completions_tools",
            "messages_tools",
            "chat_completions_structured",
            "responses_structured",
        ],
        description_prefix="Claude API",
    ),
    "claude_sdk": ProviderConfig(
        name="claude_sdk",
        base_path="/claude/sdk/v1",
        model="claude-sonnet-4-20250514",
        supported_formats=[
            "chat_completions",
            "responses",
            "messages",
            "chat_completions_structured",
        ],
        description_prefix="Claude SDK",
    ),
    "codex": ProviderConfig(
        name="codex",
        base_path="/codex/v1",
        model="gpt-5",
        supported_formats=[
            "chat_completions",
            "responses",
            "messages",
            "responses_tools",
            "responses_thinking",
            "responses_structured",
            "chat_completions_tools",
            "messages_tools",
            "chat_completions_thinking",
            "chat_completions_structured",
        ],
        description_prefix="Codex",
    ),
}


PROVIDER_TOOL_ACCUMULATORS: dict[str, type[StreamAccumulator] | None] = {
    "codex": codex_factory.tool_accumulator_class,
    "claude": claude_api_factory.tool_accumulator_class,
    "claude_sdk": claude_sdk_factory.tool_accumulator_class,
    "copilot": copilot_factory.tool_accumulator_class,
}


FORMAT_CONFIGS: dict[str, FormatConfig] = {
    "chat_completions": FormatConfig(
        name="chat_completions",
        endpoint_path="/chat/completions",
        request_type_base="openai",
        description="chat completions",
    ),
    "responses": FormatConfig(
        name="responses",
        endpoint_path="/responses",
        request_type_base="response_api",
        description="responses",
    ),
    "responses_tools": FormatConfig(
        name="responses_tools",
        endpoint_path="/responses",
        request_type_base="responses_tools",
        description="responses with tools",
    ),
    "responses_thinking": FormatConfig(
        name="responses_thinking",
        endpoint_path="/responses",
        request_type_base="responses_thinking",
        description="responses with thinking",
    ),
    "responses_structured": FormatConfig(
        name="responses_structured",
        endpoint_path="/responses",
        request_type_base="responses_structured",
        description="responses structured",
    ),
    "messages": FormatConfig(
        name="messages",
        endpoint_path="/messages",
        request_type_base="anthropic",
        description="messages",
    ),
    "chat_completions_tools": FormatConfig(
        name="chat_completions_tools",
        endpoint_path="/chat/completions",
        request_type_base="openai_tools",
        description="chat completions with tools",
    ),
    "messages_tools": FormatConfig(
        name="messages_tools",
        endpoint_path="/messages",
        request_type_base="anthropic_tools",
        description="messages with tools",
    ),
    "chat_completions_thinking": FormatConfig(
        name="chat_completions_thinking",
        endpoint_path="/chat/completions",
        request_type_base="openai_thinking",
        description="chat completions with thinking",
    ),
    "chat_completions_structured": FormatConfig(
        name="chat_completions_structured",
        endpoint_path="/chat/completions",
        request_type_base="openai_structured",
        description="chat completions structured",
    ),
}


def generate_endpoint_tests() -> list[EndpointTest]:
    """Generate all endpoint test permutations from provider and format configurations."""

    tests: list[EndpointTest] = []

    for provider_key, provider in PROVIDER_CONFIGS.items():
        for format_name in provider.supported_formats:
            format_config = FORMAT_CONFIGS.get(format_name)
            if not format_config:
                continue

            endpoint = provider.base_path + format_config.endpoint_path

            for is_streaming in [True, False]:
                stream_suffix = "_stream" if is_streaming else "_non_stream"
                request_type = format_config.request_type_base + stream_suffix

                if request_type not in REQUEST_DATA:
                    continue

                stream_name_part = "_stream" if is_streaming else ""
                test_name = f"{provider_key}_{format_config.name}{stream_name_part}"

                stream_desc = "streaming" if is_streaming else "non-streaming"
                description = f"{provider.description_prefix} {format_config.description} {stream_desc}"

                tests.append(
                    EndpointTest(
                        name=test_name,
                        endpoint=endpoint,
                        stream=is_streaming,
                        request=request_type,
                        model=provider.model,
                        description=description,
                    )
                )

    return tests


ENDPOINT_TESTS: list[EndpointTest] = generate_endpoint_tests()


def add_provider(
    name: str,
    base_path: str,
    model: str,
    supported_formats: list[str],
    description_prefix: str,
) -> None:
    """Add a new provider configuration and regenerate endpoint tests."""

    global ENDPOINT_TESTS

    PROVIDER_CONFIGS[name] = ProviderConfig(
        name=name,
        base_path=base_path,
        model=model,
        supported_formats=supported_formats,
        description_prefix=description_prefix,
    )

    ENDPOINT_TESTS = generate_endpoint_tests()


def add_format(
    name: str,
    endpoint_path: str,
    request_type_base: str,
    description: str,
) -> None:
    """Add a new format configuration and regenerate endpoint tests."""

    global ENDPOINT_TESTS

    FORMAT_CONFIGS[name] = FormatConfig(
        name=name,
        endpoint_path=endpoint_path,
        request_type_base=request_type_base,
        description=description,
    )

    ENDPOINT_TESTS = generate_endpoint_tests()


def list_available_tests() -> str:
    """Generate a formatted list of available tests for help text."""

    lines = ["Available tests:"]
    for i, test in enumerate(ENDPOINT_TESTS, 1):
        lines.append(f"  {i:2d}. {test.name:<30} - {test.description}")
    return "\n".join(lines)


__all__ = [
    "MESSAGE_PAYLOADS",
    "REQUEST_DATA",
    "FORMAT_TOOLS",
    "PROVIDER_TOOL_ACCUMULATORS",
    "ENDPOINT_TESTS",
    "list_available_tests",
    "add_provider",
    "add_format",
    "OPENAI_STRUCTURED_RESPONSE_FORMAT",
    "RESPONSES_STRUCTURED_TEXT_FORMAT",
]
