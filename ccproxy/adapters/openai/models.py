"""OpenAI-specific models for the OpenAI adapter.

This module contains OpenAI-specific data models used by the OpenAI adapter
for handling format transformations and streaming.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OpenAIMessageContent(BaseModel):
    """OpenAI message content block."""

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict[str, Any] | None = None


class OpenAIMessage(BaseModel):
    """OpenAI message model."""

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[OpenAIMessageContent] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class OpenAIFunction(BaseModel):
    """OpenAI function definition."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class OpenAITool(BaseModel):
    """OpenAI tool definition."""

    type: Literal["function"] = "function"
    function: OpenAIFunction


class OpenAIToolChoice(BaseModel):
    """OpenAI tool choice specification."""

    type: Literal["function"]
    function: dict[str, str]


class OpenAIResponseFormat(BaseModel):
    """OpenAI response format specification."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class OpenAIStreamOptions(BaseModel):
    """OpenAI stream options."""

    include_usage: bool = False


class OpenAIUsage(BaseModel):
    """OpenAI usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None


class OpenAILogprobs(BaseModel):
    """OpenAI log probabilities."""

    content: list[dict[str, Any]] | None = None


class OpenAIFunctionCall(BaseModel):
    """OpenAI function call."""

    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    """OpenAI tool call."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


class OpenAIResponseMessage(BaseModel):
    """OpenAI response message."""

    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    refusal: str | None = None


class OpenAIChoice(BaseModel):
    """OpenAI choice in response."""

    index: int
    message: OpenAIResponseMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None
    logprobs: OpenAILogprobs | None = None


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage
    system_fingerprint: str | None = None

    model_config = ConfigDict(extra="forbid")


class OpenAIStreamingDelta(BaseModel):
    """OpenAI streaming delta."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIStreamingChoice(BaseModel):
    """OpenAI streaming choice."""

    index: int
    delta: OpenAIStreamingDelta
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = (
        None
    )
    logprobs: OpenAILogprobs | None = None


class OpenAIStreamingChatCompletionResponse(BaseModel):
    """OpenAI streaming chat completion response."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIStreamingChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str | None = None

    model_config = ConfigDict(extra="forbid")


class OpenAIModelInfo(BaseModel):
    """OpenAI model information."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class OpenAIModelsResponse(BaseModel):
    """OpenAI models list response."""

    object: Literal["list"] = "list"
    data: list[OpenAIModelInfo]


class OpenAIErrorDetail(BaseModel):
    """OpenAI error detail."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    """OpenAI error response."""

    error: OpenAIErrorDetail


def generate_openai_response_id() -> str:
    """Generate an OpenAI-compatible response ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:29]}"


def generate_openai_system_fingerprint() -> str:
    """Generate an OpenAI-compatible system fingerprint."""
    return f"fp_{uuid.uuid4().hex[:8]}"


def format_openai_tool_call(tool_use: dict[str, Any]) -> OpenAIToolCall:
    """Convert Anthropic tool use to OpenAI tool call format."""
    tool_input = tool_use.get("input", {})
    if isinstance(tool_input, dict):
        arguments_str = json.dumps(tool_input)
    else:
        arguments_str = str(tool_input)

    return OpenAIToolCall(
        id=tool_use.get("id", ""),
        type="function",
        function=OpenAIFunctionCall(
            name=tool_use.get("name", ""),
            arguments=arguments_str,
        ),
    )


__all__ = [
    "OpenAIMessageContent",
    "OpenAIMessage",
    "OpenAIFunction",
    "OpenAITool",
    "OpenAIToolChoice",
    "OpenAIResponseFormat",
    "OpenAIStreamOptions",
    "OpenAIUsage",
    "OpenAILogprobs",
    "OpenAIFunctionCall",
    "OpenAIToolCall",
    "OpenAIResponseMessage",
    "OpenAIChoice",
    "OpenAIChatCompletionResponse",
    "OpenAIStreamingDelta",
    "OpenAIStreamingChoice",
    "OpenAIStreamingChatCompletionResponse",
    "OpenAIModelInfo",
    "OpenAIModelsResponse",
    "OpenAIErrorDetail",
    "OpenAIErrorResponse",
    "generate_openai_response_id",
    "generate_openai_system_fingerprint",
    "format_openai_tool_call",
]
