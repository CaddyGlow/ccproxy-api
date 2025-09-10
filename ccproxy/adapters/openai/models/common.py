"""Common OpenAI models and utilities shared across API formats."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel


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


class ResponseUsage(BaseModel):
    """Usage statistics in Response API."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, Any] | None = None
    output_tokens_details: dict[str, Any] | None = None


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
    "OpenAIUsage",
    "OpenAILogprobs",
    "OpenAIFunctionCall",
    "OpenAIToolCall",
    "ResponseUsage",
    "generate_openai_response_id",
    "generate_openai_system_fingerprint",
    "format_openai_tool_call",
]
