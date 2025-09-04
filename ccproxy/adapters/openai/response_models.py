"""OpenAI Response API models.

This module contains data models for OpenAI's Response API format
used by Codex/ChatGPT backend.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


# Request Models


class ResponseFunction(BaseModel):
    """Function definition in tool call."""

    name: str
    arguments: str  # JSON string


class ResponseToolCall(BaseModel):
    """Tool call in Response API format."""

    type: Literal["tool_call"]
    id: str
    function: ResponseFunction


class ResponseMessageContent(BaseModel):
    """Content block in a Response API message with function calling support."""

    type: Literal["input_text", "output_text", "tool_call"]
    text: str | None = None  # For text content
    # Tool call fields (when type is "tool_call")
    id: str | None = None
    function: ResponseFunction | None = None


class ResponseMessage(BaseModel):
    """Message in Response API format."""

    type: Literal["message"]
    id: str | None = None
    role: Literal["user", "assistant", "system"]
    content: list[ResponseMessageContent]


class ResponseToolFunction(BaseModel):
    """Function definition in tool for requests."""

    name: str
    description: str | None = None
    parameters: dict[str, Any]


class ResponseTool(BaseModel):
    """Tool definition for Response API."""

    type: Literal["function"]
    function: ResponseToolFunction


class ResponseToolChoice(BaseModel):
    """Tool choice object format for Response API."""

    type: Literal["function"]
    function: dict[str, str]  # {"name": "function_name"}


class ResponseReasoning(BaseModel):
    """Reasoning configuration for Response API."""

    effort: Literal["low", "medium", "high"] = "medium"
    summary: Literal["auto", "none"] | None = "auto"


class ResponseRequest(BaseModel):
    """OpenAI Response API request format with function calling support."""

    model: str
    instructions: str | None = None
    input: list[ResponseMessage]
    stream: bool = True
    tool_choice: Literal["auto", "none", "required"] | ResponseToolChoice | str = "auto"
    tools: list[ResponseTool] | None = None
    parallel_tool_calls: bool = False
    max_tool_calls: int | None = None
    reasoning: ResponseReasoning | None = None
    store: bool = False
    include: list[str] | None = None
    prompt_cache_key: str | None = None
    # Note: The following OpenAI parameters are not supported by Response API (Codex backend):
    # temperature, max_output_tokens, top_p, frequency_penalty, presence_penalty, metadata
    # If included, they'll cause "Unsupported parameter" errors


# Response Models


class ResponseOutput(BaseModel):
    """Output content in Response API."""

    id: str
    type: Literal["message"]
    status: Literal["completed", "in_progress"]
    content: list[ResponseMessageContent]
    role: Literal["assistant"]


class ResponseUsage(BaseModel):
    """Usage statistics in Response API."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: dict[str, Any] | None = None
    output_tokens_details: dict[str, Any] | None = None


class ResponseReasoningContent(BaseModel):
    """Reasoning content in response."""

    effort: Literal["low", "medium", "high"]
    summary: str | None = None
    encrypted_content: str | None = None


class ResponseData(BaseModel):
    """Complete response data structure."""

    id: str
    object: Literal["response"]
    created_at: int
    status: Literal["completed", "failed", "cancelled"]
    background: bool = False
    error: dict[str, Any] | None = None
    incomplete_details: dict[str, Any] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    model: str
    output: list[ResponseOutput]
    parallel_tool_calls: bool = False
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    reasoning: ResponseReasoningContent | None = None
    safety_identifier: str | None = None
    service_tier: str | None = None
    store: bool = False
    temperature: float | None = None
    text: dict[str, Any] | None = None
    tool_choice: str | None = None
    tools: list[dict[str, Any]] | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncation: str | None = None
    usage: ResponseUsage | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None


class ResponseCompleted(BaseModel):
    """Complete response from Response API."""

    type: Literal["response.completed"]
    sequence_number: int
    response: ResponseData


# Streaming Models


class StreamingDelta(BaseModel):
    """Delta content in streaming response with function calling."""

    content: str | None = None
    role: Literal["assistant"] | None = None
    reasoning_content: str | None = None
    output: list[dict[str, Any]] | None = None
    # Function calling deltas
    tool_calls: list[dict[str, Any]] | None = None


class StreamingChoice(BaseModel):
    """Choice in streaming response."""

    index: int
    delta: StreamingDelta
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = (
        None
    )


class StreamingChunk(BaseModel):
    """Streaming chunk from Response API."""

    id: str
    object: Literal["response.chunk", "chat.completion.chunk"]
    created: int
    model: str
    choices: list[StreamingChoice]
    usage: ResponseUsage | None = None
    system_fingerprint: str | None = None


class StreamingEvent(BaseModel):
    """Server-sent event wrapper for streaming with function calling support."""

    event: (
        Literal[
            "response.created",
            "response.output.started",
            "response.output.delta",
            "response.output.completed",
            "response.completed",
            "response.failed",
            # Function calling specific events
            "response.tool_call.started",
            "response.tool_call.delta",
            "response.tool_call.completed",
        ]
        | None
    ) = None
    data: dict[str, Any] | str


# Utility Models for Function Calling


class FunctionCallDelta(BaseModel):
    """Delta for streaming function call arguments."""

    name: str | None = None
    arguments: str | None = None  # Partial JSON string


class ToolCallState(BaseModel):
    """State tracking for streaming tool calls."""

    id: str
    name: str
    accumulated_arguments: str = ""
    completed: bool = False
