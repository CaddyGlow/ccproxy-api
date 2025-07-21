"""OpenAI-specific models for the OpenAI adapter.

This module contains OpenAI-specific data models used by the OpenAI adapter
for handling format transformations and streaming.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ccproxy.models.types import ModalityType, ReasoningEffort


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


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request model."""

    model: str = Field(..., description="ID of the model to use")
    messages: list[OpenAIMessage] = Field(
        ...,
        description="A list of messages comprising the conversation so far",
        min_length=1,
    )
    max_tokens: int | None = Field(
        None, description="The maximum number of tokens to generate", ge=1
    )
    temperature: float | None = Field(
        None, description="Sampling temperature between 0 and 2", ge=0.0, le=2.0
    )
    top_p: float | None = Field(
        None, description="Nucleus sampling parameter", ge=0.0, le=1.0
    )
    n: int | None = Field(
        1, description="Number of chat completion choices to generate", ge=1, le=128
    )
    stream: bool | None = Field(
        False, description="Whether to stream back partial progress"
    )
    stream_options: OpenAIStreamOptions | None = Field(
        None, description="Options for streaming response"
    )
    stop: str | list[str] | None = Field(
        None,
        description="Up to 4 sequences where the API will stop generating further tokens",
    )
    presence_penalty: float | None = Field(
        None,
        description="Penalize new tokens based on whether they appear in the text so far",
        ge=-2.0,
        le=2.0,
    )
    frequency_penalty: float | None = Field(
        None,
        description="Penalize new tokens based on their existing frequency in the text so far",
        ge=-2.0,
        le=2.0,
    )
    logit_bias: dict[str, float] | None = Field(
        None,
        description="Modify likelihood of specified tokens appearing in the completion",
    )
    user: str | None = Field(
        None, description="A unique identifier representing your end-user"
    )

    # Tool-related fields (new format)
    tools: list[OpenAITool] | None = Field(
        None, description="A list of tools the model may call"
    )
    tool_choice: str | OpenAIToolChoice | None = Field(
        None, description="Controls which (if any) tool is called by the model"
    )
    parallel_tool_calls: bool | None = Field(
        True, description="Whether to enable parallel function calling during tool use"
    )

    # Deprecated function calling fields (for backward compatibility)
    functions: list[dict[str, Any]] | None = Field(
        None,
        description="Deprecated. Use 'tools' instead. List of functions the model may generate JSON inputs for",
        deprecated=True,
    )
    function_call: str | dict[str, Any] | None = Field(
        None,
        description="Deprecated. Use 'tool_choice' instead. Controls how the model responds to function calls",
        deprecated=True,
    )

    # Response format
    response_format: OpenAIResponseFormat | None = Field(
        None, description="An object specifying the format that the model must output"
    )

    # Deterministic sampling
    seed: int | None = Field(
        None,
        description="This feature is in Beta. If specified, system will make a best effort to sample deterministically",
    )

    # Log probabilities
    logprobs: bool | None = Field(
        None, description="Whether to return log probabilities of the output tokens"
    )
    top_logprobs: int | None = Field(
        None,
        description="An integer between 0 and 20 specifying the number of most likely tokens to return at each token position",
        ge=0,
        le=20,
    )

    # Store/retrieval
    store: bool | None = Field(
        None,
        description="Whether to store the output for use with the Assistants API or Threads API",
    )

    # Metadata
    metadata: dict[str, Any] | None = Field(
        None, description="Additional metadata about the request"
    )

    # Reasoning effort (for o1 models)
    reasoning_effort: ReasoningEffort | None = Field(
        None,
        description="Controls how long o1 models spend thinking (only applicable to o1 models)",
    )

    # Multimodal fields
    modalities: list[ModalityType] | None = Field(
        None, description='List of modalities to use. Defaults to ["text"]'
    )

    # Audio configuration
    audio: dict[str, Any] | None = Field(
        None, description="Audio input/output configuration for multimodal models"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name - just return as-is like Anthropic endpoint."""
        return v

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[OpenAIMessage]) -> list[OpenAIMessage]:
        """Validate message structure."""
        if not v:
            raise ValueError("At least one message is required")
        return v

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, v: str | list[str] | None) -> str | list[str] | None:
        """Validate stop sequences."""
        if v is not None:
            if isinstance(v, str):
                return v
            elif isinstance(v, list):
                if len(v) > 4:
                    raise ValueError("Maximum 4 stop sequences allowed")
                return v
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[OpenAITool] | None) -> list[OpenAITool] | None:
        """Validate tools array."""
        if v is not None and len(v) > 128:
            raise ValueError("Maximum 128 tools allowed")
        return v


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
