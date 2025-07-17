"""Common type aliases used across the ccproxy models."""

from typing import Literal, TypeAlias

from typing_extensions import TypedDict


# Message and content types
MessageRole: TypeAlias = Literal["user", "assistant", "system", "tool"]
OpenAIMessageRole: TypeAlias = Literal[
    "system", "user", "assistant", "tool", "developer"
]
ContentBlockType: TypeAlias = Literal[
    "text", "image", "image_url", "tool_use", "thinking"
]
OpenAIContentType: TypeAlias = Literal["text", "image_url"]

# Tool-related types
ToolChoiceType: TypeAlias = Literal["auto", "any", "tool", "none", "required"]
OpenAIToolChoiceType: TypeAlias = Literal["none", "auto", "required"]
ToolType: TypeAlias = Literal["function", "custom"]

# Response format types
ResponseFormatType: TypeAlias = Literal["text", "json_object", "json_schema"]

# Service tier types
ServiceTier: TypeAlias = Literal["auto", "standard_only"]

# Stop reasons (re-exported from messages for convenience)
StopReason: TypeAlias = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
]

# OpenAI finish reasons
OpenAIFinishReason: TypeAlias = Literal[
    "stop", "length", "tool_calls", "content_filter"
]

# Error types
ErrorType: TypeAlias = Literal[
    "error",
    "rate_limit_error",
    "invalid_request_error",
    "authentication_error",
    "not_found_error",
    "overloaded_error",
    "internal_server_error",
]

# Stream event types
StreamEventType: TypeAlias = Literal[
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "ping",
]

# Image source types
ImageSourceType: TypeAlias = Literal["base64", "url"]

# Modality types
ModalityType: TypeAlias = Literal["text", "audio"]

# Reasoning effort types (OpenAI o1 models)
ReasoningEffort: TypeAlias = Literal["low", "medium", "high"]

# OpenAI object types
OpenAIObjectType: TypeAlias = Literal[
    "chat.completion", "chat.completion.chunk", "model", "list"
]

# Permission behavior types
PermissionBehavior: TypeAlias = Literal["allow", "deny"]


# Usage and streaming related types
class UsageData(TypedDict, total=False):
    """Token usage data extracted from streaming or non-streaming responses."""

    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    event_type: StreamEventType | None


class StreamingTokenMetrics(TypedDict, total=False):
    """Accumulated token metrics during streaming."""

    tokens_input: int | None
    tokens_output: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cost_usd: float | None
