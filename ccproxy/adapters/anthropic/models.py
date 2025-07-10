"""Anthropic-specific models for the adapter."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ccproxy.models.requests import Message, MessageContent, ToolDefinition, Usage
from ccproxy.models.types import (
    ServiceTier,
    StopReason,
    ToolChoiceType,
)


class SystemMessage(BaseModel):
    """System message for Anthropic API."""

    type: Annotated[
        Literal["text"],
        Field(description="Type of the system message"),
    ] = "text"
    text: Annotated[
        str,
        Field(description="Content of the system message"),
    ]


class ThinkingConfig(BaseModel):
    """Configuration for extended thinking process."""

    enabled: Annotated[
        bool,
        Field(description="Whether to enable thinking"),
    ] = True


class MetadataParams(BaseModel):
    """Metadata parameters for Anthropic API requests."""

    user_id: Annotated[
        str | None,
        Field(description="User identifier for tracking"),
    ] = None


class ToolChoiceParams(BaseModel):
    """Tool choice parameters for Anthropic API."""

    type: Annotated[
        ToolChoiceType,
        Field(description="How to use tools"),
    ]
    name: Annotated[
        str | None,
        Field(description="Name of specific tool to use"),
    ] = None


class MessageCreateParams(BaseModel):
    """Request parameters for creating messages via Anthropic Messages API."""

    # Required fields
    model: Annotated[
        str,
        Field(
            description="The model to use for the message",
            pattern=r"^claude-.*",
        ),
    ]
    messages: Annotated[
        list[Message],
        Field(
            description="Array of messages in the conversation",
            min_length=1,
        ),
    ]
    max_tokens: Annotated[
        int,
        Field(
            description="Maximum number of tokens to generate",
            ge=1,
            le=200000,
        ),
    ]

    # Optional Anthropic API fields
    system: Annotated[
        str | list[SystemMessage] | None,
        Field(description="System prompt to provide context and instructions"),
    ] = None
    temperature: Annotated[
        float | None,
        Field(
            description="Sampling temperature between 0.0 and 1.0",
            ge=0.0,
            le=1.0,
        ),
    ] = None
    top_p: Annotated[
        float | None,
        Field(
            description="Nucleus sampling parameter",
            ge=0.0,
            le=1.0,
        ),
    ] = None
    top_k: Annotated[
        int | None,
        Field(
            description="Top-k sampling parameter",
            ge=0,
        ),
    ] = None
    stop_sequences: Annotated[
        list[str] | None,
        Field(
            description="Custom sequences where the model should stop generating",
            max_length=4,
        ),
    ] = None
    stream: Annotated[
        bool | None,
        Field(description="Whether to stream the response"),
    ] = False
    metadata: Annotated[
        MetadataParams | None,
        Field(description="Metadata about the request, including optional user_id"),
    ] = None
    tools: Annotated[
        list[ToolDefinition] | None,
        Field(description="Available tools/functions for the model to use"),
    ] = None
    tool_choice: Annotated[
        ToolChoiceParams | None,
        Field(description="How the model should use the provided tools"),
    ] = None
    service_tier: Annotated[
        ServiceTier | None,
        Field(description="Request priority level"),
    ] = None
    thinking: Annotated[
        ThinkingConfig | None,
        Field(description="Configuration for extended thinking process"),
    ] = None

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MessageContentBlock(BaseModel):
    """Content block in a message response."""

    type: Annotated[
        Literal["text"],
        Field(description="Type of content block"),
    ] = "text"
    text: Annotated[
        str,
        Field(description="Text content of the block"),
    ]


class MessageResponse(BaseModel):
    """Response model for Anthropic Messages API endpoint."""

    id: Annotated[str, Field(description="Unique identifier for the message")]
    type: Annotated[Literal["message"], Field(description="Response type")] = "message"
    role: Annotated[Literal["assistant"], Field(description="Message role")] = (
        "assistant"
    )
    content: Annotated[
        list[MessageContentBlock],
        Field(description="Array of content blocks in the response"),
    ]
    model: Annotated[str, Field(description="The model used for the response")]
    stop_reason: Annotated[
        StopReason | None, Field(description="Reason why the model stopped generating")
    ] = None
    stop_sequence: Annotated[
        str | None,
        Field(description="The stop sequence that triggered stopping (if applicable)"),
    ] = None
    usage: Annotated[Usage, Field(description="Token usage information")]
    container: Annotated[
        dict[str, Any] | None,
        Field(description="Information about container used in the request"),
    ] = None

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


__all__ = [
    "SystemMessage",
    "ThinkingConfig",
    "MetadataParams",
    "ToolChoiceParams",
    "MessageCreateParams",
    "MessageContentBlock",
    "MessageResponse",
]
