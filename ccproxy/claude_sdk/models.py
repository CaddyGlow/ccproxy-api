"""Strongly-typed Pydantic models for Claude SDK types.

This module provides Pydantic models that mirror the Claude SDK types from the
official claude-code-sdk-python repository. These models enable strong typing
throughout the proxy system and provide runtime validation.

Based on: https://github.com/anthropics/claude-code-sdk-python/blob/main/src/claude_code_sdk/types.py
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ccproxy.models.requests import Usage


# Type variables for generic functions
T = TypeVar("T", bound=BaseModel)


# Generic conversion function
def to_sdk_variant(base_model: BaseModel, sdk_class: type[T]) -> T:
    """Convert a base model to its SDK variant using model_validate().

    Args:
        base_model: The base model instance to convert
        sdk_class: The target SDK class to convert to

    Returns:
        Instance of the SDK class with data from the base model

    Example:
        >>> text_block = TextBlock(text="message")
        >>> text_block_sdk = to_sdk_variant(text_block, TextBlockSDK)
    """
    return sdk_class.model_validate(base_model.model_dump())


# Core Content Block Types
class TextBlock(BaseModel):
    """Text content block from Claude SDK."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")

    model_config = ConfigDict(extra="allow")


class ToolUseBlock(BaseModel):
    """Tool use content block from Claude SDK."""

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(..., description="Unique identifier for the tool use")
    name: str = Field(..., description="Name of the tool being used")
    input: dict[str, Any] = Field(..., description="Input parameters for the tool")

    model_config = ConfigDict(extra="allow")


class ToolResultBlock(BaseModel):
    """Tool result content block from Claude SDK."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = Field(
        ..., description="ID of the tool use this result corresponds to"
    )
    content: str | list[dict[str, Any]] | None = Field(
        None, description="Result content from the tool"
    )
    is_error: bool | None = Field(
        None, description="Whether this result represents an error"
    )

    model_config = ConfigDict(extra="allow")


# Union type for basic content blocks
ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


# Message Types
class UserMessage(BaseModel):
    """User message from Claude SDK."""

    content: list[ContentBlock] = Field(
        ..., description="List of content blocks in the message"
    )

    model_config = ConfigDict(extra="allow")


class AssistantMessage(BaseModel):
    """Assistant message from Claude SDK."""

    content: list[ContentBlock] = Field(
        ..., description="List of content blocks in the message"
    )

    model_config = ConfigDict(extra="allow")


class SystemMessage(BaseModel):
    """System message from Claude SDK."""

    subtype: str = Field(..., description="Subtype of the system message")
    data: dict[str, Any] = Field(..., description="System message data")

    model_config = ConfigDict(extra="allow")


class ResultMessage(BaseModel):
    """Result message from Claude SDK."""

    session_id: str = Field(..., description="Session ID for the result")
    stop_reason: str = Field(..., description="Reason why generation stopped")
    usage: Usage = Field(default_factory=Usage, description="Token usage information")
    total_cost_usd: float | None = Field(None, description="Total cost in USD")

    model_config = ConfigDict(extra="allow")


# Custom Content Block Types for Internal Use
class SystemMessageBlock(SystemMessage):
    """Custom content block for system messages with source attribution."""

    type: Literal["system_message"] = "system_message"
    source: str = "claude_code_sdk"

    model_config = ConfigDict(extra="allow")


class ToolUseSDKBlock(BaseModel):
    """Custom content block for tool use with SDK metadata."""

    type: Literal["tool_use_sdk"] = "tool_use_sdk"
    id: str = Field(..., description="Unique identifier for the tool use")
    name: str = Field(..., description="Name of the tool being used")
    input: dict[str, Any] = Field(..., description="Input parameters for the tool")
    source: str = "claude_code_sdk"


class ToolResultSDKBlock(BaseModel):
    """Custom content block for tool results with SDK metadata."""

    type: Literal["tool_result_sdk"] = "tool_result_sdk"
    tool_use_id: str = Field(
        ..., description="ID of the tool use this result corresponds to"
    )
    content: str | list[dict[str, Any]] | None = Field(
        None, description="Result content from the tool"
    )
    is_error: bool | None = Field(
        None, description="Whether this result represents an error"
    )
    source: str = "claude_code_sdk"


class ResultMessageBlock(ResultMessage):
    """Custom content block for result messages with session data."""

    type: Literal["result_message"] = "result_message"
    source: str = "claude_code_sdk"


# Union type for all custom content blocks
SDKContentBlock = Annotated[
    TextBlock
    | ToolUseBlock
    | ToolResultBlock
    | SystemMessageBlock
    | ToolUseSDKBlock
    | ToolResultSDKBlock
    | ResultMessageBlock,
    Field(discriminator="type"),
]


# Extended content block type that includes both SDK and custom blocks
ExtendedContentBlock = SDKContentBlock


# Conversion Functions
def convert_sdk_text_block(text_content: str) -> TextBlock:
    """Convert raw text content to TextBlock model."""
    return TextBlock(text=text_content)


def convert_sdk_tool_use_block(
    tool_id: str, tool_name: str, tool_input: dict[str, Any]
) -> ToolUseBlock:
    """Convert raw tool use data to ToolUseBlock model."""
    return ToolUseBlock(id=tool_id, name=tool_name, input=tool_input)


def convert_sdk_tool_result_block(
    tool_use_id: str,
    content: str | list[dict[str, Any]] | None = None,
    is_error: bool | None = None,
) -> ToolResultBlock:
    """Convert raw tool result data to ToolResultBlock model."""
    return ToolResultBlock(tool_use_id=tool_use_id, content=content, is_error=is_error)


def convert_sdk_system_message(subtype: str, data: dict[str, Any]) -> SystemMessage:
    """Convert raw system message data to SystemMessage model."""
    return SystemMessage(subtype=subtype, data=data)


def convert_sdk_result_message(
    session_id: str,
    stop_reason: str,
    usage: dict[str, Any] | None = None,
    total_cost_usd: float | None = None,
) -> ResultMessage:
    """Convert raw result message data to ResultMessage model."""
    return ResultMessage(
        session_id=session_id,
        stop_reason=stop_reason,
        usage=Usage.model_validate(usage),
        total_cost_usd=total_cost_usd,
    )


__all__ = [
    # Generic conversion
    "to_sdk_variant",
    # Content blocks
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # Messages
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ResultMessage",
    # Custom content blocks
    "SystemMessageBlock",
    "ToolUseSDKBlock",
    "ToolResultSDKBlock",
    "ResultMessageBlock",
    "SDKContentBlock",
    "ExtendedContentBlock",
    # Conversion functions
    "convert_sdk_text_block",
    "convert_sdk_tool_use_block",
    "convert_sdk_tool_result_block",
    "convert_sdk_system_message",
    "convert_sdk_result_message",
]
