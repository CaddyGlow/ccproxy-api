"""Strongly-typed Pydantic models for Claude SDK types.

This module provides Pydantic models that mirror the Claude SDK types from the
official claude-code-sdk-python repository. These models enable strong typing
throughout the proxy system and provide runtime validation.

Based on: https://github.com/anthropics/claude-code-sdk-python/blob/main/src/claude_code_sdk/types.py
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeVar, cast

# Import Claude SDK types for isinstance checks
from claude_code_sdk import TextBlock as SDKTextBlock
from claude_code_sdk import ToolResultBlock as SDKToolResultBlock
from claude_code_sdk import ToolUseBlock as SDKToolUseBlock
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ccproxy.llms.models import anthropic as anthropic_models


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

    def to_sdk_block(self) -> dict[str, Any]:
        """Convert to ToolUseSDKBlock format for streaming."""
        return {
            "type": "tool_use_sdk",
            "id": self.id,
            "name": self.name,
            "input": self.input,
            "source": "claude_code_sdk",
        }


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

    def to_sdk_block(self) -> dict[str, Any]:
        """Convert to ToolResultSDKBlock format for streaming."""
        return {
            "type": "tool_result_sdk",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
            "source": "claude_code_sdk",
        }


class ThinkingBlock(BaseModel):
    """Thinking content block from Claude SDK.

    Note: Thinking blocks are not normally sent by Claude Code SDK, but this model
    is included for defensive programming to handle any future SDK changes or edge cases
    where thinking content might be included in SDK responses.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str = Field(..., description="Thinking content text")
    signature: str | None = Field(None, description="Optional thinking signature")

    model_config = ConfigDict(extra="allow")


# Union type for basic content blocks
ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock,
    Field(discriminator="type"),
]


# Message Types
class UserMessage(BaseModel):
    """User message from Claude SDK."""

    content: list[ContentBlock] = Field(
        ..., description="List of content blocks in the message"
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("content", mode="before")
    @classmethod
    def convert_content_blocks(cls, v: Any) -> list[Any]:
        """Convert Claude SDK dataclass blocks to Pydantic models."""
        if not isinstance(v, list):
            return []

        converted_blocks = []
        for block in v:
            if isinstance(block, SDKTextBlock | SDKToolUseBlock | SDKToolResultBlock):
                # Convert Claude SDK dataclass to dict and add type field
                if isinstance(block, SDKTextBlock):
                    converted_blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, SDKToolUseBlock):
                    converted_blocks.append(
                        cast(
                            Any,
                            {
                                "type": "tool_use",
                                "id": str(block.id),
                                "name": str(block.name),
                                "input": dict(block.input),
                            },
                        )
                    )
                elif isinstance(block, SDKToolResultBlock):
                    converted_blocks.append(
                        cast(
                            Any,
                            {
                                "type": "tool_result",
                                "tool_use_id": str(block.tool_use_id),
                                "content": block.content,
                                "is_error": block.is_error,
                            },
                        )
                    )
            else:
                converted_blocks.append(block)

        return converted_blocks


class AssistantMessage(BaseModel):
    """Assistant message from Claude SDK."""

    content: list[ContentBlock] = Field(
        ..., description="List of content blocks in the message"
    )

    model_config = ConfigDict(extra="allow")

    @field_validator("content", mode="before")
    @classmethod
    def convert_content_blocks(cls, v: Any) -> list[Any]:
        """Convert Claude SDK dataclass blocks to Pydantic models."""
        if not isinstance(v, list):
            return []

        converted_blocks = []
        for block in v:
            if isinstance(block, SDKTextBlock | SDKToolUseBlock | SDKToolResultBlock):
                # Convert Claude SDK dataclass to dict and add type field
                if isinstance(block, SDKTextBlock):
                    converted_blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, SDKToolUseBlock):
                    converted_blocks.append(
                        cast(
                            Any,
                            {
                                "type": "tool_use",
                                "id": str(block.id),
                                "name": str(block.name),
                                "input": dict(block.input),
                            },
                        )
                    )
                elif isinstance(block, SDKToolResultBlock):
                    converted_blocks.append(
                        cast(
                            Any,
                            {
                                "type": "tool_result",
                                "tool_use_id": str(block.tool_use_id),
                                "content": block.content,
                                "is_error": block.is_error,
                            },
                        )
                    )
            else:
                converted_blocks.append(block)

        return converted_blocks


class SystemMessage(BaseModel):
    """System message from Claude SDK."""

    type: Literal["system_message"] = "system_message"

    subtype: str = Field(default="", description="Subtype of the system message")
    data: dict[str, Any] = Field(
        default_factory=dict, description="System message data"
    )

    model_config = ConfigDict(extra="allow")


class ResultMessage(BaseModel):
    """Result message from Claude SDK."""

    type: Literal["result_message"] = "result_message"

    subtype: str = Field(default="", description="Subtype of the result message")
    duration_ms: int = Field(default=0, description="Total duration in milliseconds")
    duration_api_ms: int = Field(default=0, description="API duration in milliseconds")
    is_error: bool = Field(
        default=False, description="Whether this result represents an error"
    )
    num_turns: int = Field(default=0, description="Number of conversation turns")
    session_id: str = Field(default="", description="Session ID for the result")
    total_cost_usd: float | None = Field(None, description="Total cost in USD")
    usage: dict[str, Any] | None = Field(
        None, description="Usage information dictionary"
    )
    result: str | None = Field(None, description="Result string if available")

    # Add computed properties for backward compatibility
    @property
    def stop_reason(self) -> str:
        """Get stop reason from result or default to end_turn."""
        if self.is_error:
            return "error"
        return "end_turn"

    @property
    def usage_model(self) -> anthropic_models.Usage:
        """Get usage information as a Usage model for backward compatibility."""
        if self.usage is None:
            return anthropic_models.Usage(input_tokens=0, output_tokens=0)
        return anthropic_models.Usage.model_validate(self.usage)

    model_config = ConfigDict(extra="allow")


# Custom Content Block Types for Internal Use
class SDKMessageMode(SystemMessage):
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
    | ThinkingBlock
    | SDKMessageMode
    | ToolUseSDKBlock
    | ToolResultSDKBlock
    | ResultMessageBlock,
    Field(discriminator="type"),
]


# Extended content block type that includes both SDK and custom blocks
ExtendedContentBlock = SDKContentBlock

# Union definition moved after imports


# Plugin-specific content block union that includes core and SDK-specific types
# Note: We only include SDK-specific types to avoid discriminator conflicts
# with core types that have the same discriminator values
CCProxyContentBlock = Annotated[
    anthropic_models.RequestContentBlock
    | SDKMessageMode
    | ToolUseSDKBlock
    | ToolResultSDKBlock
    | ResultMessageBlock,
    Field(discriminator="type"),
]


# Plugin-specific MessageResponse that uses the extended content block types
class MessageResponse(BaseModel):
    """Plugin-specific response model that supports both core and SDK content blocks."""

    id: Annotated[str, Field(description="Unique identifier for the message")]
    type: Annotated[Literal["message"], Field(description="Response type")] = "message"
    role: Annotated[Literal["assistant"], Field(description="Message role")] = (
        "assistant"
    )
    content: Annotated[
        list[CCProxyContentBlock],
        Field(description="Array of content blocks in the response"),
    ]
    model: Annotated[str, Field(description="The model used for the response")]
    stop_reason: Annotated[
        str | None, Field(description="Reason why the model stopped generating")
    ] = None
    stop_sequence: Annotated[
        str | None,
        Field(description="The stop sequence that triggered stopping (if applicable)"),
    ] = None
    usage: Annotated[
        anthropic_models.Usage, Field(description="Token usage information")
    ]
    container: Annotated[
        dict[str, Any] | None,
        Field(description="Information about container used in the request"),
    ] = None

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# SDK Query Message Types
class SDKMessageContent(BaseModel):
    """Content structure for SDK query messages."""

    role: Literal["user"] = "user"
    content: str = Field(..., description="Message text content")

    model_config = ConfigDict(extra="forbid")


class SDKMessage(BaseModel):
    """Message format used to send queries over the Claude SDK.

    This represents the internal message structure expected by the
    Claude Code SDK client for query operations.
    """

    type: Literal["user"] = "user"
    message: SDKMessageContent = Field(
        ..., description="Message content with role and text"
    )
    parent_tool_use_id: str | None = Field(
        None, description="Optional parent tool use ID"
    )
    session_id: str | None = Field(
        None, description="Optional session ID for conversation continuity"
    )

    model_config = ConfigDict(extra="forbid")


def create_sdk_message(
    content: str,
    session_id: str | None = None,
    parent_tool_use_id: str | None = None,
) -> SDKMessage:
    """Create an SDKMessage instance for sending queries to Claude SDK.

    Args:
        content: The text content to send to Claude
        session_id: Optional session ID for conversation continuity
        parent_tool_use_id: Optional parent tool use ID

    Returns:
        SDKMessage instance ready to send to Claude SDK
    """
    return SDKMessage(
        message=SDKMessageContent(content=content),
        session_id=session_id,
        parent_tool_use_id=parent_tool_use_id,
    )


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
    subtype: str = "",
    duration_ms: int = 0,
    duration_api_ms: int = 0,
    is_error: bool = False,
    num_turns: int = 0,
    usage: dict[str, Any] | None = None,
    total_cost_usd: float | None = None,
    result: str | None = None,
) -> ResultMessage:
    """Convert raw result message data to ResultMessage model."""
    return ResultMessage(
        session_id=session_id,
        subtype=subtype,
        duration_ms=duration_ms,
        duration_api_ms=duration_api_ms,
        is_error=is_error,
        num_turns=num_turns,
        usage=usage,
        total_cost_usd=total_cost_usd,
        result=result,
    )


__all__ = [
    # Generic conversion
    "to_sdk_variant",
    # Content blocks
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ThinkingBlock",
    "ContentBlock",
    # Messages
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ResultMessage",
    # SDK Query Messages
    "SDKMessageContent",
    "SDKMessage",
    "create_sdk_message",
    # Custom content blocks
    "SDKMessageMode",
    "ToolUseSDKBlock",
    "ToolResultSDKBlock",
    "ResultMessageBlock",
    "SDKContentBlock",
    "ExtendedContentBlock",
    "CCProxyContentBlock",
    "MessageResponse",
    # Conversion functions
    "convert_sdk_text_block",
    "convert_sdk_tool_use_block",
    "convert_sdk_tool_result_block",
    "convert_sdk_system_message",
    "convert_sdk_result_message",
]
