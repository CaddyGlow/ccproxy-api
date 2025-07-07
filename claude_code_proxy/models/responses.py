"""Response models for Claude Proxy API Server compatible with Anthropic's API format."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .requests import MessageContent, Usage


class ToolCall(BaseModel):
    """Tool call made by the model."""

    id: str = Field(..., description="Unique identifier for the tool call")
    type: Literal["function"] = "function"
    function: dict[str, Any] = Field(
        ..., description="Function call details including name and arguments"
    )


class ToolUse(BaseModel):
    """Tool use content block."""

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(..., description="Unique identifier for the tool use")
    name: str = Field(..., description="Name of the tool being used")
    input: dict[str, Any] = Field(..., description="Input parameters for the tool")


class TextResponse(BaseModel):
    """Text response content block."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="The generated text content")


ResponseContent = TextResponse | ToolUse


class Choice(BaseModel):
    """Individual choice in a non-streaming response."""

    index: int = Field(..., description="Index of the choice")
    message: dict[str, Any] = Field(..., description="The generated message")
    finish_reason: str | None = Field(
        None,
        description="Reason why the model stopped generating",
    )

    model_config = ConfigDict(extra="forbid")


class StreamingChoice(BaseModel):
    """Individual choice in a streaming response."""

    index: int = Field(..., description="Index of the choice")
    delta: dict[str, Any] = Field(..., description="The incremental message content")
    finish_reason: str | None = Field(
        None,
        description="Reason why the model stopped generating",
    )

    model_config = ConfigDict(extra="forbid")


class ChatCompletionResponse(BaseModel):
    """Response model for Claude chat completions compatible with Anthropic's API."""

    id: str = Field(..., description="Unique identifier for the response")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ResponseContent] = Field(
        ..., description="Array of content blocks in the response"
    )
    model: str = Field(..., description="The model used for the response")
    stop_reason: str | None = Field(
        None,
        description="Reason why the model stopped generating",
    )
    stop_sequence: str | None = Field(
        None,
        description="The stop sequence that triggered stopping (if applicable)",
    )
    usage: Usage = Field(..., description="Token usage information")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StreamingChatCompletionResponse(BaseModel):
    """Streaming response model for Claude chat completions."""

    id: str = Field(..., description="Unique identifier for the response")
    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
    ] = Field(..., description="Type of streaming event")
    message: dict[str, Any] | None = Field(
        None, description="Message data for message events"
    )
    index: int | None = Field(None, description="Index of the content block")
    content_block: dict[str, Any] | None = Field(None, description="Content block data")
    delta: dict[str, Any] | None = Field(
        None, description="Delta data for incremental updates"
    )
    usage: Usage | None = Field(None, description="Token usage information")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ErrorResponse(BaseModel):
    """Error response model."""

    type: Literal["error"] = "error"
    error: dict[str, Any] = Field(
        ...,
        description="Error details including type and message",
    )

    model_config = ConfigDict(extra="forbid")


class APIError(BaseModel):
    """API error details."""

    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")

    model_config = ConfigDict(extra="forbid")


class PermissionToolAllowResponse(BaseModel):
    """Response model for allowed permission tool requests."""

    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] = Field(
        ...,
        description="Updated input parameters for the tool, or original input if unchanged",
        alias="updatedInput",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PermissionToolDenyResponse(BaseModel):
    """Response model for denied permission tool requests."""

    behavior: Literal["deny"] = "deny"
    message: str = Field(
        ..., description="Human-readable explanation of why the permission was denied"
    )

    model_config = ConfigDict(extra="forbid")


PermissionToolResponse = PermissionToolAllowResponse | PermissionToolDenyResponse


class RateLimitError(APIError):
    """Rate limit error."""

    type: Literal["rate_limit_error"] = "rate_limit_error"


class InvalidRequestError(APIError):
    """Invalid request error."""

    type: Literal["invalid_request_error"] = "invalid_request_error"


class AuthenticationError(APIError):
    """Authentication error."""

    type: Literal["authentication_error"] = "authentication_error"


class NotFoundError(APIError):
    """Not found error."""

    type: Literal["not_found_error"] = "not_found_error"


class OverloadedError(APIError):
    """Overloaded error."""

    type: Literal["overloaded_error"] = "overloaded_error"


class InternalServerError(APIError):
    """Internal server error."""

    type: Literal["internal_server_error"] = "internal_server_error"
