"""Response models for Claude Proxy API Server compatible with Anthropic's API format."""

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field

from .base import AnthropicBaseModel
from .requests import Usage


class ToolCall(AnthropicBaseModel):
    """Tool call made by the model."""

    id: Annotated[str, Field(description="Unique identifier for the tool call")]
    type: Annotated[Literal["function"], Field(description="Tool call type")] = (
        "function"
    )
    function: Annotated[
        dict[str, Any],
        Field(description="Function call details including name and arguments"),
    ]


class ToolUse(AnthropicBaseModel):
    """Tool use content block."""

    type: Annotated[Literal["tool_use"], Field(description="Content type")] = "tool_use"
    id: Annotated[str, Field(description="Unique identifier for the tool use")]
    name: Annotated[str, Field(description="Name of the tool being used")]
    input: Annotated[dict[str, Any], Field(description="Input parameters for the tool")]


class TextResponse(AnthropicBaseModel):
    """Text response content block."""

    type: Annotated[Literal["text"], Field(description="Content type")] = "text"
    text: Annotated[str, Field(description="The generated text content")]


ResponseContent = TextResponse | ToolUse


class Choice(AnthropicBaseModel):
    """Individual choice in a non-streaming response."""

    index: Annotated[int, Field(description="Index of the choice")]
    message: Annotated[dict[str, Any], Field(description="The generated message")]
    finish_reason: Annotated[
        str | None, Field(description="Reason why the model stopped generating")
    ] = None

    model_config = ConfigDict(extra="allow")


class StreamingChoice(AnthropicBaseModel):
    """Individual choice in a streaming response."""

    index: Annotated[int, Field(description="Index of the choice")]
    delta: Annotated[
        dict[str, Any], Field(description="The incremental message content")
    ]
    finish_reason: Annotated[
        str | None, Field(description="Reason why the model stopped generating")
    ] = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionResponse(AnthropicBaseModel):
    """Response model for Claude chat completions compatible with Anthropic's API."""

    id: Annotated[str, Field(description="Unique identifier for the response")]
    type: Annotated[Literal["message"], Field(description="Response type")] = "message"
    role: Annotated[Literal["assistant"], Field(description="Message role")] = (
        "assistant"
    )
    content: Annotated[
        list[ResponseContent],
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
    usage: Annotated[Usage, Field(description="Token usage information")]

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class StreamingChatCompletionResponse(AnthropicBaseModel):
    """Streaming response model for Claude chat completions."""

    id: Annotated[str, Field(description="Unique identifier for the response")]
    type: Annotated[
        Literal[
            "message_start",
            "message_delta",
            "message_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "ping",
        ],
        Field(description="Type of streaming event"),
    ]
    message: Annotated[
        dict[str, Any] | None, Field(description="Message data for message events")
    ] = None
    index: Annotated[int | None, Field(description="Index of the content block")] = None
    content_block: Annotated[
        dict[str, Any] | None, Field(description="Content block data")
    ] = None
    delta: Annotated[
        dict[str, Any] | None, Field(description="Delta data for incremental updates")
    ] = None
    usage: Annotated[Usage | None, Field(description="Token usage information")] = None

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class ErrorResponse(AnthropicBaseModel):
    """Error response model."""

    type: Annotated[Literal["error"], Field(description="Response type")] = "error"
    error: Annotated[
        dict[str, Any], Field(description="Error details including type and message")
    ]

    model_config = ConfigDict(extra="allow")


class APIError(AnthropicBaseModel):
    """API error details."""

    type: Annotated[str, Field(description="Error type")]
    message: Annotated[str, Field(description="Error message")]

    model_config = ConfigDict(
        extra="allow", validate_by_alias=True, validate_by_name=True
    )


class RateLimitError(APIError):
    """Rate limit error."""

    type: Annotated[Literal["rate_limit_error"], Field(description="Error type")] = (
        "rate_limit_error"
    )


class InvalidRequestError(APIError):
    """Invalid request error."""

    type: Annotated[
        Literal["invalid_request_error"], Field(description="Error type")
    ] = "invalid_request_error"


class AuthenticationError(APIError):
    """Authentication error."""

    type: Annotated[
        Literal["authentication_error"], Field(description="Error type")
    ] = "authentication_error"


class NotFoundError(APIError):
    """Not found error."""

    type: Annotated[Literal["not_found_error"], Field(description="Error type")] = (
        "not_found_error"
    )


class OverloadedError(APIError):
    """Overloaded error."""

    type: Annotated[Literal["overloaded_error"], Field(description="Error type")] = (
        "overloaded_error"
    )


class InternalServerError(APIError):
    """Internal server error."""

    type: Annotated[
        Literal["internal_server_error"], Field(description="Error type")
    ] = "internal_server_error"
