from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field

from ccproxy.llms.formatters.shared import LlmBaseModel


# ===================================================================
# Error Models
# ===================================================================


class ErrorDetail(LlmBaseModel):
    """Base model for an error."""

    message: str


class InvalidRequestError(ErrorDetail):
    """Error for an invalid request."""

    type: Literal["invalid_request_error"] = Field(
        default="invalid_request_error", alias="type"
    )


class AuthenticationError(ErrorDetail):
    """Error for authentication issues."""

    type: Literal["authentication_error"] = Field(
        default="authentication_error", alias="type"
    )


class BillingError(ErrorDetail):
    """Error for billing issues."""

    type: Literal["billing_error"] = Field(default="billing_error", alias="type")


class PermissionError(ErrorDetail):
    """Error for permission issues."""

    type: Literal["permission_error"] = Field(default="permission_error", alias="type")


class NotFoundError(ErrorDetail):
    """Error for a resource not being found."""

    type: Literal["not_found_error"] = Field(default="not_found_error", alias="type")


class RateLimitError(ErrorDetail):
    """Error for rate limiting."""

    type: Literal["rate_limit_error"] = Field(default="rate_limit_error", alias="type")


class GatewayTimeoutError(ErrorDetail):
    """Error for a gateway timeout."""

    type: Literal["timeout_error"] = Field(default="timeout_error", alias="type")


class APIError(ErrorDetail):
    """A generic API error."""

    type: Literal["api_error"] = Field(default="api_error", alias="type")


class OverloadedError(ErrorDetail):
    """Error for when the server is overloaded."""

    type: Literal["overloaded_error"] = Field(default="overloaded_error", alias="type")


ErrorType = Annotated[
    InvalidRequestError
    | AuthenticationError
    | BillingError
    | PermissionError
    | NotFoundError
    | RateLimitError
    | GatewayTimeoutError
    | APIError
    | OverloadedError,
    Field(discriminator="type"),
]


class ErrorResponse(LlmBaseModel):
    """The structure of an error response."""

    type: Literal["error"] = Field(default="error", alias="type")
    error: ErrorType


# ===================================================================
# Models API Models (/v1/models)
# ===================================================================


class ModelInfo(LlmBaseModel):
    """Information about an available model."""

    id: str
    type: Literal["model"] = Field(default="model", alias="type")
    created_at: datetime
    display_name: str


class ListModelsResponse(LlmBaseModel):
    """Response containing a list of available models."""

    data: list[ModelInfo]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool


# ===================================================================
# Messages API Models (/v1/messages)
# ===================================================================

# --- Base Models & Common Structures for Messages ---


class ContentBlockBase(LlmBaseModel):
    """Base model for a content block."""

    pass


class TextBlock(ContentBlockBase):
    """A block of text content."""

    type: Literal["text"] = Field(default="text", alias="type")
    text: str


class TextDelta(ContentBlockBase):
    """A delta chunk of text content used in streaming events."""

    type: Literal["text_delta"] = Field(default="text_delta", alias="type")
    text: str


class ImageSource(LlmBaseModel):
    """Source of an image."""

    type: Literal["base64"] = Field(default="base64", alias="type")
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class ImageBlock(ContentBlockBase):
    """A block of image content."""

    type: Literal["image"] = Field(default="image", alias="type")
    source: ImageSource


class ToolUseBlock(ContentBlockBase):
    """Block for a tool use."""

    type: Literal["tool_use"] = Field(default="tool_use", alias="type")
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(ContentBlockBase):
    """Block for the result of a tool use."""

    type: Literal["tool_result"] = Field(default="tool_result", alias="type")
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False


class ThinkingBlock(ContentBlockBase):
    """Block representing the model's thinking process."""

    type: Literal["thinking"] = Field(default="thinking", alias="type")
    thinking: str
    signature: str


class RedactedThinkingBlock(ContentBlockBase):
    """A block specifying internal, redacted thinking by the model."""

    type: Literal["redacted_thinking"] = Field(
        default="redacted_thinking", alias="type"
    )
    data: str


RequestContentBlock = Annotated[
    TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")
]

ResponseContentBlock = Annotated[
    TextBlock | ToolUseBlock | ThinkingBlock | RedactedThinkingBlock,
    Field(discriminator="type"),
]


class Message(LlmBaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant"]
    content: str | list[RequestContentBlock]


class CacheCreation(LlmBaseModel):
    """Breakdown of cached tokens."""

    ephemeral_1h_input_tokens: int
    ephemeral_5m_input_tokens: int


class ServerToolUsage(LlmBaseModel):
    """Server-side tool usage statistics."""

    web_search_requests: int


class Usage(LlmBaseModel):
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_creation: CacheCreation | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    server_tool_use: ServerToolUsage | None = None
    service_tier: Literal["standard", "priority", "batch"] | None = None


# --- Tool Definitions ---
class Tool(LlmBaseModel):
    """Definition of a custom tool the model can use."""

    # Discriminator field for union matching
    type: Literal["custom"] = Field(default="custom", alias="type")
    name: str = Field(
        ..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]{1,128}$"
    )
    description: str | None = None
    input_schema: dict[str, Any]


class WebSearchTool(LlmBaseModel):
    """Definition for the built-in web search tool."""

    type: Literal["web_search_20250305"] = Field(
        default="web_search_20250305", alias="type"
    )
    name: Literal["web_search"] = "web_search"


# Add other specific built-in tool models here as needed
AnyTool = Annotated[
    Tool | WebSearchTool,  # Union of all tool types
    Field(discriminator="type"),
]

# --- Supporting models for CreateMessageRequest ---


class Metadata(LlmBaseModel):
    """Metadata about the request."""

    user_id: str | None = Field(None, max_length=256)


class ThinkingConfigBase(LlmBaseModel):
    """Base model for thinking configuration."""

    pass


class ThinkingConfigEnabled(ThinkingConfigBase):
    """Configuration for enabled thinking."""

    type: Literal["enabled"] = Field(default="enabled", alias="type")
    budget_tokens: int = Field(..., ge=1024)


class ThinkingConfigDisabled(ThinkingConfigBase):
    """Configuration for disabled thinking."""

    type: Literal["disabled"] = Field(default="disabled", alias="type")


ThinkingConfig = Annotated[
    ThinkingConfigEnabled | ThinkingConfigDisabled, Field(discriminator="type")
]


class ToolChoiceBase(LlmBaseModel):
    """Base model for tool choice."""

    pass


class ToolChoiceAuto(ToolChoiceBase):
    """The model will automatically decide whether to use tools."""

    type: Literal["auto"] = Field(default="auto", alias="type")
    disable_parallel_tool_use: bool = False


class ToolChoiceAny(ToolChoiceBase):
    """The model will use any available tools."""

    type: Literal["any"] = Field(default="any", alias="type")
    disable_parallel_tool_use: bool = False


class ToolChoiceTool(ToolChoiceBase):
    """The model will use the specified tool."""

    type: Literal["tool"] = Field(default="tool", alias="type")
    name: str
    disable_parallel_tool_use: bool = False


class ToolChoiceNone(ToolChoiceBase):
    """The model will not use any tools."""

    type: Literal["none"] = Field(default="none", alias="type")


ToolChoice = Annotated[
    ToolChoiceAuto | ToolChoiceAny | ToolChoiceTool | ToolChoiceNone,
    Field(discriminator="type"),
]


class RequestMCPServerToolConfiguration(LlmBaseModel):
    """Tool configuration for an MCP server."""

    allowed_tools: list[str] | None = None
    enabled: bool | None = None


class RequestMCPServerURLDefinition(LlmBaseModel):
    """URL definition for an MCP server."""

    name: str
    type: Literal["url"] = Field(default="url", alias="type")
    url: str
    authorization_token: str | None = None
    tool_configuration: RequestMCPServerToolConfiguration | None = None


class Container(LlmBaseModel):
    """Information about the container used in a request."""

    id: str
    expires_at: datetime


# --- Request Models ---


class CreateMessageRequest(LlmBaseModel):
    """Request model for creating a new message."""

    model: str
    messages: list[Message]
    max_tokens: int
    container: str | None = None
    mcp_servers: list[RequestMCPServerURLDefinition] | None = None
    metadata: Metadata | None = None
    service_tier: Literal["auto", "standard_only"] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    system: str | list[TextBlock] | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    thinking: ThinkingConfig | None = None
    tools: list[AnyTool] | None = None
    tool_choice: ToolChoice | None = Field(default=None)
    top_k: int | None = None
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class CountMessageTokensRequest(LlmBaseModel):
    """Request model for counting tokens in a message."""

    model: str
    messages: list[Message]
    system: str | list[TextBlock] | None = None
    tools: list[AnyTool] | None = None


# --- Response Models ---


class MessageResponse(LlmBaseModel):
    """Response model for a created message."""

    id: str
    type: Literal["message"] = Field(default="message", alias="type")
    role: Literal["assistant"]
    content: list[ResponseContentBlock]
    model: str
    stop_reason: (
        Literal[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]
        | None
    ) = None
    stop_sequence: str | None = None
    usage: Usage
    container: Container | None = None


class CountMessageTokensResponse(LlmBaseModel):
    """Response model for a token count request."""

    input_tokens: int


# ===================================================================
# Streaming Models for /v1/messages
# ===================================================================


class PingEvent(LlmBaseModel):
    """A keep-alive event."""

    type: Literal["ping"] = Field(default="ping", alias="type")


class ErrorEvent(LlmBaseModel):
    """An error event in the stream."""

    type: Literal["error"] = Field(default="error", alias="type")
    error: ErrorDetail


class MessageStartEvent(LlmBaseModel):
    """Event sent when a message stream starts."""

    type: Literal["message_start"] = Field(default="message_start", alias="type")
    message: MessageResponse


class ContentBlockStartEvent(LlmBaseModel):
    """Event when a content block starts."""

    type: Literal["content_block_start"] = Field(
        default="content_block_start", alias="type"
    )
    index: int
    content_block: ResponseContentBlock


class ContentBlockDeltaEvent(LlmBaseModel):
    """Event for a delta in a content block."""

    type: Literal["content_block_delta"] = Field(
        default="content_block_delta", alias="type"
    )
    index: int
    # Anthropic streams use delta.type == "text_delta" during streaming.
    # Accept both TextBlock (some SDKs may coerce) and TextDelta.
    delta: Annotated[TextBlock | TextDelta, Field(discriminator="type")]


class ContentBlockStopEvent(LlmBaseModel):
    """Event when a content block stops."""

    type: Literal["content_block_stop"] = Field(
        default="content_block_stop", alias="type"
    )
    index: int


class MessageDelta(LlmBaseModel):
    """The delta in a message delta event."""

    stop_reason: (
        Literal[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]
        | None
    ) = None
    stop_sequence: str | None = None


class MessageDeltaEvent(LlmBaseModel):
    """Event for a delta in the message metadata."""

    type: Literal["message_delta"] = Field(default="message_delta", alias="type")
    delta: MessageDelta
    usage: Usage


class MessageStopEvent(LlmBaseModel):
    """Event sent when a message stream stops."""

    type: Literal["message_stop"] = Field(default="message_stop", alias="type")


MessageStreamEvent = Annotated[
    PingEvent
    | ErrorEvent
    | MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent,
    Field(discriminator="type"),
]
