from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ===================================================================
# Error Models
# ===================================================================


class ErrorDetail(BaseModel):
    """Base model for an error."""

    message: str


class InvalidRequestError(ErrorDetail):
    """Error for an invalid request."""

    type: Literal["invalid_request_error"] = Field(
        "invalid_request_error", alias="type"
    )


class AuthenticationError(ErrorDetail):
    """Error for authentication issues."""

    type: Literal["authentication_error"] = Field("authentication_error", alias="type")


class BillingError(ErrorDetail):
    """Error for billing issues."""

    type: Literal["billing_error"] = Field("billing_error", alias="type")


class PermissionError(ErrorDetail):
    """Error for permission issues."""

    type: Literal["permission_error"] = Field("permission_error", alias="type")


class NotFoundError(ErrorDetail):
    """Error for a resource not being found."""

    type: Literal["not_found_error"] = Field("not_found_error", alias="type")


class RateLimitError(ErrorDetail):
    """Error for rate limiting."""

    type: Literal["rate_limit_error"] = Field("rate_limit_error", alias="type")


class GatewayTimeoutError(ErrorDetail):
    """Error for a gateway timeout."""

    type: Literal["timeout_error"] = Field("timeout_error", alias="type")


class APIError(ErrorDetail):
    """A generic API error."""

    type: Literal["api_error"] = Field("api_error", alias="type")


class OverloadedError(ErrorDetail):
    """Error for when the server is overloaded."""

    type: Literal["overloaded_error"] = Field("overloaded_error", alias="type")


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


class ErrorResponse(BaseModel):
    """The structure of an error response."""

    type: Literal["error"] = Field("error", alias="type")
    error: ErrorType


# ===================================================================
# Models API Models (/v1/models)
# ===================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    type: Literal["model"] = Field("model", alias="type")
    created_at: datetime
    display_name: str


class ListModelsResponse(BaseModel):
    """Response containing a list of available models."""

    data: list[ModelInfo]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool


# ===================================================================
# Messages API Models (/v1/messages)
# ===================================================================

# --- Base Models & Common Structures for Messages ---


class ContentBlockBase(BaseModel):
    """Base model for a content block."""

    pass


class TextBlock(ContentBlockBase):
    """A block of text content."""

    type: Literal["text"] = Field("text", alias="type")
    text: str


class ImageSource(BaseModel):
    """Source of an image."""

    type: Literal["base64"] = Field("base64", alias="type")
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class ImageBlock(ContentBlockBase):
    """A block of image content."""

    type: Literal["image"] = Field("image", alias="type")
    source: ImageSource


class ToolUseBlock(ContentBlockBase):
    """Block for a tool use."""

    type: Literal["tool_use"] = Field("tool_use", alias="type")
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(ContentBlockBase):
    """Block for the result of a tool use."""

    type: Literal["tool_result"] = Field("tool_result", alias="type")
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False


class ThinkingBlock(ContentBlockBase):
    """Block representing the model's thinking process."""

    type: Literal["thinking"] = Field("thinking", alias="type")
    thinking: str
    signature: str


class RedactedThinkingBlock(ContentBlockBase):
    """A block specifying internal, redacted thinking by the model."""

    type: Literal["redacted_thinking"] = Field("redacted_thinking", alias="type")
    data: str


RequestContentBlock = Annotated[
    TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")
]

ResponseContentBlock = Annotated[
    TextBlock | ToolUseBlock | ThinkingBlock | RedactedThinkingBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """A message in the conversation."""

    role: Literal["user", "assistant"]
    content: str | list[RequestContentBlock]


class CacheCreation(BaseModel):
    """Breakdown of cached tokens."""

    ephemeral_1h_input_tokens: int
    ephemeral_5m_input_tokens: int


class ServerToolUsage(BaseModel):
    """Server-side tool usage statistics."""

    web_search_requests: int


class Usage(BaseModel):
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_creation: CacheCreation | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    server_tool_use: ServerToolUsage | None = None
    service_tier: Literal["standard", "priority", "batch"] | None = None


# --- Tool Definitions ---
class Tool(BaseModel):
    """Definition of a custom tool the model can use.

    Note: We include a discriminating `type` field for Pydantic union support.
    Anthropic's wire format may not require this for custom tools; adapters can
    drop it when serializing requests if needed.
    """

    type: Literal["custom"] = Field("custom", alias="type")
    name: str = Field(
        ..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]{1,128}$"
    )
    description: str | None = None
    input_schema: dict[str, Any]


class WebSearchTool(BaseModel):
    """Definition for the built-in web search tool."""

    type: Literal["web_search_20250305"] = Field("web_search_20250305", alias="type")
    name: Literal["web_search"] = "web_search"


# Add other specific built-in tool models here as needed
AnyTool = Annotated[
    Tool | WebSearchTool,  # Union of all tool types
    Field(
        discriminator="type",
        default_factory=lambda: locals().get("type", "custom"),
    ),
]

# --- Supporting models for CreateMessageRequest ---


class Metadata(BaseModel):
    """Metadata about the request."""

    user_id: str | None = Field(None, max_length=256)


class ThinkingConfigBase(BaseModel):
    """Base model for thinking configuration."""

    pass


class ThinkingConfigEnabled(ThinkingConfigBase):
    """Configuration for enabled thinking."""

    type: Literal["enabled"] = Field("enabled", alias="type")
    budget_tokens: int = Field(..., ge=1024)


class ThinkingConfigDisabled(ThinkingConfigBase):
    """Configuration for disabled thinking."""

    type: Literal["disabled"] = Field("disabled", alias="type")


ThinkingConfig = Annotated[
    ThinkingConfigEnabled | ThinkingConfigDisabled, Field(discriminator="type")
]


class ToolChoiceBase(BaseModel):
    """Base model for tool choice."""

    pass


class ToolChoiceAuto(ToolChoiceBase):
    """The model will automatically decide whether to use tools."""

    type: Literal["auto"] = Field("auto", alias="type")
    disable_parallel_tool_use: bool = False


class ToolChoiceAny(ToolChoiceBase):
    """The model will use any available tools."""

    type: Literal["any"] = Field("any", alias="type")
    disable_parallel_tool_use: bool = False


class ToolChoiceTool(ToolChoiceBase):
    """The model will use the specified tool."""

    type: Literal["tool"] = Field("tool", alias="type")
    name: str
    disable_parallel_tool_use: bool = False


class ToolChoiceNone(ToolChoiceBase):
    """The model will not use any tools."""

    type: Literal["none"] = Field("none", alias="type")


ToolChoice = Annotated[
    ToolChoiceAuto | ToolChoiceAny | ToolChoiceTool | ToolChoiceNone,
    Field(discriminator="type"),
]


class RequestMCPServerToolConfiguration(BaseModel):
    """Tool configuration for an MCP server."""

    allowed_tools: list[str] | None = None
    enabled: bool | None = None


class RequestMCPServerURLDefinition(BaseModel):
    """URL definition for an MCP server."""

    name: str
    type: Literal["url"] = Field("url", alias="type")
    url: str
    authorization_token: str | None = None
    tool_configuration: RequestMCPServerToolConfiguration | None = None


class Container(BaseModel):
    """Information about the container used in a request."""

    id: str
    expires_at: datetime


# --- Request Models ---


class CreateMessageRequest(BaseModel):
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
    temperature: float | None = None
    thinking: ThinkingConfig | None = None
    tools: list[AnyTool] | None = None
    tool_choice: ToolChoice | None = None
    top_k: int | None = None
    top_p: float | None = None

    model_config = ConfigDict(extra="forbid")


class CountMessageTokensRequest(BaseModel):
    """Request model for counting tokens in a message."""

    model: str
    messages: list[Message]
    system: str | list[TextBlock] | None = None
    tools: list[AnyTool] | None = None

    model_config = ConfigDict(extra="forbid")


# --- Response Models ---


class MessageResponse(BaseModel):
    """Response model for a created message."""

    id: str
    type: Literal["message"] = Field("message", alias="type")
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
    )
    stop_sequence: str | None
    usage: Usage
    container: Container | None = None


class CountMessageTokensResponse(BaseModel):
    """Response model for a token count request."""

    input_tokens: int


# ===================================================================
# Streaming Models for /v1/messages
# ===================================================================


class PingEvent(BaseModel):
    """A keep-alive event."""

    type: Literal["ping"] = Field("ping", alias="type")


class ErrorEvent(BaseModel):
    """An error event in the stream."""

    type: Literal["error"] = Field("error", alias="type")
    error: ErrorDetail


class MessageStartEvent(BaseModel):
    """Event sent when a message stream starts."""

    type: Literal["message_start"] = Field("message_start", alias="type")
    message: MessageResponse


class ContentBlockStartEvent(BaseModel):
    """Event when a content block starts."""

    type: Literal["content_block_start"] = Field("content_block_start", alias="type")
    index: int
    content_block: TextBlock | ToolUseBlock


class ContentBlockDeltaEvent(BaseModel):
    """Event for a delta in a content block."""

    type: Literal["content_block_delta"] = Field("content_block_delta", alias="type")
    index: int
    delta: TextBlock


class ContentBlockStopEvent(BaseModel):
    """Event when a content block stops."""

    type: Literal["content_block_stop"] = Field("content_block_stop", alias="type")
    index: int


class MessageDelta(BaseModel):
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


class MessageDeltaEvent(BaseModel):
    """Event for a delta in the message metadata."""

    type: Literal["message_delta"] = Field("message_delta", alias="type")
    delta: MessageDelta
    usage: Usage


class MessageStopEvent(BaseModel):
    """Event sent when a message stream stops."""

    type: Literal["message_stop"] = Field("message_stop", alias="type")


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
