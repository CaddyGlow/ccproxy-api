"""Message models for Anthropic Messages API endpoint."""

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .claude_sdk import SDKContentBlock
from .requests import Message, ToolDefinition, Usage


if TYPE_CHECKING:
    pass
from .types import ServiceTier, StopReason, ToolChoiceType


class SystemMessage(BaseModel):
    """System message content block."""

    type: Annotated[Literal["text"], Field(description="Content type")] = "text"
    text: Annotated[str, Field(description="System message text")]


class ThinkingConfig(BaseModel):
    """Configuration for extended thinking process."""

    type: Annotated[Literal["enabled"], Field(description="Enable thinking mode")] = (
        "enabled"
    )
    budget_tokens: Annotated[
        int, Field(description="Token budget for thinking process", ge=1024)
    ]


class MetadataParams(BaseModel):
    """Metadata about the request."""

    user_id: Annotated[
        str | None,
        Field(description="External identifier for the user", max_length=256),
    ] = None

    model_config = ConfigDict(extra="allow")  # Allow additional fields in metadata


class ToolChoiceParams(BaseModel):
    """Tool choice configuration."""

    type: Annotated[ToolChoiceType, Field(description="How the model should use tools")]
    name: Annotated[
        str | None, Field(description="Specific tool name (when type is 'tool')")
    ] = None
    disable_parallel_tool_use: Annotated[
        bool, Field(description="Disable parallel tool use")
    ] = False


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
    stop_reason: Annotated[
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

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate that the model is a supported Claude model."""
        supported_models = {
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
        }

        if v not in supported_models and not v.startswith("claude-"):
            raise ValueError(f"Model {v} is not supported")

        return v

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        """Validate message alternation and content."""
        if not v:
            raise ValueError("At least one message is required")

        # First message must be from user
        if v[0].role != "user":
            raise ValueError("First message must be from user")

        # Check for proper alternation
        for i in range(1, len(v)):
            if v[i].role == v[i - 1].role:
                raise ValueError("Messages must alternate between user and assistant")

        return v

    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, v: list[str] | None) -> list[str] | None:
        """Validate stop sequences."""
        if v is not None:
            if len(v) > 4:
                raise ValueError("Maximum 4 stop sequences allowed")
            for seq in v:
                if len(seq) > 100:
                    raise ValueError("Stop sequences must be 100 characters or less")
        return v

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TextContentBlock(BaseModel):
    """Text content block."""

    type: Literal["text"]
    text: str


class ToolUseContentBlock(BaseModel):
    """Tool use content block."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ThinkingContentBlock(BaseModel):
    """Thinking content block."""

    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


MessageContentBlock = Annotated[
    TextContentBlock | ToolUseContentBlock | ThinkingContentBlock,
    Field(discriminator="type"),
]


CCProxyContentBlock = MessageContentBlock | SDKContentBlock


class MessageResponse(BaseModel):
    """Response model for Anthropic Messages API endpoint."""

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
