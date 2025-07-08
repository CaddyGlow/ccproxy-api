"""Message models for Anthropic Messages API endpoint."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .requests import ClaudeCodeOptionsMixin, Message, MessageContent, Usage


class SystemMessage(BaseModel):
    """System message content block."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="System message text")


class MessageRequest(ClaudeCodeOptionsMixin):
    """Request model for Anthropic Messages API endpoint."""

    model: str = Field(
        ...,
        description="The model to use for the message",
        pattern=r"^claude-.*",
    )
    max_tokens: int = Field(
        ...,
        description="Maximum number of tokens to generate",
        ge=1,
        le=200000,
    )
    messages: list[Message] = Field(
        ...,
        description="Array of messages in the conversation",
        min_length=1,
    )
    system: str | list[SystemMessage] | None = Field(
        None,
        description="System prompt to provide context and instructions",
    )
    temperature: float | None = Field(
        None,
        description="Sampling temperature between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    top_p: float | None = Field(
        None,
        description="Nucleus sampling parameter",
        ge=0.0,
        le=1.0,
    )
    top_k: int | None = Field(
        None,
        description="Top-k sampling parameter",
        ge=0,
    )
    stream: bool | None = Field(
        False,
        description="Whether to stream the response",
    )
    stop_sequences: list[str] | None = Field(
        None,
        description="Custom sequences where the model should stop generating",
        max_length=4,
    )

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
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
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


class MessageContentBlock(BaseModel):
    """Content block in a message response."""

    type: Literal["text", "tool_use"] = Field(..., description="Type of content block")
    text: str | None = Field(None, description="Text content (for text blocks)")
    id: str | None = Field(None, description="Unique ID (for tool_use blocks)")
    name: str | None = Field(None, description="Tool name (for tool_use blocks)")
    input: dict[str, Any] | None = Field(
        None, description="Tool input (for tool_use blocks)"
    )


class MessageResponse(BaseModel):
    """Response model for Anthropic Messages API endpoint."""

    id: str = Field(..., description="Unique identifier for the message")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[MessageContentBlock] = Field(
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
