"""Request models for Claude Proxy API Server compatible with Anthropic's API format."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, validator


class ClaudeCodeOptionsMixin(BaseModel):
    """Mixin class for ClaudeCodeOptions parameters (not part of official Anthropic API)."""
    
    # Extended ClaudeCodeOptions parameters (not part of official Anthropic API)
    # These fields allow passing Claude Code SDK specific options through the API
    max_thinking_tokens: int | None = Field(None, description="Claude code settings")
    allowed_tools: list[str] | None = Field(None, description="List of allowed tools")
    disallowed_tools: list[str] | None = Field(
        None, description="List of disallowed tools"
    )
    append_system_prompt: str | None = Field(
        None, description="Additional system prompt to append"
    )
    mcp_tools: list[str] | None = Field(None, description="MCP tools to enable")
    mcp_servers: dict[str, Any] | None = Field(
        None, description="MCP server configurations"
    )
    permission_mode: Literal["default", "acceptEdits", "bypassPermissions"] | None = (
        Field(None, description="Permission mode")
    )
    continue_conversation: bool | None = Field(
        False, description="Continue previous conversation"
    )
    resume: str | None = Field(None, description="Resume conversation ID")
    max_turns: int | None = Field(None, description="Maximum conversation turns")
    permission_prompt_tool_name: str | None = Field(
        None, description="Permission prompt tool name"
    )
    cwd: str | None = Field(None, description="Working directory path")

    model_config = ConfigDict(extra="forbid")


class ImageSource(BaseModel):
    """Image source data."""

    type: Literal["base64", "url"] = Field(..., description="Source type")
    media_type: str = Field(..., description="Media type (e.g., image/jpeg, image/png)")
    data: str | None = Field(None, description="Base64 encoded image data")
    url: str | None = Field(None, description="Image URL")

    model_config = ConfigDict(extra="forbid")


class ImageContent(BaseModel):
    """Image content block for multimodal messages."""

    type: Literal["image"] = "image"
    source: ImageSource = Field(
        ..., description="Image source data with type (base64 or url) and media_type"
    )


class TextContent(BaseModel):
    """Text content block for messages."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content")


MessageContent = TextContent | ImageContent | str


class Message(BaseModel):
    """Individual message in the conversation."""

    role: Literal["user", "assistant"] = Field(
        ..., description="The role of the message sender"
    )
    content: str | list[MessageContent] = Field(
        ..., description="The content of the message"
    )


class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""

    name: str = Field(..., description="Function name")
    description: str = Field(..., description="Function description")
    parameters: dict[str, Any] = Field(
        ..., description="JSON Schema for function parameters"
    )

    model_config = ConfigDict(extra="forbid")


class ToolDefinition(BaseModel):
    """Tool definition for function calling."""

    type: Literal["function"] = "function"
    function: FunctionDefinition = Field(
        ..., description="Function definition with name, description, and parameters"
    )


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: int = Field(0, description="Number of input tokens")
    output_tokens: int = Field(0, description="Number of output tokens")
    cache_creation_input_tokens: int | None = Field(
        None, description="Number of tokens used for cache creation"
    )
    cache_read_input_tokens: int | None = Field(
        None, description="Number of tokens read from cache"
    )


class ToolChoice(BaseModel):
    """Tool choice specification."""

    type: Literal["auto", "any", "tool"] = Field(..., description="How to use tools")
    name: str | None = Field(None, description="Specific tool name to use")

    model_config = ConfigDict(extra="forbid")


class ChatCompletionRequest(ClaudeCodeOptionsMixin):
    """Request model for Claude chat completions compatible with Anthropic's API."""

    model: str = Field(
        ...,
        description="The model to use for completion",
        pattern=r"^claude-.*",
    )
    messages: list[Message] = Field(
        ...,
        description="Array of messages in the conversation",
        min_length=1,
    )
    max_tokens: int = Field(
        ...,
        description="Maximum number of tokens to generate",
        ge=1,
        le=200000,
    )
    system: str | None = Field(
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
    tools: list[ToolDefinition] | None = Field(
        None,
        description="Available tools/functions for the model to use",
    )
    tool_choice: dict[str, Any] | None = Field(
        None,
        description="How the model should use the provided tools",
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
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
        }

        if v not in supported_models and not v.startswith("claude-"):
            # Allow the model if it matches the claude pattern
            # This provides forward compatibility for new models
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

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, use_enum_values=True
    )
