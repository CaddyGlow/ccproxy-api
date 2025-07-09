"""Request models for Claude Proxy API Server compatible with Anthropic's API format."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, validator


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
