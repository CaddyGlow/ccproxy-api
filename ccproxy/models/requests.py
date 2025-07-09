"""Request models for Claude Proxy API Server compatible with Anthropic's API format."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, validator


class ImageSource(BaseModel):
    """Image source data."""

    type: Annotated[Literal["base64", "url"], Field(description="Source type")]
    media_type: Annotated[
        str, Field(description="Media type (e.g., image/jpeg, image/png)")
    ]
    data: Annotated[str | None, Field(description="Base64 encoded image data")] = None
    url: Annotated[str | None, Field(description="Image URL")] = None

    model_config = ConfigDict(extra="forbid")


class ImageContent(BaseModel):
    """Image content block for multimodal messages."""

    type: Annotated[Literal["image"], Field(description="Content type")] = "image"
    source: Annotated[
        ImageSource,
        Field(description="Image source data with type (base64 or url) and media_type"),
    ]


class TextContent(BaseModel):
    """Text content block for messages."""

    type: Annotated[Literal["text"], Field(description="Content type")] = "text"
    text: Annotated[str, Field(description="The text content")]


MessageContent = TextContent | ImageContent | str


class Message(BaseModel):
    """Individual message in the conversation."""

    role: Annotated[
        Literal["user", "assistant"],
        Field(description="The role of the message sender"),
    ]
    content: Annotated[
        str | list[MessageContent], Field(description="The content of the message")
    ]


class FunctionDefinition(BaseModel):
    """Function definition for tool calling."""

    name: Annotated[str, Field(description="Function name")]
    description: Annotated[str, Field(description="Function description")]
    parameters: Annotated[
        dict[str, Any], Field(description="JSON Schema for function parameters")
    ]

    model_config = ConfigDict(extra="forbid")


class ToolDefinition(BaseModel):
    """Tool definition for function calling."""

    type: Annotated[Literal["function"], Field(description="Tool type")] = "function"
    function: Annotated[
        FunctionDefinition,
        Field(description="Function definition with name, description, and parameters"),
    ]


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: Annotated[int, Field(description="Number of input tokens")] = 0
    output_tokens: Annotated[int, Field(description="Number of output tokens")] = 0
    cache_creation_input_tokens: Annotated[
        int | None, Field(description="Number of tokens used for cache creation")
    ] = None
    cache_read_input_tokens: Annotated[
        int | None, Field(description="Number of tokens read from cache")
    ] = None
