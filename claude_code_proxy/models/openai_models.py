"""OpenAI-compatible Pydantic models for Claude Proxy API Server.

This module provides OpenAI API compatible request and response models that can be used
as drop-in replacements for OpenAI's chat completion endpoints. The models follow OpenAI's
API specification exactly while internally mapping to Claude models.
"""

import time
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# OpenAI Message Models
class OpenAIMessageContent(BaseModel):
    """Content within an OpenAI message - can be text or image."""

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict[str, str] | None = None

    model_config = ConfigDict(extra="forbid")


class OpenAIMessage(BaseModel):
    """OpenAI-compatible message model."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message sender"
    )
    content: str | list[OpenAIMessageContent] = Field(
        ..., description="The content of the message"
    )
    name: str | None = Field(None, description="The name of the participant (optional)")
    tool_calls: list[dict[str, Any]] | None = Field(
        None,
        description="Tool calls made by the assistant (only for assistant messages)",
    )
    tool_call_id: str | None = Field(
        None,
        description="Tool call this message is responding to (only for tool messages)",
    )

    model_config = ConfigDict(extra="forbid")


# OpenAI Tool Models
class OpenAIFunction(BaseModel):
    """OpenAI function definition."""

    name: str = Field(..., description="The name of the function")
    description: str | None = Field(
        None, description="A description of what the function does"
    )
    parameters: dict[str, Any] = Field(
        ...,
        description="The parameters the function accepts, described as a JSON Schema object",
    )

    model_config = ConfigDict(extra="forbid")


class OpenAITool(BaseModel):
    """OpenAI tool definition."""

    type: Literal["function"] = Field("function", description="The type of tool")
    function: OpenAIFunction = Field(..., description="The function definition")

    model_config = ConfigDict(extra="forbid")


class OpenAIToolChoice(BaseModel):
    """OpenAI tool choice specification."""

    type: Literal["function"] = Field("function", description="The type of tool")
    function: dict[str, str] = Field(..., description="The function name to call")

    model_config = ConfigDict(extra="forbid")


# OpenAI Request Models
class OpenAIResponseFormat(BaseModel):
    """OpenAI response format specification."""

    type: Literal["text", "json_object"] = Field(
        "text", description="The type of response format"
    )

    model_config = ConfigDict(extra="forbid")


class OpenAIStreamOptions(BaseModel):
    """OpenAI streaming options."""

    include_usage: bool | None = Field(
        None, description="Whether to include usage information in streaming responses"
    )

    model_config = ConfigDict(extra="forbid")


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request model."""

    model: str = Field(..., description="ID of the model to use")
    messages: list[OpenAIMessage] = Field(
        ...,
        description="A list of messages comprising the conversation so far",
        min_length=1,
    )
    max_tokens: int | None = Field(
        None, description="The maximum number of tokens to generate", ge=1
    )
    temperature: float | None = Field(
        None, description="Sampling temperature between 0 and 2", ge=0.0, le=2.0
    )
    top_p: float | None = Field(
        None, description="Nucleus sampling parameter", ge=0.0, le=1.0
    )
    n: int | None = Field(
        1, description="Number of chat completion choices to generate", ge=1, le=128
    )
    stream: bool | None = Field(
        False, description="Whether to stream back partial progress"
    )
    stream_options: OpenAIStreamOptions | None = Field(
        None, description="Options for streaming response"
    )
    stop: str | list[str] | None = Field(
        None,
        description="Up to 4 sequences where the API will stop generating further tokens",
    )
    presence_penalty: float | None = Field(
        None,
        description="Penalize new tokens based on whether they appear in the text so far",
        ge=-2.0,
        le=2.0,
    )
    frequency_penalty: float | None = Field(
        None,
        description="Penalize new tokens based on their existing frequency in the text so far",
        ge=-2.0,
        le=2.0,
    )
    logit_bias: dict[str, float] | None = Field(
        None,
        description="Modify likelihood of specified tokens appearing in the completion",
    )
    user: str | None = Field(
        None, description="A unique identifier representing your end-user"
    )
    tools: list[OpenAITool] | None = Field(
        None, description="A list of tools the model may call"
    )
    tool_choice: Literal["none", "auto", "required"] | OpenAIToolChoice | None = Field(
        None, description="Controls which (if any) tool is called by the model"
    )
    parallel_tool_calls: bool | None = Field(
        True, description="Whether to enable parallel function calling during tool use"
    )
    response_format: OpenAIResponseFormat | None = Field(
        None, description="An object specifying the format that the model must output"
    )
    seed: int | None = Field(
        None,
        description="This feature is in Beta. If specified, system will make a best effort to sample deterministically",
    )
    logprobs: bool | None = Field(
        None, description="Whether to return log probabilities of the output tokens"
    )
    top_logprobs: int | None = Field(
        None,
        description="An integer between 0 and 20 specifying the number of most likely tokens to return at each token position",
        ge=0,
        le=20,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name - just return as-is like Anthropic endpoint."""
        return v

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[OpenAIMessage]) -> list[OpenAIMessage]:
        """Validate message structure."""
        if not v:
            raise ValueError("At least one message is required")
        return v

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, v: str | list[str] | None) -> str | list[str] | None:
        """Validate stop sequences."""
        if v is not None:
            if isinstance(v, str):
                return v
            elif isinstance(v, list):
                if len(v) > 4:
                    raise ValueError("Maximum 4 stop sequences allowed")
                return v
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: list[OpenAITool] | None) -> list[OpenAITool] | None:
        """Validate tools array."""
        if v is not None and len(v) > 128:
            raise ValueError("Maximum 128 tools allowed")
        return v


# OpenAI Response Models
class OpenAIUsage(BaseModel):
    """OpenAI usage statistics."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(
        ..., description="Number of tokens in the generated completion"
    )
    total_tokens: int = Field(
        ..., description="Total number of tokens used in the request"
    )

    model_config = ConfigDict(extra="forbid")


class OpenAILogprobs(BaseModel):
    """OpenAI log probabilities."""

    tokens: list[str] = Field(..., description="List of tokens")
    token_logprobs: list[float | None] = Field(
        ..., description="Log probabilities for each token"
    )
    top_logprobs: list[dict[str, float]] | None = Field(
        None, description="Top alternative tokens and their log probabilities"
    )

    model_config = ConfigDict(extra="forbid")


class OpenAIToolCall(BaseModel):
    """OpenAI tool call in response."""

    id: str = Field(..., description="The ID of the tool call")
    type: Literal["function"] = Field("function", description="The type of tool call")
    function: dict[str, Any] = Field(..., description="The function that was called")

    model_config = ConfigDict(extra="forbid")


class OpenAIResponseMessage(BaseModel):
    """OpenAI response message model."""

    role: Literal["assistant"] = Field(
        "assistant", description="The role of the message sender"
    )
    content: str | None = Field(None, description="The content of the message")
    tool_calls: list[OpenAIToolCall] | None = Field(
        None, description="The tool calls generated by the model"
    )

    model_config = ConfigDict(extra="forbid")


class OpenAIChoice(BaseModel):
    """OpenAI choice in response."""

    index: int = Field(..., description="The index of the choice")
    message: OpenAIResponseMessage = Field(
        ..., description="The message generated by the model"
    )
    logprobs: OpenAILogprobs | None = Field(
        None, description="Log probability information for the choice"
    )
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = Field(
        ..., description="The reason the model stopped generating tokens"
    )

    model_config = ConfigDict(extra="forbid")


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response model."""

    id: str = Field(..., description="A unique identifier for the chat completion")
    object: Literal["chat.completion"] = Field(
        "chat.completion", description="The object type"
    )
    created: int = Field(
        ..., description="The Unix timestamp of when the chat completion was created"
    )
    model: str = Field(..., description="The model used for the chat completion")
    choices: list[OpenAIChoice] = Field(
        ..., description="A list of chat completion choices"
    )
    usage: OpenAIUsage = Field(
        ..., description="Usage statistics for the completion request"
    )
    system_fingerprint: str | None = Field(
        None, description="This fingerprint represents the backend configuration"
    )

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def create(
        cls,
        model: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: Literal[
            "stop", "length", "tool_calls", "content_filter"
        ] = "stop",
        tool_calls: list[OpenAIToolCall] | None = None,
    ) -> "OpenAIChatCompletionResponse":
        """Create a chat completion response."""
        return cls(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIResponseMessage(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls,
                    ),
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            system_fingerprint=None,
        )


# OpenAI Streaming Response Models
class OpenAIStreamingChoice(BaseModel):
    """OpenAI streaming choice."""

    index: int = Field(..., description="The index of the choice")
    delta: dict[str, Any] = Field(..., description="The delta content")
    logprobs: OpenAILogprobs | None = Field(
        None, description="Log probability information for the choice"
    )
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] | None = (
        Field(None, description="The reason the model stopped generating tokens")
    )

    model_config = ConfigDict(extra="forbid")


class OpenAIStreamingChatCompletionResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response model."""

    id: str = Field(..., description="A unique identifier for the chat completion")
    object: Literal["chat.completion.chunk"] = Field(
        "chat.completion.chunk", description="The object type"
    )
    created: int = Field(
        ..., description="The Unix timestamp of when the chat completion was created"
    )
    model: str = Field(..., description="The model used for the chat completion")
    choices: list[OpenAIStreamingChoice] = Field(
        ..., description="A list of chat completion choices"
    )
    usage: OpenAIUsage | None = Field(
        None,
        description="Usage statistics for the completion request (only in last chunk)",
    )
    system_fingerprint: str | None = Field(
        None, description="This fingerprint represents the backend configuration"
    )

    model_config = ConfigDict(extra="forbid")


# OpenAI Models List Response
class OpenAIModelInfo(BaseModel):
    """OpenAI model information."""

    id: str = Field(..., description="The model identifier")
    object: Literal["model"] = Field("model", description="The object type")
    created: int = Field(
        ..., description="The Unix timestamp of when the model was created"
    )
    owned_by: str = Field(..., description="The organization that owns the model")

    model_config = ConfigDict(extra="forbid")


class OpenAIModelsResponse(BaseModel):
    """OpenAI models list response."""

    object: Literal["list"] = Field("list", description="The object type")
    data: list[OpenAIModelInfo] = Field(..., description="List of model objects")

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def create_default(cls) -> "OpenAIModelsResponse":
        """Create a default models response with Claude model names."""
        models = [
            OpenAIModelInfo(
                id="claude-3-opus-20240229",
                object="model",
                created=1687882411,
                owned_by="anthropic",
            ),
            OpenAIModelInfo(
                id="claude-3-sonnet-20240229",
                object="model",
                created=1687882411,
                owned_by="anthropic",
            ),
            OpenAIModelInfo(
                id="claude-3-haiku-20240307",
                object="model",
                created=1687882411,
                owned_by="anthropic",
            ),
            OpenAIModelInfo(
                id="claude-3-5-sonnet-20241022",
                object="model",
                created=1712361441,
                owned_by="anthropic",
            ),
            OpenAIModelInfo(
                id="claude-3-5-haiku-20241022",
                object="model",
                created=1721172741,
                owned_by="anthropic",
            ),
        ]
        return cls(object="list", data=models)


# OpenAI Error Response Models
class OpenAIErrorDetail(BaseModel):
    """OpenAI error detail."""

    message: str = Field(..., description="A human-readable error message")
    type: str = Field(..., description="The error type")
    param: str | None = Field(None, description="The parameter that caused the error")
    code: str | None = Field(None, description="The error code")

    model_config = ConfigDict(extra="forbid")


class OpenAIErrorResponse(BaseModel):
    """OpenAI error response."""

    error: OpenAIErrorDetail = Field(..., description="The error details")

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def create(
        cls,
        message: str,
        error_type: str,
        param: str | None = None,
        code: str | None = None,
    ) -> "OpenAIErrorResponse":
        """Create an error response."""
        return cls(
            error=OpenAIErrorDetail(
                message=message,
                type=error_type,
                param=param,
                code=code,
            )
        )


# Export all models
__all__ = [
    # Message models
    "OpenAIMessage",
    "OpenAIMessageContent",
    # Tool models
    "OpenAIFunction",
    "OpenAITool",
    "OpenAIToolChoice",
    # Request models
    "OpenAIResponseFormat",
    "OpenAIStreamOptions",
    "OpenAIChatCompletionRequest",
    # Response models
    "OpenAIUsage",
    "OpenAILogprobs",
    "OpenAIToolCall",
    "OpenAIResponseMessage",
    "OpenAIChoice",
    "OpenAIChatCompletionResponse",
    # Streaming models
    "OpenAIStreamingChoice",
    "OpenAIStreamingChatCompletionResponse",
    # Models list
    "OpenAIModelInfo",
    "OpenAIModelsResponse",
    # Error models
    "OpenAIErrorDetail",
    "OpenAIErrorResponse",
]
