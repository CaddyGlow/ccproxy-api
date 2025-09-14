"""
Pydantic V2 models for OpenAI API endpoints based on the provided reference.

This module contains data structures for:
- /v1/chat/completions (including streaming)
- /v1/embeddings
- /v1/models
- /v1/responses (including streaming)
- Common Error structures

The models are defined using modern Python 3.11 type hints and Pydantic V2 best practices.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel, field_validator


# ==============================================================================
# Error Models
# ==============================================================================


class ErrorDetail(BaseModel):
    """
    Detailed information about an API error.
    """

    code: str | None = Field(None, description="The error code.")
    message: str = Field(..., description="The error message.")
    param: str | None = Field(None, description="The parameter that caused the error.")
    type: str | None = Field(None, description="The type of error.")


class ErrorResponse(BaseModel):
    """
    The structure of an error response from the OpenAI API.
    """

    error: ErrorDetail = Field(..., description="Container for the error details.")


# ==============================================================================
# Models Endpoint (/v1/models)
# ==============================================================================


class Model(BaseModel):
    """
    Represents a model available in the API.
    """

    id: str = Field(..., description="The model identifier.")
    created: int = Field(
        ..., description="The Unix timestamp of when the model was created."
    )
    object: Literal["model"] = Field(
        ..., description="The object type, always 'model'."
    )
    owned_by: str = Field(..., description="The organization that owns the model.")


class ModelList(BaseModel):
    """
    A list of available models.
    """

    object: Literal["list"] = Field(..., description="The object type, always 'list'.")
    data: list[Model] = Field(..., description="A list of model objects.")


# ==============================================================================
# Embeddings Endpoint (/v1/embeddings)
# ==============================================================================


class EmbeddingRequest(BaseModel):
    """
    Request body for creating an embedding.
    """

    input: str | list[str] | list[int] | list[list[int]] = Field(
        ..., description="Input text to embed, encoded as a string or array of tokens."
    )
    model: str = Field(..., description="ID of the model to use for embedding.")
    encoding_format: Literal["float", "base64"] | None = Field(
        "float", description="The format to return the embeddings in."
    )
    dimensions: int | None = Field(
        None,
        description="The number of dimensions the resulting output embeddings should have.",
    )
    user: str | None = Field(
        None, description="A unique identifier representing your end-user."
    )


class EmbeddingData(BaseModel):
    """
    Represents a single embedding vector.
    """

    object: Literal["embedding"] = Field(
        ..., description="The object type, always 'embedding'."
    )
    embedding: list[float] = Field(..., description="The embedding vector.")
    index: int = Field(..., description="The index of the embedding in the list.")


class EmbeddingUsage(BaseModel):
    """
    Token usage statistics for an embedding request.
    """

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt.")
    total_tokens: int = Field(..., description="Total number of tokens used.")


class EmbeddingResponse(BaseModel):
    """
    Response object for an embedding request.
    """

    object: Literal["list"] = Field(..., description="The object type, always 'list'.")
    data: list[EmbeddingData] = Field(..., description="List of embedding objects.")
    model: str = Field(..., description="The model used for the embedding.")
    usage: EmbeddingUsage = Field(..., description="Token usage for the request.")


# ==============================================================================
# Chat Completions Endpoint (/v1/chat/completions)
# ==============================================================================

# --- Request Models ---


class ResponseFormat(BaseModel):
    """
    An object specifying the format that the model must output.
    """

    type: Literal["text", "json_object", "json_schema"] = Field(
        "text", description="The type of response format."
    )
    json_schema: dict[str, Any] | None = None


class FunctionDefinition(BaseModel):
    """
    The definition of a function that the model can call.
    """

    name: str = Field(..., description="The name of the function to be called.")
    description: str | None = Field(
        None, description="A description of what the function does."
    )
    parameters: dict[str, Any] = Field(
        ...,
        description="The parameters the functions accepts, described as a JSON Schema object.",
    )


class Tool(BaseModel):
    """
    A tool the model may call.
    """

    type: Literal["function"] = Field(
        ..., description="The type of the tool, currently only 'function' is supported."
    )
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall


class ChatMessage(BaseModel):
    """
    A message within a chat conversation.
    """

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None
    name: str | None = Field(
        None,
        description="The name of the author of this message. May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters.",
    )
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool role messages


class ChatCompletionRequest(BaseModel):
    """
    Request body for creating a chat completion.
    """

    messages: list[ChatMessage]
    model: str
    audio: dict[str, Any] | None = None
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = Field(None)
    logprobs: bool | None = Field(None)
    top_logprobs: int | None = Field(None, ge=0, le=20)
    max_tokens: int | None = Field(None, deprecated=True)
    max_completion_tokens: int | None = None
    n: int | None = Field(1)
    parallel_tool_calls: bool | None = None
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    reasoning_effort: str | None = None
    response_format: ResponseFormat | None = Field(None)
    seed: int | None = Field(None)
    stop: str | list[str] | None = Field(None)
    stream: bool | None = Field(None)
    stream_options: dict[str, Any] | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    tools: list[Tool] | None = Field(None)
    tool_choice: Literal["none", "auto", "required"] | dict[str, Any] | None = Field(
        None
    )
    user: str | None = Field(None)
    modalities: list[str] | None = None
    prediction: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    safety_identifier: str | None = None
    service_tier: str | None = None
    store: bool | None = None
    verbosity: str | None = None
    web_search_options: dict[str, Any] | None = None


# --- Response Models (Non-streaming) ---


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: FunctionCall


class ResponseMessage(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    role: Literal["assistant"]
    refusal: str | None = None
    annotations: list[Any] | None = None


class Choice(BaseModel):
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"]
    index: int
    message: ResponseMessage
    logprobs: dict[str, Any] | None = None


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list[Choice]
    created: int
    model: str
    system_fingerprint: str | None = None
    object: Literal["chat.completion"]
    usage: CompletionUsage
    service_tier: str | None = None


# --- Response Models (Streaming) ---


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamingChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None
    logprobs: dict[str, Any] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    system_fingerprint: str | None = None
    choices: list[StreamingChoice]
    usage: CompletionUsage | None = Field(
        None, description="Usage stats, present only in the final chunk if requested."
    )


# ==============================================================================
# Responses Endpoint (/v1/responses)
# ==============================================================================


# --- Request Models ---
class StreamOptions(BaseModel):
    include_usage: bool | None = Field(
        None,
        description="If set, an additional chunk will be streamed before the final completion chunk with usage statistics.",
    )


class ToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any]


class FunctionTool(BaseModel):
    type: Literal["function"]
    function: ToolFunction


# Valid include values for Responses API
VALID_INCLUDE_VALUES = [
    "web_search_call.action.sources",
    "code_interpreter_call.outputs",
    "computer_call_output.output.image_url",
    "file_search_call.results",
    "message.input_image.image_url",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
]


class ResponseRequest(BaseModel):
    model: str | None = None
    input: str | list[Any]
    background: bool | None = Field(
        None, description="Whether to run the model response in the background"
    )
    conversation: str | dict[str, Any] | None = Field(
        None, description="The conversation that this response belongs to"
    )
    include: list[str] | None = Field(
        None,
        description="Specify additional output data to include in the model response",
    )

    @field_validator("include")
    @classmethod
    def validate_include(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for item in v:
                if item not in VALID_INCLUDE_VALUES:
                    raise ValueError(
                        f"Invalid include value: {item}. Valid values are: {VALID_INCLUDE_VALUES}"
                    )
        return v

    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: dict[str, str] | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    prompt: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    reasoning: dict[str, Any] | None = None
    safety_identifier: str | None = None
    service_tier: str | None = None
    store: bool | None = None
    stream: bool | None = Field(None)
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    text: dict[str, Any] | None = None
    tools: list[Any] | None = None
    tool_choice: str | dict[str, Any] | None = None
    top_logprobs: int | None = None
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    truncation: str | None = None
    user: str | None = None


# --- Response Models (Non-streaming) ---
class OutputTextContent(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: list[Any] = Field(default_factory=list)


class MessageOutput(BaseModel):
    type: Literal["message"]
    id: str
    status: str
    role: Literal["assistant", "user"]
    content: list[OutputTextContent | dict[str, Any]]  # To handle various content types


class InputTokensDetails(BaseModel):
    cached_tokens: int


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int


class ResponseUsage(BaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


class IncompleteDetails(BaseModel):
    reason: str


class Reasoning(BaseModel):
    effort: Any | None = None
    summary: Any | None = None


class ResponseObject(BaseModel):
    id: str
    object: Literal["response"]
    created_at: int
    status: str
    model: str
    output: list[MessageOutput]
    parallel_tool_calls: bool
    usage: ResponseUsage | None = None
    error: ErrorDetail | None = None
    incomplete_details: IncompleteDetails | None = None
    metadata: dict[str, str] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    reasoning: Reasoning | None = None
    store: bool | None = None
    temperature: float | None = None
    text: dict[str, Any] | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: list[Any] | None = None
    top_p: float | None = None
    truncation: str | None = None
    user: str | None = None


# --- Response Models (Streaming) ---
class BaseStreamEvent(BaseModel):
    sequence_number: int


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"]
    response: ResponseObject


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"]
    response: ResponseObject


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"]
    response: ResponseObject


class ResponseFailedEvent(BaseStreamEvent):
    type: Literal["response.failed"]
    response: ResponseObject


class ResponseIncompleteEvent(BaseStreamEvent):
    type: Literal["response.incomplete"]
    response: ResponseObject


class OutputItem(BaseModel):
    id: str
    status: str
    type: str
    role: str
    content: list[Any]


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"]
    output_index: int
    item: OutputItem


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"]
    output_index: int
    item: OutputItem


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    annotations: list[Any] | None = None


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPart


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPart


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str
    logprobs: list[Any] | None = None


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"]
    item_id: str
    output_index: int
    content_index: int
    text: str
    logprobs: list[Any] | None = None


class ResponseRefusalDeltaEvent(BaseStreamEvent):
    type: Literal["response.refusal.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseRefusalDoneEvent(BaseStreamEvent):
    type: Literal["response.refusal.done"]
    item_id: str
    output_index: int
    content_index: int
    refusal: str


class ResponseFunctionCallArgumentsDeltaEvent(BaseStreamEvent):
    type: Literal["response.function_call_arguments.delta"]
    item_id: str
    output_index: int
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(BaseStreamEvent):
    type: Literal["response.function_call_arguments.done"]
    item_id: str
    output_index: int
    arguments: str


class ReasoningSummaryPart(BaseModel):
    type: str
    text: str


class ReasoningSummaryPartAddedEvent(BaseStreamEvent):
    type: Literal["response.reasoning_summary_part.added"]
    item_id: str
    output_index: int
    summary_index: int
    part: ReasoningSummaryPart


class ReasoningSummaryPartDoneEvent(BaseStreamEvent):
    type: Literal["response.reasoning_summary_part.done"]
    item_id: str
    output_index: int
    summary_index: int
    part: ReasoningSummaryPart


class ReasoningSummaryTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.reasoning_summary_text.delta"]
    item_id: str
    output_index: int
    summary_index: int
    delta: str


class ReasoningSummaryTextDoneEvent(BaseStreamEvent):
    type: Literal["response.reasoning_summary_text.done"]
    item_id: str
    output_index: int
    summary_index: int
    text: str


class ReasoningTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.reasoning_text.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ReasoningTextDoneEvent(BaseStreamEvent):
    type: Literal["response.reasoning_text.done"]
    item_id: str
    output_index: int
    content_index: int
    text: str


class FileSearchCallEvent(BaseStreamEvent):
    output_index: int
    item_id: str


class FileSearchCallInProgressEvent(FileSearchCallEvent):
    type: Literal["response.file_search_call.in_progress"]


class FileSearchCallSearchingEvent(FileSearchCallEvent):
    type: Literal["response.file_search_call.searching"]


class FileSearchCallCompletedEvent(FileSearchCallEvent):
    type: Literal["response.file_search_call.completed"]


class WebSearchCallEvent(BaseStreamEvent):
    output_index: int
    item_id: str


class WebSearchCallInProgressEvent(WebSearchCallEvent):
    type: Literal["response.web_search_call.in_progress"]


class WebSearchCallSearchingEvent(WebSearchCallEvent):
    type: Literal["response.web_search_call.searching"]


class WebSearchCallCompletedEvent(WebSearchCallEvent):
    type: Literal["response.web_search_call.completed"]


class CodeInterpreterCallEvent(BaseStreamEvent):
    output_index: int
    item_id: str


class CodeInterpreterCallInProgressEvent(CodeInterpreterCallEvent):
    type: Literal["response.code_interpreter_call.in_progress"]


class CodeInterpreterCallInterpretingEvent(CodeInterpreterCallEvent):
    type: Literal["response.code_interpreter_call.interpreting"]


class CodeInterpreterCallCompletedEvent(CodeInterpreterCallEvent):
    type: Literal["response.code_interpreter_call.completed"]


class CodeInterpreterCallCodeDeltaEvent(CodeInterpreterCallEvent):
    type: Literal["response.code_interpreter_call_code.delta"]
    delta: str


class CodeInterpreterCallCodeDoneEvent(CodeInterpreterCallEvent):
    type: Literal["response.code_interpreter_call_code.done"]
    code: str


class ErrorEvent(BaseModel):  # Does not inherit from BaseStreamEvent per docs
    type: Literal["error"]
    error: ErrorDetail


AnyStreamEvent = RootModel[
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseOutputTextDeltaEvent
    | ResponseOutputTextDoneEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ReasoningSummaryPartAddedEvent
    | ReasoningSummaryPartDoneEvent
    | ReasoningSummaryTextDeltaEvent
    | ReasoningSummaryTextDoneEvent
    | ReasoningTextDeltaEvent
    | ReasoningTextDoneEvent
    | FileSearchCallInProgressEvent
    | FileSearchCallSearchingEvent
    | FileSearchCallCompletedEvent
    | WebSearchCallInProgressEvent
    | WebSearchCallSearchingEvent
    | WebSearchCallCompletedEvent
    | CodeInterpreterCallInProgressEvent
    | CodeInterpreterCallInterpretingEvent
    | CodeInterpreterCallCompletedEvent
    | CodeInterpreterCallCodeDeltaEvent
    | CodeInterpreterCallCodeDoneEvent
    | ErrorEvent
]
