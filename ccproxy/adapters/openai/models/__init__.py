"""OpenAI models package.

This package contains all OpenAI-related models organized by API format:
- chat_completions: OpenAI Chat Completions API models
- responses: OpenAI Response API models (used by Codex)
- common: Shared models and utilities
"""

# Re-export main models for backward compatibility
from .base import OpenAIBaseModel
from .chat_completions import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIErrorDetail,
    OpenAIErrorResponse,
    OpenAIFunction,
    OpenAIMessage,
    OpenAIMessageContent,
    OpenAIModelInfo,
    OpenAIModelsResponse,
    OpenAIResponseFormat,
    OpenAIResponseMessage,
    OpenAIStreamingChatCompletionResponse,
    OpenAIStreamingChoice,
    OpenAIStreamingDelta,
    OpenAIStreamOptions,
    OpenAITool,
    OpenAIToolChoice,
)
from .common import (
    OpenAIFunctionCall,
    OpenAILogprobs,
    OpenAIToolCall,
    OpenAIUsage,
    ResponseUsage,
    format_openai_tool_call,
    generate_openai_response_id,
    generate_openai_system_fingerprint,
)
from .responses import (
    FunctionCallDelta,
    ResponseCompleted,
    ResponseData,
    ResponseFunction,
    ResponseMessage,
    ResponseMessageContent,
    ResponseOutput,
    ResponseReasoning,
    ResponseReasoningContent,
    ResponseRequest,
    ResponseTool,
    ResponseToolCall,
    ResponseToolChoice,
    ResponseToolFunction,
    StreamingChoice,
    StreamingChunk,
    StreamingDelta,
    StreamingEvent,
    ToolCallState,
)


__all__ = [
    # Base model
    "OpenAIBaseModel",
    # Chat Completions API
    "OpenAIMessageContent",
    "OpenAIMessage",
    "OpenAIFunction",
    "OpenAITool",
    "OpenAIToolChoice",
    "OpenAIResponseFormat",
    "OpenAIStreamOptions",
    "OpenAIChatCompletionRequest",
    "OpenAIResponseMessage",
    "OpenAIChoice",
    "OpenAIChatCompletionResponse",
    "OpenAIStreamingDelta",
    "OpenAIStreamingChoice",
    "OpenAIStreamingChatCompletionResponse",
    "OpenAIModelInfo",
    "OpenAIModelsResponse",
    "OpenAIErrorDetail",
    "OpenAIErrorResponse",
    # Response API
    "ResponseFunction",
    "ResponseToolCall",
    "ResponseMessageContent",
    "ResponseMessage",
    "ResponseToolFunction",
    "ResponseTool",
    "ResponseToolChoice",
    "ResponseReasoning",
    "ResponseRequest",
    "ResponseOutput",
    "ResponseReasoningContent",
    "ResponseData",
    "ResponseCompleted",
    "StreamingDelta",
    "StreamingChoice",
    "StreamingChunk",
    "StreamingEvent",
    "FunctionCallDelta",
    "ToolCallState",
    # Common/Shared
    "OpenAIUsage",
    "OpenAILogprobs",
    "OpenAIFunctionCall",
    "OpenAIToolCall",
    "ResponseUsage",
    "generate_openai_response_id",
    "generate_openai_system_fingerprint",
    "format_openai_tool_call",
]
