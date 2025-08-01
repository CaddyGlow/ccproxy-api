"""Pydantic models for Claude Proxy API Server."""

from .claude_sdk import (
    AssistantMessage,
    ContentBlock,
    ExtendedContentBlock,
    ResultMessage,
    ResultMessageBlock,
    SDKContentBlock,
    SDKMessageMode,
    TextBlock,
    ToolResultBlock,
    ToolResultSDKBlock,
    ToolUseBlock,
    ToolUseSDKBlock,
    UserMessage,
    convert_sdk_result_message,
    convert_sdk_system_message,
    convert_sdk_text_block,
    convert_sdk_tool_result_block,
    convert_sdk_tool_use_block,
    to_sdk_variant,
)
from .messages import (
    MessageContentBlock,
    MessageCreateParams,
    MessageResponse,
    MetadataParams,
    SystemMessage,
    ThinkingConfig,
    ToolChoiceParams,
)
from .requests import (
    ImageContent,
    Message,
    MessageContent,
    TextContent,
    ToolDefinition,
    Usage,
)
from .responses import (
    APIError,
    AuthenticationError,
    ChatCompletionResponse,
    Choice,
    ErrorResponse,
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
    OverloadedError,
    RateLimitError,
    ResponseContent,
    StreamingChatCompletionResponse,
    StreamingChoice,
    TextResponse,
    ToolCall,
    ToolUse,
)
from .types import (
    ContentBlockType,
    ErrorType,
    ImageSourceType,
    MessageRole,
    ModalityType,
    OpenAIFinishReason,
    PermissionBehavior,
    ResponseFormatType,
    ServiceTier,
    StopReason,
    StreamEventType,
    ToolChoiceType,
    ToolType,
)


__all__ = [
    # Type aliases
    "ContentBlockType",
    "ErrorType",
    "ImageSourceType",
    "MessageRole",
    "ModalityType",
    "OpenAIFinishReason",
    "PermissionBehavior",
    "ResponseFormatType",
    "ServiceTier",
    "StopReason",
    "StreamEventType",
    "ToolChoiceType",
    "ToolType",
    # Claude SDK models
    "AssistantMessage",
    "ContentBlock",
    "ExtendedContentBlock",
    "ResultMessage",
    "ResultMessageBlock",
    "SDKContentBlock",
    "SDKMessageMode",
    "TextBlock",
    "ToolResultBlock",
    "ToolResultSDKBlock",
    "ToolUseBlock",
    "ToolUseSDKBlock",
    "UserMessage",
    "convert_sdk_result_message",
    "convert_sdk_system_message",
    "convert_sdk_text_block",
    "convert_sdk_tool_result_block",
    "convert_sdk_tool_use_block",
    "to_sdk_variant",
    # Message models
    "MessageContentBlock",
    "MessageCreateParams",
    "MessageResponse",
    "MetadataParams",
    "SystemMessage",
    "ThinkingConfig",
    "ToolChoiceParams",
    # Request models
    "ImageContent",
    "Message",
    "MessageContent",
    "TextContent",
    "ToolDefinition",
    "Usage",
    # Response models
    "APIError",
    "AuthenticationError",
    "ChatCompletionResponse",
    "Choice",
    "ErrorResponse",
    "InternalServerError",
    "InvalidRequestError",
    "NotFoundError",
    "OverloadedError",
    "RateLimitError",
    "ResponseContent",
    "StreamingChatCompletionResponse",
    "StreamingChoice",
    "TextResponse",
    "ToolCall",
    "ToolUse",
    # OpenAI-compatible models
    "OpenAIChatCompletionRequest",
    "OpenAIChatCompletionResponse",
    "OpenAIChoice",
    "OpenAIErrorDetail",
    "OpenAIErrorResponse",
    "OpenAIFunction",
    "OpenAILogprobs",
    "OpenAIMessage",
    "OpenAIMessageContent",
    "OpenAIModelInfo",
    "OpenAIModelsResponse",
    "OpenAIResponseFormat",
    "OpenAIResponseMessage",
    "OpenAIStreamingChatCompletionResponse",
    "OpenAIStreamingChoice",
    "OpenAIStreamOptions",
    "OpenAITool",
    "OpenAIToolCall",
    "OpenAIToolChoice",
    "OpenAIUsage",
]
