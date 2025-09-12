"""Anthropic models package.

This package contains Anthropic-specific models:
- base: Common base model with extra="allow"
- types: Type aliases for Anthropic models
- requests: Request models compatible with Anthropic's API format
- responses: Response models compatible with Anthropic's API format
- messages: Anthropic Messages API models
"""

from .base import AnthropicBaseModel
from .messages import (
    CCProxyContentBlock,
    MessageContentBlock,
    MessageCreateParams,
    MessageResponse,
    MetadataParams,
    SystemMessage,
    TextContentBlock,
    ThinkingConfig,
    ThinkingContentBlock,
    ToolChoiceParams,
    ToolUseContentBlock,
)
from .requests import (
    FunctionDefinition,
    ImageContent,
    ImageSource,
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
    StreamingTokenMetrics,
    ToolChoiceType,
    ToolType,
    UsageData,
)


__all__ = [
    # Base model
    "AnthropicBaseModel",
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
    "StreamingTokenMetrics",
    "UsageData",
    # Message models
    "CCProxyContentBlock",
    "MessageContentBlock",
    "MessageCreateParams",
    "MessageResponse",
    "MetadataParams",
    "SystemMessage",
    "TextContentBlock",
    "ThinkingConfig",
    "ThinkingContentBlock",
    "ToolChoiceParams",
    "ToolUseContentBlock",
    # Request models
    "FunctionDefinition",
    "ImageContent",
    "ImageSource",
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
]
