"""Pydantic models for Claude Proxy API Server.

This package now re-exports Anthropic models from ccproxy.adapters.anthropic.models
for backward compatibility, while keeping provider-agnostic models here.
"""

from ccproxy.adapters.anthropic.models import (
    APIError,
    AuthenticationError,
    ChatCompletionResponse,
    Choice,
    ContentBlockType,
    ErrorResponse,
    ErrorType,
    ImageContent,
    ImageSourceType,
    InternalServerError,
    InvalidRequestError,
    Message,
    MessageContent,
    MessageCreateParams,
    MessageResponse,
    MessageRole,
    MetadataParams,
    ModalityType,
    NotFoundError,
    OpenAIFinishReason,
    OverloadedError,
    PermissionBehavior,
    RateLimitError,
    ResponseContent,
    ResponseFormatType,
    ServiceTier,
    StopReason,
    StreamEventType,
    StreamingChatCompletionResponse,
    StreamingChoice,
    SystemMessage,
    TextContent,
    TextResponse,
    ThinkingConfig,
    ToolCall,
    ToolChoiceParams,
    ToolChoiceType,
    ToolDefinition,
    ToolType,
    ToolUse,
    Usage,
)
from ccproxy.adapters.anthropic.models import (
    CCProxyContentBlock as MessageContentBlock,
)

from .provider import ProviderConfig


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
    # Provider models
    "ProviderConfig",
]
