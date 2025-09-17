"""Pydantic models for Claude Proxy API Server.

This package now re-exports Anthropic models from ccproxy.llms.models.anthropic
for backward compatibility, while keeping provider-agnostic models here.
"""

from ccproxy.llms.models import anthropic as anthropic_models


# Re-export all models for backward compatibility
# Note: Some models may not exist in the new structure and are commented out
APIError = anthropic_models.APIError
AuthenticationError = anthropic_models.AuthenticationError
# ChatCompletionResponse = anthropic_models.ChatCompletionResponse  # Check if exists
# Choice = anthropic_models.Choice  # Check if exists
# ContentBlockType = anthropic_models.ContentBlockType  # Use string types instead
ErrorResponse = anthropic_models.ErrorResponse
# ErrorType = anthropic_models.ErrorType  # Use string types instead
ImageContent = anthropic_models.ImageBlock  # Mapped to ImageBlock
ImageSourceType = "url"  # Use string literal type
# InternalServerError = anthropic_models.InternalServerError  # Use APIError
InvalidRequestError = anthropic_models.InvalidRequestError
Message = anthropic_models.Message
# MessageContent = anthropic_models.MessageContent  # Use RequestContentBlock
MessageCreateParams = anthropic_models.CreateMessageRequest
MessageResponse = anthropic_models.MessageResponse
MessageRole = "user"  # Use string literal types: "user", "assistant"
# MetadataParams = anthropic_models.MetadataParams  # Check if exists
# ModalityType = anthropic_models.ModalityType  # Use string types
NotFoundError = anthropic_models.NotFoundError
# OpenAIFinishReason = anthropic_models.OpenAIFinishReason  # OpenAI specific
OverloadedError = anthropic_models.OverloadedError
# PermissionBehavior = anthropic_models.PermissionBehavior  # Use string types
RateLimitError = anthropic_models.RateLimitError
ResponseContent = anthropic_models.ResponseContentBlock
# ResponseFormatType = anthropic_models.ResponseFormatType  # Use string types
# ServiceTier = anthropic_models.ServiceTier  # Use string types
StopReason = "end_turn"  # Use string literal types
# StreamEventType = anthropic_models.StreamEventType  # Use string types
# StreamingChatCompletionResponse = anthropic_models.StreamingChatCompletionResponse  # Check if exists
# StreamingChoice = anthropic_models.StreamingChoice  # Check if exists
# SystemMessage = anthropic_models.SystemMessage  # Use string content
TextContent = anthropic_models.TextBlock
# TextResponse = anthropic_models.TextResponse  # Check if exists
# ThinkingConfig = anthropic_models.ThinkingConfig  # Check if exists
# ToolCall = anthropic_models.ToolCall  # Use ToolUseBlock
# ToolChoiceParams = anthropic_models.ToolChoiceParams  # Use ToolChoice
# ToolChoiceType = anthropic_models.ToolChoiceType  # Use string types
ToolDefinition = anthropic_models.Tool
# ToolType = anthropic_models.ToolType  # Use string types
ToolUse = anthropic_models.ToolUseBlock
Usage = anthropic_models.Usage

# Map MessageContentBlock to the appropriate content block type
MessageContentBlock = anthropic_models.RequestContentBlock

from .provider import ProviderConfig


__all__ = [
    # Type aliases (simplified to string literals)
    "ImageSourceType",
    "MessageRole",
    "StopReason",
    # Message models
    "MessageContentBlock",
    "MessageCreateParams",
    "MessageResponse",
    # Request models
    "ImageContent",
    "Message",
    "TextContent",
    "ToolDefinition",
    "Usage",
    # Response models
    "APIError",
    "AuthenticationError",
    "ErrorResponse",
    "InvalidRequestError",
    "NotFoundError",
    "OverloadedError",
    "RateLimitError",
    "ResponseContent",
    "ToolUse",
    # Provider models
    "ProviderConfig",
]
