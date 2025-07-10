"""OpenAI adapter module for API format conversion.

This module provides the OpenAI adapter implementation for converting
between OpenAI and Anthropic API formats.
"""

from .adapter import OpenAIAdapter, map_openai_model_to_claude
from .models import (
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIMessageContent,
    OpenAIResponseMessage,
    OpenAIStreamingChatCompletionResponse,
    OpenAIToolCall,
    OpenAIUsage,
    format_openai_tool_call,
    generate_openai_response_id,
    generate_openai_system_fingerprint,
)
from .streaming import OpenAISSEFormatter, OpenAIStreamProcessor


__all__ = [
    # Adapter
    "OpenAIAdapter",
    "map_openai_model_to_claude",
    # Models
    "OpenAIMessage",
    "OpenAIMessageContent",
    "OpenAIResponseMessage",
    "OpenAIChoice",
    "OpenAIChatCompletionResponse",
    "OpenAIStreamingChatCompletionResponse",
    "OpenAIToolCall",
    "OpenAIUsage",
    "format_openai_tool_call",
    "generate_openai_response_id",
    "generate_openai_system_fingerprint",
    # Streaming
    "OpenAISSEFormatter",
    "OpenAIStreamProcessor",
]
