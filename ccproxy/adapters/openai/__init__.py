"""OpenAI adapter module for API format conversion.

This module provides the OpenAI adapter implementation for converting
between OpenAI and Anthropic API formats.

New organized structure:
- adapters/: Specialized adapters for different conversions
- models/: Organized models by API format
"""

# Legacy imports for backward compatibility
from .adapter import OpenAIAdapter

# Import new specialized adapters
from .adapters import (
    ChatCompletionsAdapter,
    ChatToResponsesAdapter,
    ResponsesAdapter,
)
from .anthropic_response_adapter import AnthropicResponseAPIAdapter

# Import from new organized structure
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
    # Legacy adapters
    "OpenAIAdapter",
    "AnthropicResponseAPIAdapter",
    # New specialized adapters
    "ChatCompletionsAdapter",
    "ResponsesAdapter",
    "ChatToResponsesAdapter",
    # Models (backward compatibility)
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
