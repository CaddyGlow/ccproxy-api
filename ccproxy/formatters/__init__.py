"""Formatters for data transformation between API formats.

This module contains pure data transformation utilities for converting between
different API formats (OpenAI, Anthropic) and handling streaming responses.

Key components:
- translator: Convert between OpenAI and Anthropic API formats
- stream_transformer: Unified streaming transformation framework
- openai_streaming_formatter: OpenAI-specific streaming formatting
- anthropic_streaming: Anthropic-specific streaming formatting
"""

from .anthropic_streaming import (
    StreamingFormatter,
    stream_anthropic_message_response,
    stream_claude_response,
)
from .openai_streaming_formatter import (
    OpenAIStreamingFormatter,
    stream_claude_response_openai,
    stream_claude_response_openai_simple,
)
from .stream_transformer import (
    AnthropicStreamTransformer,
    OpenAIStreamTransformer,
    StreamingConfig,
)
from .translator import OpenAITranslator


__all__ = [
    "OpenAITranslator",
    "OpenAIStreamingFormatter",
    "stream_claude_response_openai",
    "stream_claude_response_openai_simple",
    "StreamingFormatter",
    "stream_claude_response",
    "stream_anthropic_message_response",
    "OpenAIStreamTransformer",
    "AnthropicStreamTransformer",
    "StreamingConfig",
]
