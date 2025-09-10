"""Adapter modules for API format conversion.

This package provides adapters for different API formats:
- openai: OpenAI-specific adapters (Chat Completions, Response API)
- anthropic: Anthropic-specific adapters (placeholder)
- base: Base adapter interfaces
"""

from .base import APIAdapter, BaseAPIAdapter

# Import the new organized adapters
from .openai.adapters import (
    ChatCompletionsAdapter,
    ChatToResponsesAdapter,
    ResponsesAdapter,
)


# Legacy import - will be deprecated
try:
    from .openai import OpenAIAdapter
except ImportError:
    # Fallback if old adapter doesn't exist
    OpenAIAdapter = None  # type: ignore[misc, assignment]


__all__ = [
    "APIAdapter",
    "BaseAPIAdapter",
    # New organized adapters
    "ChatCompletionsAdapter",
    "ResponsesAdapter",
    "ChatToResponsesAdapter",
    # Legacy (deprecated)
    "OpenAIAdapter",
]
