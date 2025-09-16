"""Shared utilities for LLM format adapters."""

from .constants import (
    ANTHROPIC_TO_OPENAI_ERROR_TYPE,
    ANTHROPIC_TO_OPENAI_FINISH_REASON,
    DEFAULT_MAX_TOKENS,
    OPENAI_TO_ANTHROPIC_ERROR_TYPE,
    OPENAI_TO_ANTHROPIC_STOP_REASON,
)
from .errors import (
    convert_anthropic_error_to_openai,
    convert_openai_error_to_anthropic,
    normalize_openai_error,
)
from .usage import (
    convert_anthropic_usage_to_openai_completion_usage,
    convert_anthropic_usage_to_openai_response_usage,
    convert_openai_completion_usage_to_anthropic_usage,
    convert_openai_response_usage_to_anthropic_usage,
    safe_extract_usage_tokens,
)

__all__ = [
    "ANTHROPIC_TO_OPENAI_ERROR_TYPE",
    "ANTHROPIC_TO_OPENAI_FINISH_REASON",
    "DEFAULT_MAX_TOKENS",
    "OPENAI_TO_ANTHROPIC_ERROR_TYPE",
    "OPENAI_TO_ANTHROPIC_STOP_REASON",
    "convert_anthropic_error_to_openai",
    "convert_openai_error_to_anthropic",
    "normalize_openai_error",
    "convert_anthropic_usage_to_openai_completion_usage",
    "convert_anthropic_usage_to_openai_response_usage",
    "convert_openai_completion_usage_to_anthropic_usage",
    "convert_openai_response_usage_to_anthropic_usage",
    "safe_extract_usage_tokens",
]
