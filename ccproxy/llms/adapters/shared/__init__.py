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
__all__ = [
    "ANTHROPIC_TO_OPENAI_ERROR_TYPE",
    "ANTHROPIC_TO_OPENAI_FINISH_REASON",
    "DEFAULT_MAX_TOKENS",
    "OPENAI_TO_ANTHROPIC_ERROR_TYPE",
    "OPENAI_TO_ANTHROPIC_STOP_REASON",
    "convert_anthropic_error_to_openai",
    "convert_openai_error_to_anthropic",
    "normalize_openai_error",
]
