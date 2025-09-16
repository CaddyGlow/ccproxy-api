"""Compatibility layer for adapter mapping utilities.

This shim was previously used to re-export usage converters that have now been
inlined into their respective adapter helpers. It now only re-exports constants
and error conversion utilities that remain shared.
"""

from __future__ import annotations

from ccproxy.llms.adapters.shared import (
    ANTHROPIC_TO_OPENAI_ERROR_TYPE,
    ANTHROPIC_TO_OPENAI_FINISH_REASON,
    DEFAULT_MAX_TOKENS,
    OPENAI_TO_ANTHROPIC_ERROR_TYPE,
    OPENAI_TO_ANTHROPIC_STOP_REASON,
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
