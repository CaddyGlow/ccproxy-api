"""Compatibility layer for adapter mapping utilities.

This shim was previously used to re-export usage converters that have now been
inlined into their respective adapter helpers. It now only re-exports constants
and error conversion utilities that remain shared.
"""

from __future__ import annotations

from ccproxy.llms.formatters.anthropic_to_openai import (
    convert__anthropic_to_openai__error,
)
from ccproxy.llms.formatters.constants import (
    ANTHROPIC_TO_OPENAI_ERROR_TYPE,
    ANTHROPIC_TO_OPENAI_FINISH_REASON,
    DEFAULT_MAX_TOKENS,
    OPENAI_TO_ANTHROPIC_ERROR_TYPE,
    OPENAI_TO_ANTHROPIC_STOP_REASON,
)
from ccproxy.llms.formatters.openai_to_anthropic import (
    convert__openai_to_anthropic__error,
)


__all__ = [
    "ANTHROPIC_TO_OPENAI_ERROR_TYPE",
    "ANTHROPIC_TO_OPENAI_FINISH_REASON",
    "DEFAULT_MAX_TOKENS",
    "OPENAI_TO_ANTHROPIC_ERROR_TYPE",
    "OPENAI_TO_ANTHROPIC_STOP_REASON",
    "convert__anthropic_to_openai__error",
    "convert__openai_to_anthropic__error",
]
