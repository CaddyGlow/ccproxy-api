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
# Import within try/except to avoid circular import at package import time
try:  # pragma: no cover - import-time guard
    from ..anthropic_to_openai.helpers import (
        convert_anthropic_usage_to_openai_completion_usage,
        convert_anthropic_usage_to_openai_response_usage,
    )
except Exception:  # pragma: no cover
    convert_anthropic_usage_to_openai_completion_usage = None  # type: ignore
    convert_anthropic_usage_to_openai_response_usage = None  # type: ignore

try:  # pragma: no cover
    from ..openai_to_anthropic.helpers import (
        convert_openai_completion_usage_to_anthropic_usage,
        convert_openai_response_usage_to_anthropic_usage,
    )
except Exception:  # pragma: no cover
    convert_openai_completion_usage_to_anthropic_usage = None  # type: ignore
    convert_openai_response_usage_to_anthropic_usage = None  # type: ignore

try:  # pragma: no cover
    from ..openai_to_openai.helpers import (
        convert_openai_response_usage_to_openai_completion_usage,
    )
except Exception:  # pragma: no cover
    convert_openai_response_usage_to_openai_completion_usage = None  # type: ignore

__all__ = [
    "ANTHROPIC_TO_OPENAI_ERROR_TYPE",
    "ANTHROPIC_TO_OPENAI_FINISH_REASON",
    "DEFAULT_MAX_TOKENS",
    "OPENAI_TO_ANTHROPIC_ERROR_TYPE",
    "OPENAI_TO_ANTHROPIC_STOP_REASON",
    "convert_anthropic_error_to_openai",
    "convert_openai_error_to_anthropic",
    "normalize_openai_error",
    # expose usage conversion helpers for backwards compatibility
    "convert_anthropic_usage_to_openai_completion_usage",
    "convert_anthropic_usage_to_openai_response_usage",
    "convert_openai_completion_usage_to_anthropic_usage",
    "convert_openai_response_usage_to_anthropic_usage",
    "convert_openai_response_usage_to_openai_completion_usage",
]
