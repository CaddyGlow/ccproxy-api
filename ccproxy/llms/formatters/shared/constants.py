"""Shared constant mappings for LLM adapters."""

from __future__ import annotations

from typing import Final


ANTHROPIC_TO_OPENAI_FINISH_REASON: Final[dict[str, str]] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    # Anthropic-specific values mapped to closest reasonable OpenAI value
    "pause_turn": "stop",
    "refusal": "stop",
}

OPENAI_TO_ANTHROPIC_STOP_REASON: Final[dict[str, str]] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}

OPENAI_TO_ANTHROPIC_ERROR_TYPE: Final[dict[str, str]] = {
    "invalid_request_error": "invalid_request_error",
    "authentication_error": "invalid_request_error",
    "permission_error": "invalid_request_error",
    "not_found_error": "invalid_request_error",
    "rate_limit_error": "rate_limit_error",
    "internal_server_error": "api_error",
    "overloaded_error": "api_error",
}

ANTHROPIC_TO_OPENAI_ERROR_TYPE: Final[dict[str, str]] = {
    "invalid_request_error": "invalid_request_error",
    "authentication_error": "authentication_error",
    "permission_error": "permission_error",
    "not_found_error": "invalid_request_error",  # OpenAI doesn't expose not_found
    "rate_limit_error": "rate_limit_error",
    "api_error": "api_error",
    "overloaded_error": "api_error",
    "billing_error": "invalid_request_error",
    "timeout_error": "api_error",
}

DEFAULT_MAX_TOKENS: Final[int] = 1024


__all__ = [
    "ANTHROPIC_TO_OPENAI_FINISH_REASON",
    "OPENAI_TO_ANTHROPIC_STOP_REASON",
    "OPENAI_TO_ANTHROPIC_ERROR_TYPE",
    "ANTHROPIC_TO_OPENAI_ERROR_TYPE",
    "DEFAULT_MAX_TOKENS",
]
