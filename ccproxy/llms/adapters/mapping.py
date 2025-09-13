"""Endpoint mapping definitions for llms adapters.

This module documents and centralizes the key field mappings between:
- OpenAI Chat Completions <-> Anthropic Messages
- OpenAI Responses <-> Anthropic Messages
- OpenAI Chat Completions <-> OpenAI Responses

It is intentionally minimal and focused on the core, broadly supported fields
that we can map reliably across formats. Adapters implement the behavior and
use these helpers to keep mapping rules consistent.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Finish reason mapping (Anthropic -> OpenAI ChatCompletions)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Minimal defaults used by adapters
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS: Final[int] = 1024


__all__ = [
    "ANTHROPIC_TO_OPENAI_FINISH_REASON",
    "OPENAI_TO_ANTHROPIC_STOP_REASON",
    "DEFAULT_MAX_TOKENS",
]
