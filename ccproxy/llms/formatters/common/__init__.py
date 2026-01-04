"""Shared helpers used by formatter adapters."""

from .identifiers import ensure_identifier, normalize_suffix
from .streams import (
    IndexedToolCallTracker,
    ObfuscationTokenFactory,
    ReasoningBuffer,
    ReasoningPartState,
    ToolCallState,
    ToolCallTracker,
    build_anthropic_tool_use_block,
    emit_anthropic_tool_use_events,
)
from .thinking import (
    THINKING_CLOSE_PATTERN,
    THINKING_OPEN_PATTERN,
    THINKING_PATTERN,
    ThinkingSegment,
    merge_thinking_segments,
)
from .usage import (
    convert_anthropic_usage_to_openai_completion_usage,
    convert_anthropic_usage_to_openai_responses_usage,
    convert_openai_completion_usage_to_responses_usage,
    convert_openai_responses_usage_to_anthropic_usage,
    convert_openai_responses_usage_to_completion_usage,
)


__all__ = [
    "ensure_identifier",
    "normalize_suffix",
    "THINKING_PATTERN",
    "THINKING_OPEN_PATTERN",
    "THINKING_CLOSE_PATTERN",
    "ThinkingSegment",
    "merge_thinking_segments",
    "ReasoningBuffer",
    "ReasoningPartState",
    "ToolCallState",
    "ToolCallTracker",
    "IndexedToolCallTracker",
    "ObfuscationTokenFactory",
    "build_anthropic_tool_use_block",
    "emit_anthropic_tool_use_events",
    "convert_anthropic_usage_to_openai_completion_usage",
    "convert_anthropic_usage_to_openai_responses_usage",
    "convert_openai_completion_usage_to_responses_usage",
    "convert_openai_responses_usage_to_anthropic_usage",
    "convert_openai_responses_usage_to_completion_usage",
]
