"""ChatCompletion accumulator for OpenAI streaming format."""

from __future__ import annotations

import copy
from typing import Any

from .delta_utils import accumulate_delta


class ChatCompletionAccumulator:
    """Accumulator for OpenAI ChatCompletion streaming format.

    Handles partial tool calls and other streaming data by accumulating
    chunks until complete objects are ready for validation.

    Follows the OpenAI SDK ChatCompletionStreamManager pattern.
    """

    def __init__(self) -> None:
        self._accumulated: dict[str, Any] = {}
        self._done_tool_calls: set[int] = set()
        self._current_tool_call_index: int | None = None

    def accumulate_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Accumulate a streaming chunk and return complete object if ready.

        Args:
            chunk: The incoming stream chunk data

        Returns:
            None if accumulation is ongoing, or the complete object when ready
            for validation
        """
        # For chunks without tool calls, return immediately UNLESS we have accumulated state
        # (in which case this might be a finish_reason chunk)
        if not self._has_tool_calls(chunk) and not self._accumulated:
            return chunk

        # For the first chunk, copy the base structure
        if not self._accumulated:
            self._accumulated = copy.deepcopy(chunk)
        else:
            # For subsequent chunks, preserve base fields and only accumulate deltas
            base_fields = {"id", "object", "created", "model"}
            chunk_copy = copy.deepcopy(chunk)

            # Remove base fields from chunk_copy to avoid concatenation
            for field in base_fields:
                if field in chunk_copy:
                    del chunk_copy[field]

            # Use accumulate_delta for the remaining fields (choices, etc.)
            self._accumulated = accumulate_delta(self._accumulated, chunk_copy)

        # Track tool call progress if present
        if self._has_tool_calls(chunk):
            self._track_tool_call_progress(chunk)

        # Don't validate if we have incomplete tool calls
        if self._has_incomplete_tool_calls():
            return None  # Continue accumulating

        # Return a copy for validation if chunk seems complete
        if self._should_emit_chunk(chunk):
            return copy.deepcopy(self._accumulated)

        # Continue accumulating
        return None

    def reset(self) -> None:
        """Reset accumulator state for next message."""
        self._accumulated.clear()
        self._done_tool_calls.clear()
        self._current_tool_call_index = None

    def _has_tool_calls(self, chunk: dict[str, Any]) -> bool:
        """Check if chunk contains tool call data."""
        if not isinstance(chunk, dict) or "choices" not in chunk:
            return False

        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta", {})
            if isinstance(delta, dict) and "tool_calls" in delta:
                return True

        return False

    def _track_tool_call_progress(self, chunk: dict[str, Any]) -> None:
        """Track progress of tool calls in this chunk."""
        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            tool_calls = delta.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                # Track current tool call index
                if "index" in tool_call:
                    self._current_tool_call_index = tool_call["index"]

                # Mark tool call as done if it has complete structure
                if self._is_tool_call_complete(tool_call):
                    index = tool_call.get("index", self._current_tool_call_index)
                    if index is not None:
                        self._done_tool_calls.add(index)

    def _is_tool_call_complete(self, tool_call: dict[str, Any]) -> bool:
        """Check if a tool call has all required fields."""
        if not tool_call.get("id"):
            return False

        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            return False

        if not function.get("name"):
            return False

        # Arguments can be empty string, but should be present
        return "arguments" in function

    def _has_incomplete_tool_calls(self) -> bool:
        """Check if accumulated state has incomplete tool calls."""
        if not self._accumulated.get("choices"):
            return False

        for choice in self._accumulated["choices"]:
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            tool_calls = delta.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                # Check if this tool call is incomplete
                if not self._is_tool_call_complete(tool_call):
                    return True

        return False

    def _should_emit_chunk(self, chunk: dict[str, Any]) -> bool:
        """Determine if we should emit the accumulated chunk for validation.

        We emit when:
        1. No tool calls are present (regular content)
        2. All tool calls in the accumulated state are complete AND we see a finish_reason
        """
        # If no tool calls in accumulated state, emit immediately
        if not self._has_any_tool_calls_in_accumulated():
            return True

        # For tool calls, only emit when both conditions are met:
        # 1. All tool calls are complete
        # 2. We see a finish_reason (indicates end of tool call sequence)
        has_finish_reason = False
        for choice in chunk.get("choices", []):
            if isinstance(choice, dict) and choice.get("finish_reason"):
                has_finish_reason = True
                break

        return bool(has_finish_reason and not self._has_incomplete_tool_calls())

    def _has_any_tool_calls_in_accumulated(self) -> bool:
        """Check if accumulated state has any tool calls."""
        if not self._accumulated.get("choices"):
            return False

        for choice in self._accumulated["choices"]:
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            tool_calls = delta.get("tool_calls", [])
            if isinstance(tool_calls, list) and tool_calls:
                return True

        return False
