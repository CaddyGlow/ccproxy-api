"""Helpers for incrementally parsing server-sent events (SSE)."""

from __future__ import annotations

import json
from typing import Any


class SSEStreamParser:
    """Accumulate SSE fragments and yield decoded ``data:`` payloads.

    The parser keeps track of partial lines and events across ``feed`` calls so
    callers can push raw provider chunks (``str`` or ``bytes``) and only receive
    payloads when a full SSE event has been received. ``data: [DONE]`` sentinel
    events are filtered out automatically.
    """

    __slots__ = ("_line_remainder", "_event_lines", "_errors")

    def __init__(self) -> None:
        self._line_remainder: str = ""
        self._event_lines: list[str] = []
        self._errors: list[tuple[str, Exception]] = []

    def feed(self, chunk: str | bytes | None) -> list[Any]:
        """Process a streaming fragment and return decoded JSON payloads.

        Args:
            chunk: Raw chunk from the provider. ``bytes`` inputs are decoded
                using UTF-8. ``None`` or empty values yield no events.

        Returns:
            List of decoded JSON payloads for completed events. ``[DONE]``
            sentinels are omitted.
        """

        if not chunk:
            return []

        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="ignore")

        if not chunk:
            return []

        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
        buffered = f"{self._line_remainder}{chunk}"

        lines = buffered.split("\n")
        if buffered.endswith("\n"):
            self._line_remainder = ""
        else:
            self._line_remainder = lines.pop()

        completed: list[Any] = []

        for line in lines:
            if line == "":
                payload = self._finalize_event()
                if payload:
                    completed.append(payload)
                continue

            self._event_lines.append(line)

        return completed

    def flush(self) -> list[Any]:
        """Return any buffered payload when the stream ends."""

        if self._line_remainder:
            self._event_lines.append(self._line_remainder)
            self._line_remainder = ""

        payload = self._finalize_event()
        return [payload] if payload else []

    def consume_errors(self) -> list[tuple[str, Exception]]:
        """Return and clear parsing errors captured since the last call."""

        errors = self._errors
        self._errors = []
        return errors

    def _finalize_event(self) -> Any | None:
        if not self._event_lines:
            return None

        event_lines, self._event_lines = self._event_lines, []

        data_fields: list[str] = []
        for line in event_lines:
            if line.startswith("data:"):
                data_fields.append(line[5:].lstrip(" "))

        if not data_fields:
            return None

        # Try newline-preserving join first (Anthropic style), then a collapsed
        # join for providers that stream JSON without explicit newlines.
        candidates = ["\n".join(data_fields).strip()]
        collapsed = "".join(data_fields).strip()
        if collapsed and collapsed != candidates[0]:
            candidates.append(collapsed)

        last_exception: json.JSONDecodeError | None = None

        for candidate in candidates:
            if not candidate or candidate == "[DONE]":
                return None

            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_exception = exc
                continue

        if last_exception:
            # Reconstruct the raw event so we can retry when more data arrives.
            raw_event = "\n".join(event_lines) + "\n\n"
            self._line_remainder = f"{raw_event}{self._line_remainder}"
            self._errors.append((raw_event, last_exception))

        return None


__all__ = ["SSEStreamParser"]
