"""SSE parsers for extracting final responses from streaming data.

This module provides functions that implement the SSEParserProtocol
to parse Server-Sent Event (SSE) streams and extract the final JSON response.
"""

import json
import re
from typing import Any


def last_json_data_event(raw: str) -> dict[str, Any] | None:
    """Parse SSE stream and extract the last JSON data event.

    This parser is designed to work with OpenAI/Codex-style SSE streams that
    contain multiple 'data:' events with JSON payloads, where the final response
    is typically in the last non-empty data event.

    Args:
        raw: Raw SSE stream content as a string

    Returns:
        Parsed JSON dict from the last data event, or None if no valid JSON found

    Example SSE format:
        data: {"choices": [{"delta": {"content": "Hello"}}]}

        data: {"choices": [{"delta": {"content": " world"}}]}

        data: [DONE]
    """
    if not raw:
        return None

    # Find all data: lines using regex
    data_pattern = r"^data:\s*(.+)$"
    data_lines = []

    for line in raw.strip().split("\n"):
        line = line.strip()
        match = re.match(data_pattern, line)
        if match:
            data_content = match.group(1).strip()
            # Skip [DONE] markers and empty content
            if data_content and data_content != "[DONE]":
                data_lines.append(data_content)

    # Process data lines in reverse order to find the last valid JSON
    for data_content in reversed(data_lines):
        try:
            parsed = json.loads(data_content)
            if isinstance(parsed, dict) and parsed:  # Non-empty dict
                return parsed
        except (json.JSONDecodeError, ValueError):
            # Skip invalid JSON and continue searching
            continue

    return None


def openai_completion_parser(raw: str) -> dict[str, Any] | None:
    """Parse OpenAI completion-style SSE stream.

    This is an alias for last_json_data_event since OpenAI completions
    use the same SSE format.

    Args:
        raw: Raw SSE stream content

    Returns:
        Parsed final completion response or None
    """
    return last_json_data_event(raw)


def anthropic_message_parser(raw: str) -> dict[str, Any] | None:
    """Parse Anthropic message-style SSE stream.

    Anthropic uses a similar SSE format but may have different event types.
    For now, this uses the same logic as OpenAI but can be customized later.

    Args:
        raw: Raw SSE stream content

    Returns:
        Parsed final message response or None
    """
    return last_json_data_event(raw)
