"""SSE (Server-Sent Events) parser for Codex responses."""

import json
from typing import Any


def parse_sse_line(line: str) -> tuple[str | None, Any | None]:
    """Parse a single SSE line.

    Args:
        line: SSE line to parse

    Returns:
        Tuple of (event_type, data) or (None, None) if not parseable
    """
    line = line.strip()

    if not line:
        return None, None

    if line.startswith("event:"):
        return line[6:].strip(), None

    if line.startswith("data:"):
        data_str = line[5:].strip()

        if data_str == "[DONE]":
            return "done", None

        try:
            return "data", json.loads(data_str)
        except json.JSONDecodeError:
            return None, None

    return None, None


def extract_final_response(sse_content: str) -> dict[str, Any] | None:
    """Extract the final response from SSE content.

    Looks for the response.completed event in SSE stream.

    Args:
        sse_content: Complete SSE response content

    Returns:
        Final response data or None if not found
    """
    lines = sse_content.strip().split("\n")
    final_response = None

    for line in lines:
        event_type, data = parse_sse_line(line)

        if event_type == "data" and data and isinstance(data, dict):
            # Check for response.completed event
            if data.get("type") == "response.completed":
                # Found the completed response
                if "response" in data:
                    final_response = data["response"]
                else:
                    final_response = data
            elif data.get("type") == "response.in_progress" and "response" in data:
                # Update with in-progress data, but keep looking
                final_response = data["response"]

    return final_response


def parse_sse_stream(chunks: list[bytes]) -> dict[str, Any] | None:
    """Parse SSE stream chunks to extract final response.

    Args:
        chunks: List of byte chunks from SSE stream

    Returns:
        Final response data or None if not found
    """
    # Combine all chunks
    full_content = b"".join(chunks).decode("utf-8", errors="replace")
    return extract_final_response(full_content)


def is_sse_response(content: bytes | str) -> bool:
    """Check if content appears to be SSE format.

    Args:
        content: Response content to check

    Returns:
        True if content appears to be SSE format
    """
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8", errors="replace")
        except Exception:
            return False

    # Check for SSE markers
    content_start = content[:100].strip()
    return (
        content_start.startswith("event:")
        or content_start.startswith("data:")
        or "\nevent:" in content_start
        or "\ndata:" in content_start
    )
