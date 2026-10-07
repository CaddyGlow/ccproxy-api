"""Regression tests for truncation creating invalid Anthropic tool blocks."""

from collections import Counter
from typing import Any

from ccproxy.llms.formatters.openai_to_anthropic.requests import _sanitize_tool_results
from ccproxy.llms.utils import estimate_request_tokens, truncate_to_fit


Message = dict[str, Any]


def _content_blocks(content: Any) -> list[Any]:
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return [content]
    return []


def _count_orphaned_tool_results(messages: list[Message]) -> int:
    """Count tool_result blocks without a matching preceding assistant tool_use."""
    orphan_count = 0

    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue

        valid_tool_uses: Counter[str] = Counter()
        if index > 0 and messages[index - 1].get("role") == "assistant":
            for block in _content_blocks(messages[index - 1].get("content")):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    if tool_id:
                        valid_tool_uses[str(tool_id)] += 1

        seen_results: Counter[str] = Counter()
        for block in _content_blocks(message.get("content")):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue

            tool_use_id = block.get("tool_use_id")
            key = str(tool_use_id) if tool_use_id else ""
            if key and seen_results[key] < valid_tool_uses.get(key, 0):
                seen_results[key] += 1
            else:
                orphan_count += 1

    return orphan_count


def _count_orphaned_tool_uses(messages: list[Message]) -> int:
    """Count assistant tool_use blocks without a matching next user tool_result."""
    orphan_count = 0

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        next_results: Counter[str] = Counter()
        if index + 1 < len(messages) and messages[index + 1].get("role") == "user":
            for block in _content_blocks(messages[index + 1].get("content")):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id:
                        next_results[str(tool_use_id)] += 1

        seen_uses: Counter[str] = Counter()
        for block in _content_blocks(message.get("content")):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            tool_id = block.get("id")
            key = str(tool_id) if tool_id else ""
            if key and seen_uses[key] < next_results.get(key, 0):
                seen_uses[key] += 1
            else:
                orphan_count += 1

    return orphan_count


def _tool_pair(tool_id: str, tool_payload_size: int = 2000) -> list[Message]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "analyze_data",
                    "input": {"dataset": "x" * tool_payload_size},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": "Analysis complete: found 42 items",
                }
            ],
        },
    ]


def test_sanitize_before_truncate_can_leave_orphaned_tool_result() -> None:
    """Document why adapters must truncate before sanitizing."""
    messages = [
        *_tool_pair("tool_split"),
        {"role": "user", "content": "Summarize the analysis."},
    ]

    initially_sanitized = _sanitize_tool_results(messages)
    truncated, was_truncated = truncate_to_fit(
        {"model": "claude-3-opus-20240229", "messages": initially_sanitized},
        max_input_tokens=200,
        preserve_recent=2,
        safety_margin=0.9,
    )

    assert was_truncated is True
    assert _count_orphaned_tool_results(truncated["messages"]) == 1


def test_truncate_then_sanitize_removes_orphaned_tool_result() -> None:
    messages = [
        *_tool_pair("tool_split"),
        {"role": "user", "content": "Summarize the analysis."},
    ]

    truncated, was_truncated = truncate_to_fit(
        {"model": "claude-3-opus-20240229", "messages": messages},
        max_input_tokens=200,
        preserve_recent=2,
        safety_margin=0.9,
    )

    assert was_truncated is True
    assert _count_orphaned_tool_results(truncated["messages"]) == 1

    sanitized = _sanitize_tool_results(truncated["messages"])

    assert _count_orphaned_tool_results(sanitized) == 0
    assert _count_orphaned_tool_uses(sanitized) == 0
    assert any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and "Previous tool results" in block.get("text", "")
        for message in sanitized
        for block in _content_blocks(message.get("content"))
    )


def test_sanitize_removes_tool_use_without_next_result() -> None:
    messages = [
        {"role": "user", "content": "Inspect the project."},
        *_tool_pair("tool_removed_result")[:1],
        {"role": "user", "content": "Continue after compaction."},
    ]

    assert _count_orphaned_tool_uses(messages) == 1

    sanitized = _sanitize_tool_results(messages)

    assert _count_orphaned_tool_results(sanitized) == 0
    assert _count_orphaned_tool_uses(sanitized) == 0
    assert any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and "Tool calls from compacted history" in block.get("text", "")
        for message in sanitized
        for block in _content_blocks(message.get("content"))
    )


def test_few_messages_with_massive_tool_results_are_reduced() -> None:
    messages: list[Message] = [
        {"role": "user", "content": "Read the README file"},
        *_tool_pair("tool_readme", tool_payload_size=100),
        {"role": "user", "content": "Now read the package.json"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_package",
                    "name": "Read",
                    "input": {"file_path": "/project/package.json"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_package",
                    "content": "package.json content line\n" * 15000,
                }
            ],
        },
        {"role": "user", "content": "Summarize both files for me."},
    ]
    request = {"model": "claude-3-opus-20240229", "messages": messages}
    original_tokens = estimate_request_tokens(request)

    truncated, was_truncated = truncate_to_fit(
        request,
        max_input_tokens=50000,
        preserve_recent=10,
        safety_margin=0.9,
    )

    assert was_truncated is True
    assert estimate_request_tokens(truncated) < original_tokens

    sanitized = _sanitize_tool_results(truncated["messages"])
    assert _count_orphaned_tool_results(sanitized) == 0
    assert _count_orphaned_tool_uses(sanitized) == 0
