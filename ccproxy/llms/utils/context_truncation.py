"""Context window truncation utilities."""

import copy
from typing import Any

from ccproxy.core.logging import get_logger

from .token_estimation import estimate_request_tokens


logger = get_logger(__name__)


# Maximum characters to keep for truncated content blocks
MAX_TRUNCATED_CONTENT_CHARS = 10000


def _truncate_large_content_blocks(
    messages: list[dict[str, Any]],
    max_chars: int = MAX_TRUNCATED_CONTENT_CHARS,
) -> tuple[list[dict[str, Any]], int]:
    """Truncate large content blocks within messages.

    This is a fallback when message-level truncation isn't enough.
    Targets large tool_result blocks and text content.

    Args:
        messages: List of messages to process
        max_chars: Maximum characters to keep per content block

    Returns:
        Tuple of (modified_messages, blocks_truncated_count)
    """
    truncated_count = 0
    modified_messages = []

    for msg in messages:
        msg_copy = copy.deepcopy(msg)
        content = msg_copy.get("content")

        if isinstance(content, str) and len(content) > max_chars:
            # Truncate large string content
            msg_copy["content"] = (
                content[:max_chars]
                + f"\n\n[Content truncated - {len(content) - max_chars} characters removed]"
            )
            truncated_count += 1

        elif isinstance(content, list):
            # Process content blocks
            new_content = []
            for block in content:
                if isinstance(block, dict):
                    block_copy = copy.deepcopy(block)

                    # Handle tool_result blocks with large content
                    if block_copy.get("type") == "tool_result":
                        tool_content = block_copy.get("content", "")
                        if (
                            isinstance(tool_content, str)
                            and len(tool_content) > max_chars
                        ):
                            block_copy["content"] = (
                                tool_content[:max_chars]
                                + f"\n\n[Tool result truncated - {len(tool_content) - max_chars} characters removed]"
                            )
                            truncated_count += 1

                    # Handle text blocks with large content
                    elif block_copy.get("type") == "text":
                        text = block_copy.get("text", "")
                        if len(text) > max_chars:
                            block_copy["text"] = (
                                text[:max_chars]
                                + f"\n\n[Text truncated - {len(text) - max_chars} characters removed]"
                            )
                            truncated_count += 1

                    new_content.append(block_copy)
                else:
                    new_content.append(block)

            msg_copy["content"] = new_content

        modified_messages.append(msg_copy)

    return modified_messages, truncated_count


def truncate_to_fit(
    request_data: dict[str, Any],
    max_input_tokens: int,
    preserve_recent: int = 10,
    safety_margin: float = 0.9,
) -> tuple[dict[str, Any], bool]:
    """Truncate request to fit within token limit.

    Strategy:
    1. Always preserve system prompt and tools
    2. Try to preserve the last N messages (preserve_recent)
    3. Remove oldest messages first
    4. If too few messages to truncate, reduce preserve_recent dynamically
    5. As a last resort, truncate large content blocks within messages
    6. Add a truncation notice when content is removed

    Args:
        request_data: The request payload
        max_input_tokens: Model's max input token limit
        preserve_recent: Number of recent messages to always keep
        safety_margin: Target this fraction of max to allow for estimation error

    Returns:
        Tuple of (modified_request_data, was_truncated)
    """
    target_tokens = int(max_input_tokens * safety_margin)

    current_tokens = estimate_request_tokens(request_data)
    if current_tokens <= target_tokens:
        return request_data, False

    # Work on a copy
    modified = copy.deepcopy(request_data)
    messages = modified.get("messages", [])

    # If we have fewer messages than preserve_recent, reduce preserve_recent
    # We need at least 1 message to be truncatable for this strategy to work
    effective_preserve = min(preserve_recent, len(messages) - 1)

    # If we have 0 or 1 messages, we can't do message-level truncation
    # Skip to content-level truncation
    if effective_preserve < 0:
        effective_preserve = 0

    # Split into truncatable and preserved messages
    if effective_preserve > 0:
        truncatable = messages[:-effective_preserve]
        preserved = messages[-effective_preserve:]
    else:
        truncatable = list(messages)
        preserved = []

    # Remove oldest messages until we're under the limit
    removed_count = 0
    while truncatable and estimate_request_tokens(modified) > target_tokens:
        truncatable.pop(0)
        removed_count += 1
        modified["messages"] = truncatable + preserved

    # Check if we're still over the limit after removing all truncatable messages
    if estimate_request_tokens(modified) > target_tokens:
        logger.info(
            "context_truncation_message_level_insufficient",
            reason="still_over_limit_after_message_truncation",
            message_count=len(modified.get("messages", [])),
            current_tokens=estimate_request_tokens(modified),
            target_tokens=target_tokens,
            category="context_management",
        )

        # Fallback: truncate large content blocks within remaining messages
        truncated_messages, blocks_truncated = _truncate_large_content_blocks(
            modified.get("messages", [])
        )
        modified["messages"] = truncated_messages

        if blocks_truncated > 0:
            logger.info(
                "context_truncation_content_level",
                blocks_truncated=blocks_truncated,
                current_tokens=estimate_request_tokens(modified),
                target_tokens=target_tokens,
                category="context_management",
            )

        # If still over limit after content truncation, log error
        final_tokens = estimate_request_tokens(modified)
        if final_tokens > target_tokens:
            logger.error(
                "context_truncation_failed",
                reason="still_over_limit_after_all_truncation",
                final_tokens=final_tokens,
                target_tokens=target_tokens,
                messages_removed=removed_count,
                blocks_truncated=blocks_truncated,
                category="context_management",
            )
            # Still return the truncated version - it's better than nothing
            # The API will return an error, but at least we tried

    # Add truncation notice as first user message if we removed content
    if removed_count > 0:
        notice = {
            "role": "user",
            "content": f"[Context truncated - {removed_count} earlier messages removed to fit context window]",
        }
        modified["messages"] = [notice] + modified["messages"]

    final_tokens = estimate_request_tokens(modified)

    logger.info(
        "context_truncated",
        original_tokens=current_tokens,
        final_tokens=final_tokens,
        messages_removed=removed_count,
        target_tokens=target_tokens,
        effective_preserve_recent=effective_preserve,
        category="context_management",
    )

    return modified, True
