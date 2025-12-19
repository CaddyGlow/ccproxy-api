"""Test _sanitize_tool_results method for removing orphaned tool_result blocks.

This module tests the bug fix for orphaned tool_result blocks that occur when
conversation history is compacted. When tool_use blocks are removed during
compaction, their corresponding tool_result blocks become orphaned, causing
the Anthropic API to reject requests with:
"unexpected tool_use_id found in tool_result blocks"

The _sanitize_tool_results method fixes this by:
1. Removing orphaned tool_result blocks that don't have matching tool_use blocks
   in the immediately preceding assistant message
2. Converting orphaned results to text blocks to preserve information
3. Keeping valid tool_result blocks that have matching tool_use blocks

Real-world scenario:
- A long conversation with multiple tool calls gets compacted to stay within token limits
- Earlier assistant messages with tool_use blocks are removed
- Their corresponding tool_result blocks remain, becoming orphaned
- The sanitization method removes these orphaned blocks before sending to Anthropic API
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from ccproxy.adapters.openai.adapter import OpenAIAdapter


@pytest.fixture
def adapter() -> OpenAIAdapter:
    """Create OpenAI adapter instance for testing."""
    return OpenAIAdapter()


@pytest.fixture
def mock_logger() -> Mock:
    """Mock the structlog logger to avoid logging during tests."""
    with patch("ccproxy.adapters.openai.adapter.logger") as mock:
        yield mock


# Helper functions for creating test messages


def create_user_text_message(text: str) -> dict[str, Any]:
    """Create a user message with text content."""
    return {"role": "user", "content": text}


def create_assistant_text_message(text: str) -> dict[str, Any]:
    """Create an assistant message with text content."""
    return {"role": "assistant", "content": text}


def create_assistant_with_tool_use(
    text: str, tool_uses: list[dict[str, Any]]
) -> dict[str, Any]:
    """Create an assistant message with text and tool_use blocks.

    Args:
        text: Text content for the message
        tool_uses: List of tool_use blocks, each should have 'id', 'name', 'input'
    """
    content = [{"type": "text", "text": text}]
    for tool_use in tool_uses:
        content.append(
            {
                "type": "tool_use",
                "id": tool_use["id"],
                "name": tool_use.get("name", "test_tool"),
                "input": tool_use.get("input", {}),
            }
        )
    return {"role": "assistant", "content": content}


def create_user_with_tool_result(
    tool_results: list[dict[str, Any]], text: str | None = None
) -> dict[str, Any]:
    """Create a user message with tool_result blocks and optional text.

    Args:
        tool_results: List of tool_result blocks, each should have 'tool_use_id', 'content'
        text: Optional text content to include
    """
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for tool_result in tool_results:
        content.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_result["tool_use_id"],
                "content": tool_result.get("content", ""),
            }
        )
    return {"role": "user", "content": content}


# Test Cases


class TestSanitizeToolResults:
    """Test suite for _sanitize_tool_results method."""

    def test_valid_tool_result_preserved(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that valid tool_result blocks are preserved.

        Scenario: Normal tool use flow
        - Assistant message with tool_use(id="tool_123")
        - User message with tool_result(tool_use_id="tool_123")
        Result: tool_result should be kept unchanged
        """
        messages = [
            create_assistant_with_tool_use(
                "I'll help you with that.",
                [{"id": "tool_123", "name": "calculator", "input": {"x": 5}}],
            ),
            create_user_with_tool_result([{"tool_use_id": "tool_123", "content": "10"}]),
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "tool_result"
        assert result[1]["content"][0]["tool_use_id"] == "tool_123"
        # Should not log any warnings for valid results
        mock_logger.warning.assert_not_called()

    def test_orphaned_tool_result_removed(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that orphaned tool_result blocks are removed and converted to text.

        Scenario: After conversation compaction
        - User message with tool_result(tool_use_id="orphan_456")
        - NO preceding assistant with matching tool_use
        Result: tool_result should be removed and converted to text
        """
        messages = [
            create_user_text_message("Hello"),
            create_user_with_tool_result(
                [{"tool_use_id": "orphan_456", "content": "orphaned result"}]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        # The orphaned result should be converted to text
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "text"
        assert "Previous tool results" in result[1]["content"][0]["text"]
        assert "orphan_456" in result[1]["content"][0]["text"]
        # Should log warning about orphaned result
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[0][0] == "orphaned_tool_result_removed"

    def test_mixed_valid_and_orphaned(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test mixed valid and orphaned tool_result blocks.

        Scenario: Partial compaction
        - Assistant with tool_use(id="valid_1")
        - User with tool_result(tool_use_id="valid_1") AND tool_result(tool_use_id="orphan_2")
        Result: valid_1 kept, orphan_2 converted to text
        """
        messages = [
            create_assistant_with_tool_use(
                "Let me check that.",
                [{"id": "valid_1", "name": "search", "input": {"query": "test"}}],
            ),
            create_user_with_tool_result(
                [
                    {"tool_use_id": "valid_1", "content": "Found 5 results"},
                    {"tool_use_id": "orphan_2", "content": "old result"},
                ]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 2
        user_content = result[1]["content"]

        # Should have text block (from orphaned) + valid tool_result
        assert len(user_content) == 2

        # First should be text block with orphaned info
        assert user_content[0]["type"] == "text"
        assert "Previous tool results" in user_content[0]["text"]
        assert "orphan_2" in user_content[0]["text"]

        # Second should be the valid tool_result
        assert user_content[1]["type"] == "tool_result"
        assert user_content[1]["tool_use_id"] == "valid_1"

        # Should log warning about orphaned result
        mock_logger.warning.assert_called_once()

    def test_multiple_tool_uses_preserved(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that multiple valid tool_uses and results are all preserved.

        Scenario: Multiple tools used in one turn
        - Assistant with tool_use(id="t1"), tool_use(id="t2"), tool_use(id="t3")
        - User with tool_result for all three
        Result: all three should be preserved
        """
        messages = [
            create_assistant_with_tool_use(
                "I'll use three tools.",
                [
                    {"id": "t1", "name": "tool1", "input": {}},
                    {"id": "t2", "name": "tool2", "input": {}},
                    {"id": "t3", "name": "tool3", "input": {}},
                ],
            ),
            create_user_with_tool_result(
                [
                    {"tool_use_id": "t1", "content": "result1"},
                    {"tool_use_id": "t2", "content": "result2"},
                    {"tool_use_id": "t3", "content": "result3"},
                ]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 2
        user_content = result[1]["content"]
        assert len(user_content) == 3

        # All should be valid tool_results
        for i, expected_id in enumerate(["t1", "t2", "t3"]):
            assert user_content[i]["type"] == "tool_result"
            assert user_content[i]["tool_use_id"] == expected_id

        # No warnings should be logged
        mock_logger.warning.assert_not_called()

    def test_conversation_compaction_scenario(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test the real bug scenario: conversation compaction leaves orphaned results.

        Scenario: The actual bug from production
        - Message 0: user "hello"
        - Message 1: assistant "I'll help" + tool_use(id="original_tool")
        - Message 2: user tool_result(tool_use_id="original_tool")
        - Message 3: assistant "Here's the result"
        - Message 4: user "thanks, now do X"
        - Message 5: assistant "Sure" + tool_use(id="new_tool")
        - Message 6: user tool_result(tool_use_id="new_tool") + tool_result(tool_use_id="original_tool")

        After compaction, messages 0-3 are removed, leaving message 6 with orphaned "original_tool"
        """
        # Simulate AFTER compaction - messages 0-3 removed
        messages = [
            create_user_text_message("thanks, now do X"),
            create_assistant_with_tool_use(
                "Sure, I'll do that.",
                [{"id": "new_tool", "name": "new_action", "input": {}}],
            ),
            create_user_with_tool_result(
                [
                    {"tool_use_id": "new_tool", "content": "new result"},
                    {
                        "tool_use_id": "original_tool",
                        "content": "old result from removed history",
                    },
                ]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 3
        user_content = result[2]["content"]

        # Should have text block (orphaned) + valid tool_result
        assert len(user_content) == 2

        # First is text with orphaned info
        assert user_content[0]["type"] == "text"
        assert "original_tool" in user_content[0]["text"]

        # Second is valid tool_result
        assert user_content[1]["type"] == "tool_result"
        assert user_content[1]["tool_use_id"] == "new_tool"

        # Should log warning
        mock_logger.warning.assert_called_once()

    def test_empty_messages_list(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that empty messages list is handled gracefully.

        Scenario: Edge case - empty input
        Result: Returns empty list without errors
        """
        messages: list[dict[str, Any]] = []
        result = adapter._sanitize_tool_results(messages)
        assert result == []
        mock_logger.warning.assert_not_called()

    def test_messages_with_no_tool_content(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test messages with no tool_use or tool_result blocks.

        Scenario: Normal conversation without tools
        Result: Messages passed through unchanged
        """
        messages = [
            create_user_text_message("Hello"),
            create_assistant_text_message("Hi there!"),
            create_user_text_message("How are you?"),
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 3
        assert result == messages
        mock_logger.warning.assert_not_called()

    def test_user_message_text_preserved_with_tool_result(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that text blocks in user messages are preserved alongside tool_results.

        Scenario: User provides both text and tool results
        - Assistant with tool_use(id="tool_1")
        - User with text "Here's the result:" AND tool_result(tool_use_id="tool_1")
        Result: Both text and tool_result should remain
        """
        messages = [
            create_assistant_with_tool_use(
                "Let me check that.",
                [{"id": "tool_1", "name": "checker", "input": {}}],
            ),
            create_user_with_tool_result(
                [{"tool_use_id": "tool_1", "content": "result data"}],
                text="Here's the result:",
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 2
        user_content = result[1]["content"]
        assert len(user_content) == 2

        # Text block should be preserved
        assert user_content[0]["type"] == "text"
        assert user_content[0]["text"] == "Here's the result:"

        # Tool result should be preserved
        assert user_content[1]["type"] == "tool_result"
        assert user_content[1]["tool_use_id"] == "tool_1"

        mock_logger.warning.assert_not_called()

    def test_first_message_user_with_tool_result(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test user message with tool_result as first message (no preceding assistant).

        Scenario: Invalid state - tool_result without any preceding assistant
        - User message with tool_result as FIRST message
        Result: tool_result should be removed (no valid tool_use possible)
        """
        messages = [
            create_user_with_tool_result(
                [{"tool_use_id": "impossible_tool", "content": "orphaned"}]
            )
        ]

        result = adapter._sanitize_tool_results(messages)

        # Should convert to text block
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert "impossible_tool" in result[0]["content"][0]["text"]

        mock_logger.warning.assert_called_once()

    def test_tool_result_with_complex_content(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test tool_result with complex content (list of blocks).

        Scenario: Tool result contains structured content
        - Orphaned tool_result with content as list of blocks
        Result: Should extract text from blocks when converting to text
        """
        messages = [
            create_user_text_message("Hello"),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "orphan_complex",
                        "content": [
                            {"type": "text", "text": "Part 1"},
                            {"type": "text", "text": "Part 2"},
                        ],
                    }
                ],
            },
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 2
        user_content = result[1]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"
        # Should include both parts
        assert "Part 1" in user_content[0]["text"]
        assert "Part 2" in user_content[0]["text"]

        mock_logger.warning.assert_called_once()

    def test_tool_result_with_long_content_truncated(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that very long tool_result content is truncated in text conversion.

        Scenario: Orphaned tool result with very long content
        Result: Content should be truncated to 500 chars with ellipsis
        """
        long_content = "x" * 1000  # 1000 character string
        messages = [
            create_user_text_message("Hello"),
            create_user_with_tool_result(
                [{"tool_use_id": "orphan_long", "content": long_content}]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 2
        user_content = result[1]["content"]
        text_block = user_content[0]["text"]

        # Should be truncated with ellipsis
        assert "..." in text_block
        assert len(text_block) < len(long_content)
        assert "orphan_long" in text_block

        mock_logger.warning.assert_called_once()

    def test_only_orphaned_tool_results_message_preserved(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that a message with only orphaned tool_results is preserved as text.

        Scenario: User message contains only orphaned tool_results (no text, no valid results)
        Result: Message should be preserved with text content describing the orphaned results
        """
        messages = [
            create_user_text_message("Hello"),
            create_user_with_tool_result(
                [
                    {"tool_use_id": "orphan_1", "content": "result 1"},
                    {"tool_use_id": "orphan_2", "content": "result 2"},
                ]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        # Message should be preserved but converted to text
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "text"
        assert "orphan_1" in result[1]["content"][0]["text"]
        assert "orphan_2" in result[1]["content"][0]["text"]

    def test_non_user_messages_passed_through(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that non-user messages (assistant) are passed through unchanged.

        Scenario: Mix of user and assistant messages
        Result: Only user messages with tool_results are processed
        """
        messages = [
            create_assistant_text_message("Hello"),
            create_user_text_message("Hi"),
            create_assistant_with_tool_use(
                "Let me help.",
                [{"id": "tool_1", "name": "helper", "input": {}}],
            ),
            create_user_with_tool_result(
                [{"tool_use_id": "tool_1", "content": "done"}]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 4
        # Assistant messages should be unchanged
        assert result[0] == messages[0]
        assert result[2] == messages[2]
        # User messages processed correctly
        assert result[1] == messages[1]
        assert result[3]["content"][0]["tool_use_id"] == "tool_1"

    def test_user_message_with_string_content(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test that user messages with string content (not list) are handled.

        Scenario: User message has string content instead of list
        Result: Message passed through unchanged (no tool_results to process)
        """
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = adapter._sanitize_tool_results(messages)

        assert len(result) == 2
        assert result == messages
        mock_logger.warning.assert_not_called()

    def test_partial_match_orphaned(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test when some tool_use_ids match but others don't.

        Scenario: Multiple results, only some have matching tool_use
        - Assistant with tool_use(id="valid_1") and tool_use(id="valid_2")
        - User with results for "valid_1", "orphan_3", "valid_2"
        Result: valid_1 and valid_2 kept, orphan_3 converted to text
        """
        messages = [
            create_assistant_with_tool_use(
                "Using two tools.",
                [
                    {"id": "valid_1", "name": "tool1", "input": {}},
                    {"id": "valid_2", "name": "tool2", "input": {}},
                ],
            ),
            create_user_with_tool_result(
                [
                    {"tool_use_id": "valid_1", "content": "result1"},
                    {"tool_use_id": "orphan_3", "content": "orphaned"},
                    {"tool_use_id": "valid_2", "content": "result2"},
                ]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        user_content = result[1]["content"]
        # Should have text block + 2 valid results
        assert len(user_content) == 3

        # First is text with orphaned info
        assert user_content[0]["type"] == "text"
        assert "orphan_3" in user_content[0]["text"]

        # Other two are valid results
        assert user_content[1]["type"] == "tool_result"
        assert user_content[1]["tool_use_id"] == "valid_1"
        assert user_content[2]["type"] == "tool_result"
        assert user_content[2]["tool_use_id"] == "valid_2"

    def test_assistant_with_string_content_no_tool_use(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test assistant message with string content (no tool_use blocks).

        Scenario: Preceding assistant has string content, not list
        - Assistant with string content (no tool_use)
        - User with tool_result
        Result: Tool result should be orphaned (no valid tool_use found)
        """
        messages = [
            {"role": "assistant", "content": "Just a text response"},
            create_user_with_tool_result(
                [{"tool_use_id": "orphan", "content": "result"}]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        # Tool result should be converted to text
        assert len(result) == 2
        user_content = result[1]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"
        assert "orphan" in user_content[0]["text"]

        mock_logger.warning.assert_called_once()

    def test_multiple_orphaned_conversions(
        self, adapter: OpenAIAdapter, mock_logger: Mock
    ) -> None:
        """Test multiple orphaned tool_results are all included in text conversion.

        Scenario: Multiple orphaned results in one message
        Result: All orphaned results should be listed in the text block
        """
        messages = [
            create_user_text_message("Hello"),
            create_user_with_tool_result(
                [
                    {"tool_use_id": "orphan_1", "content": "result one"},
                    {"tool_use_id": "orphan_2", "content": "result two"},
                    {"tool_use_id": "orphan_3", "content": "result three"},
                ]
            ),
        ]

        result = adapter._sanitize_tool_results(messages)

        user_content = result[1]["content"]
        text_block = user_content[0]["text"]

        # All three orphaned results should be mentioned
        assert "orphan_1" in text_block
        assert "orphan_2" in text_block
        assert "orphan_3" in text_block
        assert "result one" in text_block
        assert "result two" in text_block
        assert "result three" in text_block
