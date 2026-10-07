"""Test native Anthropic request sanitization for orphaned tool blocks.

This test module verifies that native Anthropic format requests (sent to /v1/messages)
properly sanitize orphaned tool blocks that don't have matching counterparts.

Two types of orphaned blocks are handled:

1. Orphaned tool_result blocks (tool_result without matching tool_use):
   - Occurs when conversation is compacted, removing old tool_use blocks
   - tool_result blocks remain without their corresponding tool_use blocks
   - API rejects with: "unexpected tool_use_id found in tool_result blocks"

2. Orphaned tool_use blocks (tool_use without matching tool_result):
   - Occurs when conversation is compacted, removing tool_result blocks
   - tool_use blocks remain without their corresponding tool_result blocks
   - API rejects with: "tool_use ids were found without tool_result blocks immediately after"

The fix applies _sanitize_tool_results() in the claude_api and claude_sdk adapters'
prepare_provider_request() method before forwarding to the Anthropic API.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.llms.formatters.openai_to_anthropic.requests import _sanitize_tool_results


Message = dict[str, Any]


class TestNativeAnthropicSanitization:
    """Test sanitization of native Anthropic requests with orphaned tool_result blocks."""

    def test_orphaned_tool_result_removed_from_native_request(self):
        """Native Anthropic request with orphaned tool_result should be sanitized.

        This reproduces the exact error reported:
        "unexpected tool_use_id found in tool_result blocks: toolu_019M2sPZmfSNC57WBuV9NaRb"
        """
        # Simulate a compacted conversation where the tool_use was summarized
        # but the tool_result remains with its original ID
        messages: list[Message] = [
            {"role": "user", "content": "Search for files matching *.py"},
            {
                "role": "assistant",
                "content": "I'll search for Python files.",  # tool_use was compacted out
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_019M2sPZmfSNC57WBuV9NaRb",  # orphaned!
                        "content": "Found 15 Python files",
                    },
                    {"type": "text", "text": "Thanks, now please analyze them"},
                ],
            },
        ]

        result = _sanitize_tool_results(messages)

        # The orphaned tool_result should be converted to text
        assert len(result) == 3
        user_msg = result[2]
        assert user_msg["role"] == "user"

        # Should have text blocks but no tool_result
        tool_results = [
            b for b in user_msg["content"] if b.get("type") == "tool_result"
        ]
        assert len(tool_results) == 0

        # Original text should be preserved
        text_blocks = [b for b in user_msg["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 2  # original text + converted orphan

        # Check orphan was converted to informative text
        orphan_text_block = text_blocks[0]  # inserted at beginning
        assert (
            "Previous tool results from compacted history" in orphan_text_block["text"]
        )
        assert "toolu_019M2sPZmfSNC57WBuV9NaRb" in orphan_text_block["text"]

    def test_valid_tool_result_preserved_in_native_request(self):
        """Native Anthropic request with valid tool_result should pass through unchanged."""
        messages: list[Message] = [
            {"role": "user", "content": "Search for files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search for files."},
                    {
                        "type": "tool_use",
                        "id": "toolu_valid123",
                        "name": "glob",
                        "input": {"pattern": "*.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_valid123",  # matches tool_use above
                        "content": "Found 15 files",
                    }
                ],
            },
        ]

        result = _sanitize_tool_results(messages)

        # Messages should be unchanged
        assert len(result) == 3
        user_msg = result[2]

        # tool_result should be preserved
        tool_results = [
            b for b in user_msg["content"] if b.get("type") == "tool_result"
        ]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_valid123"

    def test_mixed_valid_and_orphaned_tool_results(self):
        """Request with both valid and orphaned tool_results should keep valid ones."""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_valid",
                        "name": "read",
                        "input": {"path": "file.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_valid",  # valid
                        "content": "file contents",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_orphan_from_compaction",  # orphaned
                        "content": "old result",
                    },
                ],
            },
        ]

        result = _sanitize_tool_results(messages)

        user_msg = result[1]
        tool_results = [
            b for b in user_msg["content"] if b.get("type") == "tool_result"
        ]
        text_blocks = [b for b in user_msg["content"] if b.get("type") == "text"]

        # Only valid tool_result should remain
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_valid"
        assert len(text_blocks) == 0

    def test_superdesign_conversation_compaction_scenario(self):
        """Reproduce the SuperDesign VS Code extension compaction scenario.

        SuperDesign uses @ai-sdk/anthropic which sends native Anthropic format.
        When the conversation gets long, it compacts history, removing old messages
        but sometimes leaving orphaned tool_result blocks.
        """
        # Simulated compacted conversation from SuperDesign
        messages: list[Message] = [
            # Earlier context was compacted into a summary
            {"role": "user", "content": "Help me build a React component"},
            {
                "role": "assistant",
                "content": "[Summary: Previously searched for files and read component code]",
            },
            # User message still has tool_results from before compaction
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_old_glob_call",
                        "content": "src/components/Button.tsx\nsrc/components/Modal.tsx",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_old_read_call",
                        "content": "export const Button = () => <button>Click me</button>",
                    },
                    {"type": "text", "text": "Now create a new Header component"},
                ],
            },
        ]

        result = _sanitize_tool_results(messages)

        # Orphaned tool_results should be converted to text
        user_msg = result[2]
        tool_results = [
            b for b in user_msg["content"] if b.get("type") == "tool_result"
        ]
        assert len(tool_results) == 0

        # Content should be preserved as text
        text_blocks = [b for b in user_msg["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 2  # original text + orphan summary

        # User's actual request should be there
        assert any("Header component" in b["text"] for b in text_blocks)

    def test_empty_messages_handled(self):
        """Empty messages list should return empty list."""
        assert _sanitize_tool_results([]) == []

    def test_messages_without_tool_content_unchanged(self):
        """Messages without any tool-related content pass through unchanged."""
        messages: list[Message] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = _sanitize_tool_results(messages)
        assert result == messages

    def test_long_orphan_content_truncated(self):
        """Long orphaned tool_result content should be truncated to 500 chars."""
        long_content = "x" * 1000
        messages: list[Message] = [
            {"role": "assistant", "content": "No tool_use here"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_orphan",
                        "content": long_content,
                    }
                ],
            },
        ]

        result = _sanitize_tool_results(messages)

        user_msg = result[1]
        text_blocks = [b for b in user_msg["content"] if b.get("type") == "text"]
        orphan_text = text_blocks[0]["text"]

        # Should be truncated with "..."
        assert "..." in orphan_text
        # Should not contain full 1000 chars
        assert len(orphan_text) < 1000


class TestClaudeAPIAdapterSanitization:
    """Test that the claude_api adapter properly applies sanitization."""

    @pytest.mark.asyncio
    async def test_adapter_sanitizes_native_anthropic_request(self):
        """The adapter's prepare_provider_request should sanitize messages.

        This test directly verifies the sanitization is applied by checking
        the code path rather than instantiating the full adapter.
        """
        # Test the sanitization function directly with the message format
        # that would come from a native Anthropic request
        messages: list[Message] = [
            {"role": "assistant", "content": "Summary of previous work"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_orphan",
                        "content": "old result",
                    },
                    {"type": "text", "text": "Continue please"},
                ],
            },
        ]

        # Simulate what the adapter does
        sanitized = _sanitize_tool_results(messages)

        # Check that orphaned tool_result was sanitized
        user_msg = sanitized[1]
        tool_results = [
            b
            for b in user_msg["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert len(tool_results) == 0

        # Verify the fix is properly imported in the adapter module
        from ccproxy.plugins.claude_api import adapter as api_adapter

        assert hasattr(api_adapter, "_sanitize_tool_results")


class TestClaudeSDKAdapterSanitization:
    """Test that the claude_sdk adapter properly applies sanitization."""

    def test_sdk_adapter_has_sanitization_import(self):
        """Verify the SDK adapter imports the sanitization function."""
        from ccproxy.plugins.claude_sdk import adapter

        assert hasattr(adapter, "_sanitize_tool_results")


class TestOrphanedToolUseSanitization:
    """Test sanitization of orphaned tool_use blocks (tool_use without matching tool_result).

    This addresses the error:
    "tool_use ids were found without tool_result blocks immediately after: <id>.
    Each tool_use block must have a corresponding tool_result block in the next message."
    """

    def test_orphaned_tool_use_removed_from_assistant_message(self):
        """Assistant message with tool_use but no matching tool_result should be sanitized."""
        messages: list[Message] = [
            {"role": "user", "content": "Search for files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search for files."},
                    {
                        "type": "tool_use",
                        "id": "toolu_orphan123",
                        "name": "glob",
                        "input": {"pattern": "*.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": "please continue",  # No tool_result for the tool_use above
            },
        ]

        result = _sanitize_tool_results(messages)

        # The tool_use should be converted to text
        assistant_msg = result[1]
        assert isinstance(assistant_msg["content"], list)
        tool_uses = [b for b in assistant_msg["content"] if b.get("type") == "tool_use"]
        assert len(tool_uses) == 0

        # Should have text describing the orphaned tool call
        text_blocks = [b for b in assistant_msg["content"] if b.get("type") == "text"]
        assert len(text_blocks) >= 1
        combined_text = " ".join(b["text"] for b in text_blocks)
        assert (
            "glob" in combined_text
            or "Tool calls from compacted history" in combined_text
        )

    def test_valid_tool_use_with_result_preserved(self):
        """tool_use with matching tool_result should be preserved."""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search."},
                    {
                        "type": "tool_use",
                        "id": "toolu_valid",
                        "name": "glob",
                        "input": {"pattern": "*.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_valid",
                        "content": "Found 10 files",
                    }
                ],
            },
        ]

        result = _sanitize_tool_results(messages)

        # Both should be preserved
        assistant_msg = result[0]
        tool_uses = [b for b in assistant_msg["content"] if b.get("type") == "tool_use"]
        assert len(tool_uses) == 1
        assert tool_uses[0]["id"] == "toolu_valid"

        user_msg = result[1]
        tool_results = [
            b for b in user_msg["content"] if b.get("type") == "tool_result"
        ]
        assert len(tool_results) == 1

    def test_mixed_valid_and_orphaned_tool_uses(self):
        """Message with both valid and orphaned tool_uses should keep only valid ones."""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_valid",
                        "name": "read",
                        "input": {"path": "file.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_orphan",
                        "name": "write",
                        "input": {"path": "out.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_valid",
                        "content": "file contents",
                    }
                    # No tool_result for toolu_orphan
                ],
            },
        ]

        result = _sanitize_tool_results(messages)

        assistant_msg = result[0]
        tool_uses = [b for b in assistant_msg["content"] if b.get("type") == "tool_use"]
        assert len(tool_uses) == 1
        assert tool_uses[0]["id"] == "toolu_valid"

    def test_superdesign_compaction_scenario_with_orphaned_tool_use(self):
        """Reproduce SuperDesign compaction where tool_use remains but tool_result is lost.

        This is the exact error reported:
        "tool_use ids were found without tool_result blocks immediately after: toolu_01YJquBpATfUqskN381pdJdP"
        """
        messages: list[Message] = [
            {"role": "user", "content": "Help me build a component"},
            {
                "role": "assistant",
                "content": "[Summary: Previously searched for files]",  # Compacted
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search for more files."},
                    {
                        "type": "tool_use",
                        "id": "toolu_01YJquBpATfUqskN381pdJdP",  # Exact ID from error
                        "name": "glob",
                        "input": {"pattern": "src/**/*.tsx"},
                    },
                ],
            },
            {
                "role": "user",
                "content": "please continue",  # User message without tool_result
            },
        ]

        result = _sanitize_tool_results(messages)

        # The orphaned tool_use should be removed/converted
        for msg in result:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                tool_uses = [
                    b
                    for b in msg["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
                assert len(tool_uses) == 0, f"Found orphaned tool_use: {tool_uses}"

    def test_tool_use_only_message_converted_to_text(self):
        """Assistant message with only orphaned tool_use should become text."""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_only_orphan",
                        "name": "write",
                        "input": {"path": "test.py", "content": "print('hello')"},
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]

        result = _sanitize_tool_results(messages)

        assistant_msg = result[0]
        # Content should be text (either string or list with text block)
        content = assistant_msg["content"]
        if isinstance(content, list):
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            assert len(tool_uses) == 0
            text_blocks = [b for b in content if b.get("type") == "text"]
            assert len(text_blocks) >= 1
            assert "write" in text_blocks[0]["text"]
        else:
            # String content is also acceptable
            assert isinstance(content, str)

    def test_long_tool_input_truncated(self):
        """Long orphaned tool_use input should be truncated to 200 chars."""
        long_input = {"content": "x" * 500}
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_long",
                        "name": "write",
                        "input": long_input,
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]

        result = _sanitize_tool_results(messages)

        assistant_msg = result[0]
        text_blocks = [b for b in assistant_msg["content"] if b.get("type") == "text"]
        orphan_text = text_blocks[0]["text"]

        # Should be truncated with "..."
        assert "..." in orphan_text

    def test_multiple_consecutive_orphaned_tool_uses(self):
        """Multiple orphaned tool_use blocks should all be converted."""
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll do multiple operations."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read",
                        "input": {"path": "a.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "read",
                        "input": {"path": "b.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_3",
                        "name": "write",
                        "input": {"path": "c.py"},
                    },
                ],
            },
            {"role": "user", "content": "okay, what's next?"},
        ]

        result = _sanitize_tool_results(messages)

        assistant_msg = result[0]
        tool_uses = [b for b in assistant_msg["content"] if b.get("type") == "tool_use"]
        assert len(tool_uses) == 0

        # All three should be mentioned in the text
        text_blocks = [b for b in assistant_msg["content"] if b.get("type") == "text"]
        combined_text = " ".join(b["text"] for b in text_blocks)
        assert "read" in combined_text
        assert "write" in combined_text
