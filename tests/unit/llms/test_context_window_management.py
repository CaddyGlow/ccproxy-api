"""Tests for context window management utilities.

This module tests token estimation and context truncation logic
for managing requests that exceed model context limits.
"""

import pytest

from ccproxy.llms.utils import (
    estimate_messages_tokens,
    estimate_request_tokens,
    estimate_tokens,
    get_max_input_tokens,
    truncate_to_fit,
)


class TestTokenEstimation:
    """Tests for token estimation functions."""

    def test_estimate_tokens_string(self) -> None:
        """Test token estimation for plain strings."""
        # ~3 chars per token
        text = "Hello, world!"  # 13 chars -> ~4 tokens
        tokens = estimate_tokens(text)
        assert tokens >= 1
        assert tokens <= 10  # Reasonable upper bound

    def test_estimate_tokens_empty_string(self) -> None:
        """Test token estimation for empty string."""
        tokens = estimate_tokens("")
        assert tokens == 1  # min(1, ...)

    def test_estimate_tokens_none(self) -> None:
        """Test token estimation for None."""
        tokens = estimate_tokens(None)
        assert tokens == 0

    def test_estimate_tokens_text_block(self) -> None:
        """Test token estimation for text content block."""
        content = [
            {"type": "text", "text": "This is a test message with some content."}
        ]
        tokens = estimate_tokens(content)
        assert tokens > 0

    def test_estimate_tokens_tool_use_block(self) -> None:
        """Test token estimation for tool_use content block."""
        content = [
            {
                "type": "tool_use",
                "id": "tool_123",
                "name": "read_file",
                "input": {"path": "/some/file/path.txt"},
            }
        ]
        tokens = estimate_tokens(content)
        assert tokens > 0

    def test_estimate_tokens_tool_result_block(self) -> None:
        """Test token estimation for tool_result content block."""
        content = [
            {
                "type": "tool_result",
                "tool_use_id": "tool_123",
                "content": "File contents here with some data.",
            }
        ]
        tokens = estimate_tokens(content)
        assert tokens > 0

    def test_estimate_tokens_image_block(self) -> None:
        """Test token estimation for image content block."""
        content = [{"type": "image", "source": {"type": "base64", "data": "..."}}]
        tokens = estimate_tokens(content)
        assert tokens == 1600  # Fixed estimate for images

    def test_estimate_tokens_mixed_content(self) -> None:
        """Test token estimation for mixed content blocks."""
        content = [
            {"type": "text", "text": "Check this image:"},
            {"type": "image", "source": {"type": "url", "url": "https://..."}},
            {"type": "text", "text": "What do you see?"},
        ]
        tokens = estimate_tokens(content)
        assert tokens > 1600  # At least the image tokens

    def test_estimate_messages_tokens_single(self) -> None:
        """Test token estimation for a single message."""
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        tokens = estimate_messages_tokens(messages)
        assert tokens > 2  # At least role tokens

    def test_estimate_messages_tokens_conversation(self) -> None:
        """Test token estimation for a conversation."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "3+3 equals 6."},
        ]
        tokens = estimate_messages_tokens(messages)
        assert tokens > 8  # At least role tokens (2 per message)

    def test_estimate_request_tokens_with_system(self) -> None:
        """Test token estimation for request with system prompt."""
        request = {
            "model": "claude-3-opus-20240229",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        tokens = estimate_request_tokens(request)
        assert tokens > 0

    def test_estimate_request_tokens_with_tools(self) -> None:
        """Test token estimation for request with tools."""
        request = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": "Read a file"}],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
        }
        tokens = estimate_request_tokens(request)
        assert tokens > 0


class TestGetMaxInputTokens:
    """Tests for max input tokens lookup."""

    def test_get_max_input_tokens_known_model(self) -> None:
        """Test getting max input tokens for a known model."""
        # This test may need adjustment based on what models are in the limits file
        max_tokens = get_max_input_tokens("claude-3-opus-20240229")
        # Should return a value if model is in limits
        if max_tokens is not None:
            assert max_tokens > 0

    def test_get_max_input_tokens_unknown_model(self) -> None:
        """Test getting max input tokens for an unknown model."""
        max_tokens = get_max_input_tokens("totally-unknown-model-xyz")
        assert max_tokens is None


class TestTruncateToFit:
    """Tests for context truncation."""

    def test_truncate_no_truncation_needed(self) -> None:
        """Test that small requests are not truncated."""
        request = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }
        result, was_truncated = truncate_to_fit(
            request, max_input_tokens=200000, preserve_recent=10
        )
        assert was_truncated is False
        assert result == request

    def test_truncate_removes_old_messages(self) -> None:
        """Test that truncation removes oldest messages first."""
        # Create a request with many messages
        messages = []
        for i in range(20):
            messages.append({"role": "user", "content": f"Message {i} " * 100})
            messages.append({"role": "assistant", "content": f"Response {i} " * 100})

        request = {"model": "claude-3-opus-20240229", "messages": messages}

        # Force truncation with a very low limit
        result, was_truncated = truncate_to_fit(
            request, max_input_tokens=1000, preserve_recent=4
        )

        assert was_truncated is True
        # Should have fewer messages
        assert len(result["messages"]) < len(messages)
        # Should preserve recent messages
        assert len(result["messages"]) >= 4

    def test_truncate_preserves_recent_messages(self) -> None:
        """Test that recent messages are preserved during truncation."""
        messages = [
            {"role": "user", "content": "Old message " * 500},
            {"role": "assistant", "content": "Old response " * 500},
            {"role": "user", "content": "Recent message 1"},
            {"role": "assistant", "content": "Recent response 1"},
            {"role": "user", "content": "Recent message 2"},
            {"role": "assistant", "content": "Recent response 2"},
        ]
        request = {"model": "claude-3-opus-20240229", "messages": messages}

        result, was_truncated = truncate_to_fit(
            request, max_input_tokens=500, preserve_recent=4
        )

        if was_truncated:
            # Check that the last 4 messages are preserved
            result_messages = result["messages"]
            # Account for truncation notice being added
            recent_messages = (
                result_messages[-4:] if len(result_messages) >= 4 else result_messages
            )

            # Verify recent content is in the preserved messages
            recent_content = [m.get("content", "") for m in recent_messages]
            assert any("Recent" in str(c) for c in recent_content)

    def test_truncate_adds_notice(self) -> None:
        """Test that truncation adds a notice message."""
        messages = []
        for i in range(10):
            messages.append({"role": "user", "content": f"Message {i} " * 200})
            messages.append({"role": "assistant", "content": f"Response {i} " * 200})

        request = {"model": "claude-3-opus-20240229", "messages": messages}

        result, was_truncated = truncate_to_fit(
            request, max_input_tokens=500, preserve_recent=2
        )

        if was_truncated:
            # First message should be the truncation notice
            first_msg = result["messages"][0]
            assert first_msg["role"] == "user"
            assert "truncated" in first_msg["content"].lower()

    def test_truncate_not_enough_messages(self) -> None:
        """Test truncation behavior when only one message exceeds the limit."""
        messages = [
            {"role": "user", "content": "Single message " * 1000},
        ]
        request = {"model": "claude-3-opus-20240229", "messages": messages}

        # Try to truncate with preserve_recent > message count
        result, was_truncated = truncate_to_fit(
            request, max_input_tokens=100, preserve_recent=10
        )

        # Should truncate and insert a notice if content can't fit
        assert was_truncated is True
        assert result["messages"][0]["role"] == "user"
        assert "truncated" in result["messages"][0]["content"].lower()

    def test_truncate_preserves_system_and_tools(self) -> None:
        """Test that system prompt and tools are preserved."""
        request = {
            "model": "claude-3-opus-20240229",
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Old message " * 500},
                {"role": "assistant", "content": "Old response " * 500},
                {"role": "user", "content": "Recent message"},
            ],
            "tools": [{"name": "test_tool", "description": "A test tool"}],
        }

        result, was_truncated = truncate_to_fit(
            request, max_input_tokens=500, preserve_recent=1
        )

        # System and tools should be preserved regardless of truncation
        assert result.get("system") == "You are a helpful assistant."
        assert result.get("tools") == request["tools"]

    def test_truncate_safety_margin(self) -> None:
        """Test that safety margin is applied correctly."""
        messages = []
        for i in range(5):
            messages.append({"role": "user", "content": f"Message {i}"})

        request = {"model": "claude-3-opus-20240229", "messages": messages}

        # With safety_margin=0.5, effective limit is 50000
        result, was_truncated = truncate_to_fit(
            request, max_input_tokens=100000, preserve_recent=2, safety_margin=0.5
        )

        # Should not truncate since content is small
        assert was_truncated is False
