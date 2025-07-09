"""Legacy tests for message models - kept for backward compatibility.

Note: Most comprehensive tests are now in test_message_models.py
This file contains only specific edge case tests not covered elsewhere.
"""

import pytest
from pydantic import ValidationError

from ccproxy.models.messages import MessageCreateParams
from ccproxy.models.requests import Message


@pytest.mark.unit
class TestMessageCreateParamsEdgeCases:
    """Test MessageCreateParams edge cases not covered in main tests."""

    def test_large_max_tokens_values(self):
        """Test that large max_tokens values like 64000 are accepted."""
        # This tests a specific edge case for large token values
        request = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            max_tokens=64000,
            messages=[Message(role="user", content="Hello")],
        )
        assert request.max_tokens == 64000

    def test_temperature_edge_values(self):
        """Test temperature at boundary values."""
        # Test at 0.0
        request = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            temperature=0.0,
        )
        assert request.temperature == 0.0

        # Test at 1.0
        request = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            temperature=1.0,
        )
        assert request.temperature == 1.0

    def test_top_p_and_top_k_edge_values(self):
        """Test top_p and top_k at boundary values."""
        request = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            top_p=0.0,
            top_k=0,
        )
        assert request.top_p == 0.0
        assert request.top_k == 0
