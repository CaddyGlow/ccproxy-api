"""Tests for message models."""

import pytest
from pydantic import ValidationError

from claude_code_proxy.models.messages import MessageRequest


class TestMessageRequest:
    """Test MessageRequest model validation."""

    def test_max_tokens_validation(self):
        """Test max_tokens validation allows large values."""
        # Test that large values like 64000 are accepted
        request = MessageRequest(  # type: ignore[call-arg]
            model="claude-3-5-sonnet-20241022",
            max_tokens=64000,
            messages=[{"role": "user", "content": "Hello"}],  # type: ignore[list-item]
        )
        assert request.max_tokens == 64000

        # Test maximum allowed value
        request = MessageRequest(  # type: ignore[call-arg]
            model="claude-3-5-sonnet-20241022",
            max_tokens=200000,
            messages=[{"role": "user", "content": "Hello"}],  # type: ignore[list-item]
        )
        assert request.max_tokens == 200000

        # Test exceeding maximum should fail
        with pytest.raises(ValidationError):
            MessageRequest(  # type: ignore[call-arg]
                model="claude-3-5-sonnet-20241022",
                max_tokens=200001,
                messages=[{"role": "user", "content": "Hello"}],  # type: ignore[list-item]
            )

        # Test minimum value validation (should fail)
        with pytest.raises(ValidationError):
            MessageRequest(  # type: ignore[call-arg]
                model="claude-3-5-sonnet-20241022",
                max_tokens=0,
                messages=[{"role": "user", "content": "Hello"}],  # type: ignore[list-item]
            )
