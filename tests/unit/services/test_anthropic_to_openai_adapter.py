"""Unit tests for OpenAIToAnthropicAdapter.

DISABLED: This module previously tested OpenAIToAnthropicAdapter which was removed
during the refactoring to use the new ccproxy.llms.formatters system.

The tests followed TESTING.md guidelines:
- Fast unit tests with proper type annotations
- Mock at service boundaries only
- Test real internal behavior
- Use essential fixtures from conftest.py
"""

# All tests in this file have been disabled because the OpenAIToAnthropicAdapter
# class was removed during the refactoring to use the new ccproxy.llms.formatters system.

import pytest


@pytest.mark.skip(
    reason="OpenAIToAnthropicAdapter removed in refactoring - tests disabled"
)
def test_openai_to_anthropic_adapter_disabled():
    """Placeholder test indicating the original adapter tests were disabled."""
    pass
