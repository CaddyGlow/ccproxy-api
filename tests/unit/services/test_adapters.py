"""Test adapter logic for format conversion between OpenAI and Anthropic APIs.

DISABLED: This module previously tested OpenAI adapter capabilities that were removed
during the refactoring to use the new ccproxy.llms.formatters system.

The tests included:
- OpenAI to Anthropic message format conversion
- Anthropic to OpenAI response format conversion
- System message handling
- Tool/function call conversion
- Image content conversion
- Streaming format conversion
- Edge cases and error handling

These adapter classes were removed and replaced with the new formatters system.
"""

# All tests in this file have been disabled because the OpenAIAdapter class
# and related adapters were removed during the refactoring to use the new
# ccproxy.llms.formatters system.

import pytest


@pytest.mark.skip(reason="OpenAI adapters removed in refactoring - tests disabled")
def test_adapters_disabled() -> None:
    """Placeholder test indicating the original adapter tests were disabled."""
    pass
