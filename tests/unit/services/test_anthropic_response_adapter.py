"""Unit tests for AnthropicResponseAPIAdapter.

DISABLED: This module previously tested AnthropicResponseAPIAdapter which was removed
during the refactoring to use the new ccproxy.llms.formatters system.

The tests covered:
- Request conversion (messages → input, system → instructions, fields passthrough)
- Response conversion from both nested `response.output` and `choices` styles
- Streaming conversion for response.output_text.delta and response.done
- Internal helper: messages → input adds required fields
"""

# All tests in this file have been disabled because the AnthropicResponseAPIAdapter
# class was removed during the refactoring to use the new ccproxy.llms.formatters system.

import pytest


@pytest.mark.skip(
    reason="AnthropicResponseAPIAdapter removed in refactoring - tests disabled"
)
def test_anthropic_response_adapter_disabled():
    """Placeholder test indicating the original adapter tests were disabled."""
    pass
