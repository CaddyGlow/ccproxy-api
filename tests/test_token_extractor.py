"""Test token extraction utilities."""

import pytest

from ccproxy.utils.token_extractor import (
    TokenUsage,
    TokenUsageAccumulator,
    extract_anthropic_usage,
    extract_claude_sdk_usage,
    extract_openai_usage,
    extract_usage_from_stream_event,
)


class TestTokenUsage:
    """Test TokenUsage data model."""

    def test_token_usage_defaults(self):
        """Test default values."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0
        assert usage.total_cost_usd is None

    def test_total_input_tokens(self):
        """Test total input tokens calculation."""
        usage = TokenUsage(
            input_tokens=100,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=25,
        )
        assert usage.total_input_tokens == 175

    def test_total_tokens(self):
        """Test total tokens calculation."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=200,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=25,
        )
        assert usage.total_tokens == 375


class TestAnthropicUsageExtraction:
    """Test Anthropic API usage extraction."""

    def test_extract_anthropic_usage_complete(self):
        """Test extracting complete usage data."""
        response = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 200,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 25,
            }
        }
        usage = extract_anthropic_usage(response)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200
        assert usage.cache_creation_input_tokens == 50
        assert usage.cache_read_input_tokens == 25

    def test_extract_anthropic_usage_partial(self):
        """Test extracting partial usage data."""
        response = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 200,
            }
        }
        usage = extract_anthropic_usage(response)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_extract_anthropic_usage_missing(self):
        """Test handling missing usage data."""
        response = {"id": "msg_123", "content": "Hello"}
        usage = extract_anthropic_usage(response)
        assert usage is None

    def test_extract_anthropic_usage_invalid(self):
        """Test handling invalid usage data."""
        response = {"usage": "invalid"}
        usage = extract_anthropic_usage(response)
        assert usage is None


class TestOpenAIUsageExtraction:
    """Test OpenAI API usage extraction."""

    def test_extract_openai_usage(self):
        """Test extracting OpenAI usage data."""
        response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
            }
        }
        usage = extract_openai_usage(response)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_extract_openai_usage_missing(self):
        """Test handling missing OpenAI usage data."""
        response = {"choices": [{"text": "Hello"}]}
        usage = extract_openai_usage(response)
        assert usage is None


class TestClaudeSDKUsageExtraction:
    """Test Claude SDK usage extraction."""

    def test_extract_claude_sdk_usage_full(self):
        """Test extracting full usage from ResultMessage."""

        class MockResultMessage:
            def __init__(self):
                self.usage = {
                    "input_tokens": 150,
                    "output_tokens": 250,
                    "cache_creation_tokens": 75,
                    "cache_read_tokens": 35,
                }
                self.total_cost_usd = 0.0125

        result_message = MockResultMessage()
        usage = extract_claude_sdk_usage(result_message)
        assert usage is not None
        assert usage.input_tokens == 150
        assert usage.output_tokens == 250
        assert usage.cache_creation_input_tokens == 75
        assert usage.cache_read_input_tokens == 35
        assert usage.total_cost_usd == 0.0125

    def test_extract_claude_sdk_usage_cost_only(self):
        """Test extracting cost only from ResultMessage."""

        class MockResultMessage:
            def __init__(self):
                self.usage = None
                self.total_cost_usd = 0.0125

        result_message = MockResultMessage()
        usage = extract_claude_sdk_usage(result_message)
        assert usage is not None
        assert usage.total_cost_usd == 0.0125
        assert usage.input_tokens == 0

    def test_extract_claude_sdk_usage_no_data(self):
        """Test handling ResultMessage with no usage data."""

        class MockResultMessage:
            pass

        result_message = MockResultMessage()
        usage = extract_claude_sdk_usage(result_message)
        assert usage is None


class TestStreamEventExtraction:
    """Test stream event usage extraction."""

    def test_extract_anthropic_stream_event(self):
        """Test extracting usage from Anthropic stream event."""
        event = {
            "type": "message_delta",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }
        usage = extract_usage_from_stream_event(event)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_openai_stream_event(self):
        """Test extracting usage from OpenAI stream event."""
        event = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }
        usage = extract_usage_from_stream_event(event)
        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50


class TestTokenUsageAccumulator:
    """Test token usage accumulator."""

    def test_accumulator_empty(self):
        """Test empty accumulator."""
        accumulator = TokenUsageAccumulator()
        usage = accumulator.get_usage()
        assert usage is None

    def test_accumulator_add_usage(self):
        """Test adding usage to accumulator."""
        accumulator = TokenUsageAccumulator()

        usage1 = TokenUsage(input_tokens=100, output_tokens=50)
        accumulator.add_usage(usage1)

        usage2 = TokenUsage(input_tokens=50, output_tokens=100)
        accumulator.add_usage(usage2)

        total = accumulator.get_usage()
        assert total is not None
        assert total.input_tokens == 150
        assert total.output_tokens == 150

    def test_accumulator_add_event(self):
        """Test adding events to accumulator."""
        accumulator = TokenUsageAccumulator()

        event1 = {
            "type": "message_delta",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        accumulator.add_event(event1)

        event2 = {"usage": {"prompt_tokens": 50, "completion_tokens": 100}}
        accumulator.add_event(event2)

        total = accumulator.get_usage()
        assert total is not None
        assert total.input_tokens == 150
        assert total.output_tokens == 150

    def test_accumulator_cost_override(self):
        """Test that cost is overridden, not accumulated."""
        accumulator = TokenUsageAccumulator()

        usage1 = TokenUsage(input_tokens=100, total_cost_usd=0.01)
        accumulator.add_usage(usage1)

        usage2 = TokenUsage(input_tokens=50, total_cost_usd=0.02)
        accumulator.add_usage(usage2)

        total = accumulator.get_usage()
        assert total is not None
        assert total.input_tokens == 150
        assert total.total_cost_usd == 0.02  # Last cost wins

    def test_accumulator_with_cache_tokens(self):
        """Test accumulating cache tokens."""
        accumulator = TokenUsageAccumulator()

        usage1 = TokenUsage(
            input_tokens=100,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=25,
        )
        accumulator.add_usage(usage1)

        usage2 = TokenUsage(
            input_tokens=50,
            cache_creation_input_tokens=30,
            cache_read_input_tokens=15,
        )
        accumulator.add_usage(usage2)

        total = accumulator.get_usage()
        assert total is not None
        assert total.input_tokens == 150
        assert total.cache_creation_input_tokens == 80
        assert total.cache_read_input_tokens == 40
        assert total.total_input_tokens == 270
