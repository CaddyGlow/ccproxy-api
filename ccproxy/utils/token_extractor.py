"""Token usage extraction utilities for various API response formats."""

import logging
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    """Unified token usage data model."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0)

    @property
    def total_input_tokens(self) -> int:
        """Calculate total input tokens including cache tokens."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.total_input_tokens + self.output_tokens


def extract_anthropic_usage(response: dict[str, Any]) -> TokenUsage | None:
    """Extract token usage from Anthropic API response.

    Args:
        response: Anthropic API response dict

    Returns:
        TokenUsage object if usage data found, None otherwise
    """
    try:
        usage = response.get("usage")
        if not usage:
            return None

        return TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        )
    except Exception as e:
        logger.warning(f"Failed to extract Anthropic usage: {e}")
        return None


def extract_openai_usage(response: dict[str, Any]) -> TokenUsage | None:
    """Extract token usage from OpenAI API response.

    OpenAI format uses prompt_tokens and completion_tokens.
    We map these to our unified format.

    Args:
        response: OpenAI API response dict

    Returns:
        TokenUsage object if usage data found, None otherwise
    """
    try:
        usage = response.get("usage")
        if not usage:
            return None

        return TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            # OpenAI doesn't have cache tokens
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    except Exception as e:
        logger.warning(f"Failed to extract OpenAI usage: {e}")
        return None


def extract_claude_sdk_usage(result_message: Any) -> TokenUsage | None:
    """Extract token usage from Claude Code SDK ResultMessage.

    Args:
        result_message: ResultMessage from Claude Code SDK

    Returns:
        TokenUsage object if usage data found, None otherwise
    """
    try:
        if not hasattr(result_message, "usage") or not result_message.usage:
            # If no usage data but we have cost, return with just cost
            if (
                hasattr(result_message, "total_cost_usd")
                and result_message.total_cost_usd
            ):
                return TokenUsage(total_cost_usd=result_message.total_cost_usd)
            return None

        usage = result_message.usage
        token_usage = TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_tokens", 0),
        )

        # Add cost if available
        if hasattr(result_message, "total_cost_usd") and result_message.total_cost_usd:
            token_usage.total_cost_usd = result_message.total_cost_usd

        return token_usage
    except Exception as e:
        logger.warning(f"Failed to extract Claude SDK usage: {e}")
        return None


def extract_usage_from_stream_event(event: dict[str, Any]) -> TokenUsage | None:
    """Extract token usage from a streaming event.

    Handles both Anthropic and OpenAI streaming formats.

    Args:
        event: Streaming event dict

    Returns:
        TokenUsage object if usage data found, None otherwise
    """
    try:
        # Anthropic stream format
        if event.get("type") == "message_delta" and "usage" in event:
            return extract_anthropic_usage({"usage": event["usage"]})

        # OpenAI stream format - usage typically comes in the final chunk
        if "usage" in event:
            return extract_openai_usage({"usage": event["usage"]})

        return None
    except Exception as e:
        logger.warning(f"Failed to extract usage from stream event: {e}")
        return None


class TokenUsageAccumulator:
    """Accumulate token usage across streaming events."""

    def __init__(self) -> None:
        """Initialize the accumulator."""
        self.total_usage = TokenUsage()
        self._has_data = False

    def add_usage(self, usage: TokenUsage | None) -> None:
        """Add usage data to the accumulator.

        Args:
            usage: Token usage to add
        """
        if not usage:
            return

        self._has_data = True
        self.total_usage.input_tokens += usage.input_tokens
        self.total_usage.output_tokens += usage.output_tokens
        self.total_usage.cache_creation_input_tokens += (
            usage.cache_creation_input_tokens
        )
        self.total_usage.cache_read_input_tokens += usage.cache_read_input_tokens

        # For cost, we typically want the final cost, not accumulated
        if usage.total_cost_usd is not None:
            self.total_usage.total_cost_usd = usage.total_cost_usd

    def add_event(self, event: dict[str, Any]) -> None:
        """Add usage from a streaming event.

        Args:
            event: Streaming event dict
        """
        usage = extract_usage_from_stream_event(event)
        self.add_usage(usage)

    def get_usage(self) -> TokenUsage | None:
        """Get accumulated usage.

        Returns:
            Accumulated token usage or None if no data
        """
        return self.total_usage if self._has_data else None
