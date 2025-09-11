"""Anthropic to OpenAI format adapter - unidirectional conversion."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import APIAdapter
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class AnthropicToOpenAIAdapter(APIAdapter):
    """Converts Anthropic format responses to OpenAI format - unidirectional."""

    def __init__(self) -> None:
        """Initialize the Anthropic to OpenAI adapter."""
        pass

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic request format to OpenAI format.

        Args:
            response: Anthropic format response

        Returns:
            OpenAI format response
        """
        raise NotImplementedError()

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI response format to Anthropic format.

        Args:
            response: OpenAI format response

        Returns:
            Antrhopic format response
        """
        raise NotImplementedError()

    def adapt_error(self, error_response: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI to Anthropic error format.

        Args:
            error_response: error response

        Returns:
            OpenAI error response
        """
        raise NotImplementedError()

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert OpenAI streaming data to Anthropic streaming format.

        Args:
            stream_data: OpenAI streaming event data

        Returns:
            Anthropic streaming event data
        """
        raise NotImplementedError()
