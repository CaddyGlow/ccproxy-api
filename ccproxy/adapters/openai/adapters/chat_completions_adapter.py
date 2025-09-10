"""OpenAI Chat Completions format adapter.

This adapter handles pure Chat Completions format operations without
cross-format conversion.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import BaseAPIAdapter
from ccproxy.adapters.openai.models.chat_completions import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)


class ChatCompletionsAdapter(BaseAPIAdapter):
    """Adapter for OpenAI Chat Completions API format."""

    def __init__(self) -> None:
        super().__init__("chat_completions")

    async def adapt_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Adapt Chat Completions request.

        For pure Chat Completions, this is mostly validation and normalization.

        Args:
            request_data: OpenAI Chat Completions request

        Returns:
            Validated Chat Completions request
        """
        # Validate using Pydantic model
        request = OpenAIChatCompletionRequest(**request_data)
        return request.model_dump(exclude_none=True)

    async def adapt_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Adapt Chat Completions response.

        For pure Chat Completions, this is mostly validation.

        Args:
            response_data: OpenAI Chat Completions response

        Returns:
            Validated Chat Completions response
        """
        # Validate using Pydantic model
        response = OpenAIChatCompletionResponse(**response_data)
        return response.model_dump(exclude_none=True)

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Adapt Chat Completions streaming response.

        For pure Chat Completions, this is passthrough with validation.

        Args:
            stream: Chat Completions streaming chunks

        Yields:
            Validated Chat Completions streaming chunks
        """
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Implementation of stream adaptation."""
        async for chunk in stream:
            # For pure Chat Completions, pass through chunks
            yield chunk


__all__ = ["ChatCompletionsAdapter"]
