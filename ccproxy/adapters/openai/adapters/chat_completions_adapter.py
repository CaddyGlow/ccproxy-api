"""OpenAI Chat Completions format adapter.

This adapter handles pure Chat Completions format operations without
cross-format conversion.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import BaseAPIAdapter
from ccproxy.adapters.openai.models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)


class ChatCompletionsAdapter(BaseAPIAdapter):
    """Adapter for OpenAI Chat Completions API format."""

    def __init__(self) -> None:
        super().__init__("chat_completions")

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Adapt Chat Completions request.

        For pure Chat Completions, this is mostly validation and normalization.

        Args:
            request_data: OpenAI Chat Completions request

        Returns:
            Validated Chat Completions request
        """
        # Validate using Pydantic model
        request_validated = OpenAIChatCompletionRequest.model_validate(request)
        result = request_validated.model_dump(exclude_none=True)
        return dict(result)

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Adapt Chat Completions response.

        For pure Chat Completions, this is mostly validation.

        Args:
            response_data: OpenAI Chat Completions response

        Returns:
            Validated Chat Completions response
        """
        # Validate using Pydantic model
        response_validated = OpenAIChatCompletionResponse.model_validate(response)
        result = response_validated.model_dump(exclude_none=True)
        return dict(result)

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
        emitted_role = False
        stream_id: str | None = None
        model_name: str | None = None

        async for chunk in stream:
            # Capture id/model when available
            if isinstance(chunk, dict):
                stream_id = stream_id or chunk.get("id")
                model_name = model_name or chunk.get("model")

            # Detect Copilot prelude style chunk that lacks OpenAI fields
            is_prelude = False
            if isinstance(chunk, dict):
                has_choices = bool(chunk.get("choices"))
                has_model = bool(chunk.get("model"))
                # Copilot often sends prompt_filter_results with empty choices first
                is_prelude = (not has_choices) and (
                    "prompt_filter_results" in chunk or not has_model
                )

            # If prelude, synthesize an initial role chunk and skip prelude payload
            if is_prelude and not emitted_role:
                synth_chunk = {
                    "id": stream_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name or "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                emitted_role = True
                yield synth_chunk
                # Skip yielding the prelude itself
                continue

            # Ensure we emit a role chunk before the first content delta if needed
            if not emitted_role and isinstance(chunk, dict):
                # If this first real chunk doesn't include a role, emit one ahead
                choices = (
                    chunk.get("choices")
                    if isinstance(chunk.get("choices"), list)
                    else []
                )
                first_delta = (
                    (choices[0].get("delta") if choices else None)
                    if isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                    else None
                )
                has_role_in_delta = bool(first_delta and first_delta.get("role"))
                if not has_role_in_delta:
                    synth_chunk = {
                        "id": stream_id
                        or chunk.get("id")
                        or f"chatcmpl-{uuid.uuid4().hex[:24]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name or chunk.get("model") or "gpt-4o",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    emitted_role = True
                    yield synth_chunk

            # Mark role emitted if this chunk carries it
            if isinstance(chunk, dict):
                choices = (
                    chunk.get("choices")
                    if isinstance(chunk.get("choices"), list)
                    else []
                )
                if (
                    isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                    and isinstance(choices[0].get("delta"), dict)
                    and choices[0]["delta"].get("role") == "assistant"
                ):
                    emitted_role = True

            # Yield the original chunk
            yield chunk

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Convert error format - pass through for Chat Completions.

        Args:
            error: Error response

        Returns:
            Pass-through error response
        """
        # For pure Chat Completions, pass through errors
        return error


__all__ = ["ChatCompletionsAdapter"]
