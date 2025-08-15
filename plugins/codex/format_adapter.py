"""Codex format adapter for OpenAI conversion."""

from collections.abc import AsyncIterator
from typing import Any

import structlog

from ccproxy.adapters.base import APIAdapter
from ccproxy.adapters.openai.response_adapter import ResponseAdapter


logger = structlog.get_logger(__name__)


class CodexFormatAdapter(APIAdapter):
    """Handles format conversion between OpenAI Chat Completions and Codex Response API formats.

    This adapter delegates to the ResponseAdapter which knows how to:
    1. Convert Chat Completions → Response API format (for requests)
    2. Convert Response API → Chat Completions format (for responses)
    3. Handle SSE streaming conversion (for streaming responses)
    """

    def __init__(self) -> None:
        """Initialize the format adapter."""
        self._response_adapter = ResponseAdapter()

    async def adapt_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Chat Completions request to Response API format.

        Args:
            request_data: OpenAI Chat Completions request

        Returns:
            Codex Response API formatted request
        """
        if "messages" in request_data:
            # Use ResponseAdapter to convert Chat Completions → Response API
            logger.debug("converting_chat_completions_to_response_api")
            response_request = self._response_adapter.chat_to_response_request(
                request_data
            )
            codex_request = response_request.model_dump()

            # Ensure Codex-specific defaults
            if "model" not in codex_request:
                codex_request["model"] = "gpt-5"

            logger.info(
                "codex_request_conversion",
                original_keys=list(request_data.keys()),
                converted_keys=list(codex_request.keys()),
            )
            return codex_request

        # Native Response API format - passthrough
        logger.info("codex_request_passthrough", request_keys=list(request_data.keys()))
        return request_data

    async def adapt_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API response to Chat Completions format.

        Args:
            response_data: Codex Response API response

        Returns:
            OpenAI Chat Completions formatted response
        """
        # Check if this is a Response API format response
        if self._is_response_api_format(response_data):
            logger.debug("converting_response_api_to_chat_completions")
            chat_response = self._response_adapter.response_to_chat_completion(
                response_data
            )
            return chat_response.model_dump()

        return response_data

    async def adapt_stream(  # type: ignore[override,misc]
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Convert individual Response API events to Chat Completions format.

        Uses the same pattern as OpenAI adapter's streaming processor.

        Args:
            stream: Individual Response API events (already parsed by ProxyService)

        Yields:
            OpenAI Chat Completions streaming chunks
        """
        # Extract the stream processing logic from OpenAI adapter pattern
        import time

        from ccproxy.adapters.openai.models import generate_openai_response_id

        message_id = generate_openai_response_id()
        created = int(time.time())
        role_sent = False

        logger.info("codex_stream_processing_started", message_id=message_id)

        async for event in stream:
            logger.debug(
                "processing_response_event",
                event_type=event.get("type"),
                message_id=message_id,
            )

            # Send initial role chunk if not sent yet
            if not role_sent:
                yield {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                role_sent = True

            # Convert Response API events to ChatCompletion deltas
            chunk = self._convert_response_event_to_chat_delta(
                event, message_id, created
            )
            if chunk:
                logger.debug("yielding_chat_chunk", message_id=message_id)
                yield chunk

    def _convert_response_event_to_chat_delta(
        self, event: dict[str, Any], stream_id: str, created: int
    ) -> dict[str, Any] | None:
        """Convert a Response API event to ChatCompletion delta format."""
        event_type = event.get("type", "")

        # Handle content deltas (main text output)
        if event_type == "response.output_text.delta":
            delta_text = event.get("delta", "")
            if delta_text:
                return {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None,
                        }
                    ],
                }

        # Handle structured output deltas
        elif event_type == "response.output.delta":
            # Extract content from nested output structure
            output = event.get("output", [])
            delta_content = ""

            for output_item in output:
                if output_item.get("type") == "message":
                    content_blocks = output_item.get("content", [])
                    for block in content_blocks:
                        if block.get("type") in ["output_text", "text"]:
                            delta_content += block.get("text", "")

            if delta_content:
                return {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_content},
                            "finish_reason": None,
                        }
                    ],
                }

        # Handle completion events
        elif event_type == "response.completed":
            response = event.get("response", {})
            usage = response.get("usage")

            chunk_data = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.get("model", "gpt-5"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            # Add usage if available
            if usage:
                chunk_data["usage"] = {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            return chunk_data

        # Skip other event types
        logger.debug("skipping_event_type", event_type=event_type)
        return None

    def _is_response_api_format(self, response_data: dict[str, Any]) -> bool:
        """Check if response is in Response API format (used by Codex)."""
        # Response API responses have 'output' field or are wrapped in 'response'
        return "output" in response_data or "response" in response_data
