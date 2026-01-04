"""Mock response handler for bypass mode."""

import asyncio
import json
import random
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi.responses import StreamingResponse

from ccproxy.core.request_context import RequestContext
from ccproxy.services.adapters.format_adapter import DictFormatAdapter
from ccproxy.services.adapters.simple_converters import (
    convert_anthropic_to_openai_response,
)
from ccproxy.testing import RealisticMockResponseGenerator


logger = structlog.get_logger(__name__)


class MockResponseHandler:
    """Handles bypass mode with realistic mock responses."""

    def __init__(
        self,
        mock_generator: RealisticMockResponseGenerator,
        openai_adapter: DictFormatAdapter | None = None,
        error_rate: float = 0.05,
        latency_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        """Initialize with mock generator and format adapter.

        - Uses existing testing utilities
        - Supports both Anthropic and OpenAI formats
        """
        self.mock_generator = mock_generator
        if openai_adapter is None:
            openai_adapter = DictFormatAdapter(
                response=convert_anthropic_to_openai_response,
                name="mock_anthropic_to_openai",
            )
        self.openai_adapter = openai_adapter
        self.error_rate = error_rate
        self.latency_range = latency_range

    def extract_message_type(self, body: bytes | None) -> str:
        """Analyze request body to determine response type.

        - Checks for 'tools' field → returns 'tool_use'
        - Analyzes message length → returns 'long'|'medium'|'short'
        - Handles JSON decode errors gracefully
        """
        if not body:
            return "short"

        try:
            data = json.loads(body)

            # Check for tool use
            if "tools" in data:
                return "tool_use"

            # Analyze message content length
            messages = data.get("messages", [])
            if messages:
                total_content_length = sum(
                    len(msg.get("content", ""))
                    for msg in messages
                    if isinstance(msg.get("content"), str)
                )

                if total_content_length > 1000:
                    return "long"
                elif total_content_length > 200:
                    return "medium"

            return "short"

        except (json.JSONDecodeError, TypeError):
            return "short"

    def should_simulate_error(self) -> bool:
        """Randomly decide if error should be simulated.

        - Uses configuration-based error rate
        - Provides realistic error distribution
        """
        return random.random() < self.error_rate

    async def generate_standard_response(
        self,
        model: str | None,
        is_openai_format: bool,
        ctx: RequestContext,
        message_type: str = "short",
    ) -> tuple[int, dict[str, str], bytes]:
        """Generate non-streaming mock response.

        - Simulates realistic latency (configurable)
        - Generates appropriate token counts
        - Updates request context with metrics
        - Returns (status_code, headers, body)
        """
        # Simulate latency
        latency = random.uniform(*self.latency_range)
        await asyncio.sleep(latency)

        # Check if we should simulate an error
        if self.should_simulate_error():
            error_response = self._generate_error_response(is_openai_format)
            return 429, {"content-type": "application/json"}, error_response

        # Generate mock response based on type
        if message_type == "tool_use":
            mock_response = self.mock_generator.generate_tool_use_response(model=model)
        elif message_type == "long":
            mock_response = self.mock_generator.generate_long_response(model=model)
        elif message_type == "medium":
            mock_response = self.mock_generator.generate_medium_response(model=model)
        else:
            mock_response = self.mock_generator.generate_short_response(model=model)

        # Convert to OpenAI format if needed
        if is_openai_format and message_type != "tool_use":
            # Use dict-based conversion
            mock_response = await self.openai_adapter.convert_response(mock_response)

        # Update context with metrics
        if ctx:
            ctx.metrics["mock_response_type"] = message_type
            ctx.metrics["mock_latency_ms"] = int(latency * 1000)

        headers = {
            "content-type": "application/json",
            "x-request-id": ctx.request_id if ctx else "mock-request",
        }

        return 200, headers, json.dumps(mock_response).encode()

    async def generate_streaming_response(
        self,
        model: str | None,
        is_openai_format: bool,
        ctx: RequestContext,
        message_type: str = "short",
    ) -> StreamingResponse:
        """Generate SSE streaming mock response.

        - Simulates realistic token generation rate
        - Properly formatted SSE events
        - Includes [DONE] marker
        """

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            # Generate base response
            if message_type == "tool_use":
                base_response = self.mock_generator.generate_tool_use_response(
                    model=model
                )
            elif message_type == "long":
                base_response = self.mock_generator.generate_long_response(model=model)
            else:
                base_response = self.mock_generator.generate_short_response(model=model)

            content = base_response.get("content", [{"text": "Mock response"}])
            if isinstance(content, list) and content:
                text_content = content[0].get("text", "Mock response")
            else:
                text_content = "Mock response"

            # Split content into chunks
            words = text_content.split()
            chunk_size = 3  # Words per chunk

            # Send initial event
            if is_openai_format:
                initial_event = {
                    "id": f"chatcmpl-{ctx.request_id if ctx else 'mock'}",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": model or "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(initial_event)}\n\n".encode()
            else:
                initial_event = {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{ctx.request_id if ctx else 'mock'}",
                        "type": "message",
                        "role": "assistant",
                        "model": model or "claude-3-opus-20240229",
                        "content": [],
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                }
                yield f"data: {json.dumps(initial_event)}\n\n".encode()

            # Stream content chunks
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i : i + chunk_size]
                chunk_text = " ".join(chunk_words)
                if i + chunk_size < len(words):
                    chunk_text += " "

                await asyncio.sleep(0.05)  # Simulate token generation delay

                if is_openai_format:
                    chunk_event = {
                        "id": f"chatcmpl-{ctx.request_id if ctx else 'mock'}",
                        "object": "chat.completion.chunk",
                        "created": 1234567890,
                        "model": model or "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                else:
                    chunk_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": chunk_text},
                    }

                yield f"data: {json.dumps(chunk_event)}\n\n".encode()

            # Send final event
            if is_openai_format:
                final_event = {
                    "id": f"chatcmpl-{ctx.request_id if ctx else 'mock'}",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": model or "gpt-4",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_event)}\n\n".encode()
            else:
                final_event = {
                    "type": "message_stop",
                    "message": {
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": len(text_content.split()),
                        }
                    },
                }
                yield f"data: {json.dumps(final_event)}\n\n".encode()

            # Send [DONE] marker
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Request-ID": ctx.request_id if ctx else "mock-request",
            },
        )

    def _generate_error_response(self, is_openai_format: bool) -> bytes:
        """Generate a mock error response."""
        if is_openai_format:
            error: dict[str, Any] = {
                "error": {
                    "message": "Rate limit exceeded (mock error)",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            }
        else:
            error = {
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit exceeded (mock error)",
                },
            }
        return json.dumps(error).encode()
