"""Mock response handler for bypass mode."""

import asyncio
import json
import random
from collections.abc import AsyncGenerator
from time import time
from typing import Any, TypeAlias

import structlog
from fastapi.responses import StreamingResponse

from ccproxy.core.constants import (
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.core.request_context import RequestContext
from ccproxy.services.adapters.format_adapter import DictFormatAdapter
from ccproxy.services.adapters.simple_converters import (
    convert_anthropic_to_openai_response,
    convert_anthropic_to_openai_responses_response,
)
from ccproxy.testing import RealisticMockResponseGenerator


logger = structlog.get_logger(__name__)

PROMPT_EXTRACTION_KEYS = ("instructions", "content", "text", "input", "messages")
MAX_PROMPT_EXTRACTION_DEPTH = 10

TargetFormat: TypeAlias = str
PromptValue: TypeAlias = str | list[Any] | dict[str, Any] | int | float | bool | None


class MockResponseHandler:
    """Handles bypass mode with realistic mock responses."""

    def __init__(
        self,
        mock_generator: RealisticMockResponseGenerator,
        openai_adapter: DictFormatAdapter | None = None,
        openai_responses_adapter: DictFormatAdapter | None = None,
        error_rate: float = 0.05,
        latency_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        """Initialize with mock generator and format adapter.

        - Uses existing testing utilities
        - Supports both Anthropic and OpenAI formats
        """
        self.mock_generator = mock_generator
        if openai_adapter is None:
            openai_adapter = self._create_openai_adapter()
        self.openai_adapter = openai_adapter
        if openai_responses_adapter is None:
            openai_responses_adapter = self._create_openai_responses_adapter()
        self.openai_responses_adapter = openai_responses_adapter
        self.error_rate = error_rate
        self.latency_range = latency_range

    @staticmethod
    def _create_openai_adapter() -> DictFormatAdapter:
        """Create the adapter used for Anthropic -> OpenAI chat mocks."""

        return DictFormatAdapter(
            response=convert_anthropic_to_openai_response,
            name="mock_anthropic_to_openai",
        )

    @staticmethod
    def _create_openai_responses_adapter() -> DictFormatAdapter:
        """Create the adapter used for Anthropic -> OpenAI responses mocks."""

        return DictFormatAdapter(
            response=convert_anthropic_to_openai_responses_response,
            name="mock_anthropic_to_openai_responses",
        )

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

    def extract_prompt_text(self, body: bytes | None) -> str:
        """Extract a plain-text prompt summary from common request shapes."""

        if not body:
            return ""

        try:
            data: PromptValue = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return ""

        parts: list[str] = []

        seen: set[int] = set()

        def collect(value: PromptValue, depth: int = 0) -> None:
            if depth > MAX_PROMPT_EXTRACTION_DEPTH:
                logger.debug(
                    "prompt_extraction_max_depth_reached",
                    depth=depth,
                    max_depth=MAX_PROMPT_EXTRACTION_DEPTH,
                )
                return

            if isinstance(value, str):
                text = value.strip()
                if text:
                    parts.append(text)
                return

            if isinstance(value, list):
                value_id = id(value)
                if value_id in seen:
                    return
                seen.add(value_id)
                for item in value:
                    collect(item, depth + 1)
                return

            if not isinstance(value, dict):
                return

            value_id = id(value)
            if value_id in seen:
                return
            seen.add(value_id)

            for key in PROMPT_EXTRACTION_KEYS:
                if key not in value:
                    continue
                try:
                    collect(value[key], depth + 1)
                except (KeyError, TypeError, AttributeError) as exc:
                    logger.debug(
                        "prompt_extraction_value_skipped",
                        key=key,
                        error=str(exc),
                    )

        collect(data)
        return "\n".join(parts)

    async def generate_standard_response(
        self,
        model: str | None,
        target_format: TargetFormat,
        ctx: RequestContext,
        message_type: str = "short",
        prompt_text: str = "",
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
            error_response = self._generate_error_response(target_format)
            return 429, {"content-type": "application/json"}, error_response

        if message_type == "tool_use":
            mock_response = self.mock_generator.generate_tool_use_response(model=model)
        elif message_type == "long":
            mock_response = self.mock_generator.generate_long_response(model=model)
        elif message_type == "medium":
            mock_response = self.mock_generator.generate_medium_response(model=model)
        else:
            mock_response = self.mock_generator.generate_short_response(model=model)

        # Convert to OpenAI format if needed
        if target_format == FORMAT_OPENAI_CHAT and message_type != "tool_use":
            mock_response = await self.openai_adapter.convert_response(mock_response)
        elif target_format == FORMAT_OPENAI_RESPONSES:
            mock_response = await self.openai_responses_adapter.convert_response(
                mock_response
            )

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
        target_format: TargetFormat,
        ctx: RequestContext,
        message_type: str = "short",
        prompt_text: str = "",
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

            response_id = f"resp_{ctx.request_id if ctx else 'mock'}"
            msg_id = f"msg_{ctx.request_id if ctx else 'mock'}"
            used_model = model or "claude-3-opus-20240229"
            created_at = int(time())
            sequence_number = 0

            # Send initial event
            if target_format == FORMAT_OPENAI_CHAT:
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
            elif target_format == FORMAT_OPENAI_RESPONSES:
                created_event = {
                    "type": "response.created",
                    "sequence_number": sequence_number,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "status": "in_progress",
                        "model": used_model,
                        "output": [],
                        "parallel_tool_calls": False,
                    },
                }
                yield f"data: {json.dumps(created_event)}\n\n".encode()
                sequence_number += 1
                item_added_event = {
                    "type": "response.output_item.added",
                    "sequence_number": sequence_number,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": msg_id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                }
                yield f"data: {json.dumps(item_added_event)}\n\n".encode()
                sequence_number += 1
                part_added_event = {
                    "type": "response.content_part.added",
                    "sequence_number": sequence_number,
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                }
                yield f"data: {json.dumps(part_added_event)}\n\n".encode()
                sequence_number += 1
            else:
                initial_event = {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": used_model,
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

                if target_format == FORMAT_OPENAI_CHAT:
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
                elif target_format == FORMAT_OPENAI_RESPONSES:
                    chunk_event = {
                        "type": "response.output_text.delta",
                        "sequence_number": sequence_number,
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": chunk_text,
                    }
                    sequence_number += 1
                else:
                    chunk_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": chunk_text},
                    }

                yield f"data: {json.dumps(chunk_event)}\n\n".encode()

            # Send final event
            if target_format == FORMAT_OPENAI_CHAT:
                final_event = {
                    "id": f"chatcmpl-{ctx.request_id if ctx else 'mock'}",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": model or "gpt-4",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_event)}\n\n".encode()
            elif target_format == FORMAT_OPENAI_RESPONSES:
                output_tokens = len(text_content.split())
                completed_message = {
                    "type": "message",
                    "id": msg_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text_content}],
                }
                text_done_event = {
                    "type": "response.output_text.done",
                    "sequence_number": sequence_number,
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text_content,
                }
                yield f"data: {json.dumps(text_done_event)}\n\n".encode()
                sequence_number += 1
                part_done_event = {
                    "type": "response.content_part.done",
                    "sequence_number": sequence_number,
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text_content},
                }
                yield f"data: {json.dumps(part_done_event)}\n\n".encode()
                sequence_number += 1
                item_done_event = {
                    "type": "response.output_item.done",
                    "sequence_number": sequence_number,
                    "output_index": 0,
                    "item": completed_message,
                }
                yield f"data: {json.dumps(item_done_event)}\n\n".encode()
                sequence_number += 1
                completed_event = {
                    "type": "response.completed",
                    "sequence_number": sequence_number,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "status": "completed",
                        "model": used_model,
                        "output": [completed_message],
                        "parallel_tool_calls": False,
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": output_tokens,
                            "total_tokens": 10 + output_tokens,
                        },
                    },
                }
                yield f"data: {json.dumps(completed_event)}\n\n".encode()
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

    def _generate_error_response(self, target_format: TargetFormat) -> bytes:
        """Generate a mock error response."""
        if target_format in {FORMAT_OPENAI_CHAT, FORMAT_OPENAI_RESPONSES}:
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
