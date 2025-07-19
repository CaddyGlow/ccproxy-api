"""Mock response generation for realistic testing."""

import json
import random
import time
from typing import Any

from ccproxy.testing.config import MockResponseConfig
from ccproxy.testing.content_generation import MessageContentGenerator


class RealisticMockResponseGenerator:
    """Generate realistic mock responses with proper randomization."""

    def __init__(self, config: MockResponseConfig | None = None):
        self.config = config or MockResponseConfig()
        self.content_generator: MessageContentGenerator = MessageContentGenerator()

    def generate_response_content(
        self, message_type: str, model: str
    ) -> tuple[str, int, int]:
        """Generate response content with realistic token counts."""
        return self.content_generator.get_response_content(message_type, model)

    def generate_cache_tokens(self) -> tuple[int, int]:
        """Generate realistic cache token counts."""
        if random.random() < self.config.cache_token_probability:
            cache_read = random.randint(*self.config.cache_read_range)
            cache_write = random.randint(*self.config.cache_write_range)
            return cache_read, cache_write
        return 0, 0

    def should_simulate_error(self) -> bool:
        """Determine if this response should be an error."""
        return (
            self.config.simulate_errors
            and random.random() < self.config.error_probability
        )

    def generate_error_response(self, api_format: str) -> tuple[dict[str, Any], int]:
        """Generate realistic error response."""
        error_types = [
            {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded. Please try again later.",
                "status_code": 429,
            },
            {
                "type": "invalid_request_error",
                "message": "Invalid request format.",
                "status_code": 400,
            },
            {
                "type": "overloaded_error",
                "message": "Service temporarily overloaded.",
                "status_code": 503,
            },
        ]

        error = random.choice(error_types)
        status_code: int = error["status_code"]  # type: ignore[assignment]

        if api_format == "openai":
            return {
                "error": {
                    "message": error["message"],
                    "type": error["type"],
                    "code": error["type"],
                }
            }, status_code
        else:
            return {
                "type": "error",
                "error": {"type": error["type"], "message": error["message"]},
            }, status_code

    def generate_realistic_anthropic_stream(
        self,
        request_id: str,
        model: str,
        content: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> list[dict[str, Any]]:
        """Generate realistic Anthropic streaming chunks."""

        chunks = []

        # Message start
        chunks.append(
            {
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            }
        )

        # Content block start
        chunk_start: dict[str, Any] = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        chunks.append(chunk_start)

        # Split content into realistic chunks (by words)
        words = content.split()
        chunk_sizes = []

        # Generate realistic chunk sizes
        i = 0
        while i < len(words):
            # Random chunk size between 1-5 words
            chunk_size = random.randint(1, min(5, len(words) - i))
            chunk_sizes.append(chunk_size)
            i += chunk_size

        # Generate content deltas
        word_index = 0
        for chunk_size in chunk_sizes:
            chunk_words = words[word_index : word_index + chunk_size]
            chunk_text = (
                " " + " ".join(chunk_words) if word_index > 0 else " ".join(chunk_words)
            )

            chunk_delta: dict[str, Any] = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk_text},
            }
            chunks.append(chunk_delta)
            word_index += chunk_size

        # Content block stop
        chunk_stop: dict[str, Any] = {"type": "content_block_stop", "index": 0}
        chunks.append(chunk_stop)

        # Message delta with final usage
        chunks.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": cache_write_tokens,
                    "cache_read_input_tokens": cache_read_tokens,
                },
            }
        )

        # Message stop
        chunks.append({"type": "message_stop"})

        return chunks

    def generate_realistic_openai_stream(
        self,
        request_id: str,
        model: str,
        content: str,
        input_tokens: int,
        output_tokens: int,
    ) -> list[dict[str, Any]]:
        """Generate realistic OpenAI streaming chunks by converting Anthropic format."""

        # Generate Anthropic chunks first
        anthropic_chunks = self.generate_realistic_anthropic_stream(
            request_id, model, content, input_tokens, output_tokens, 0, 0
        )

        # Convert to OpenAI format
        openai_chunks = []
        for chunk in anthropic_chunks:
            # Use simplified conversion logic
            if chunk.get("type") == "message_start":
                openai_chunks.append(
                    {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            elif chunk.get("type") == "content_block_delta":
                delta_text = chunk.get("delta", {}).get("text", "")
                openai_chunks.append(
                    {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            elif chunk.get("type") == "message_stop":
                openai_chunks.append(
                    {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                )

        return openai_chunks

    def calculate_realistic_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> float:
        """Calculate realistic cost based on current Claude pricing."""

        # Simplified pricing (should use actual cost calculator)
        if "sonnet" in model.lower():
            input_cost_per_token = 0.000003  # $3 per million tokens
            output_cost_per_token = 0.000015  # $15 per million tokens
        elif "haiku" in model.lower():
            input_cost_per_token = 0.00000025  # $0.25 per million tokens
            output_cost_per_token = 0.00000125  # $1.25 per million tokens
        else:
            input_cost_per_token = 0.000003
            output_cost_per_token = 0.000015

        base_cost = (
            input_tokens * input_cost_per_token + output_tokens * output_cost_per_token
        )

        # Cache costs (typically lower)
        cache_cost = (
            cache_read_tokens * input_cost_per_token * 0.1  # 10% of input cost
            + cache_write_tokens * input_cost_per_token * 0.5  # 50% of input cost
        )

        return round(base_cost + cache_cost, 6)
