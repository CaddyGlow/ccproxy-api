"""OpenAI-format streaming formatter utilities.

This module provides formatting utilities for OpenAI-compatible SSE (Server-Sent Events) streams.
It contains the OpenAIStreamingFormatter class and basic wrapper functions that use the
unified stream_transformer.py for actual transformation logic.

The actual streaming transformation logic is handled by:
- stream_transformer.py: Unified transformation framework that supports multiple input sources

For Anthropic-format streaming, see:
- anthropic_streaming.py: Anthropic native SSE format utilities
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


class OpenAIStreamingFormatter:
    """Formats streaming responses to match OpenAI's SSE format."""

    @staticmethod
    def format_data_event(data: dict[str, Any]) -> str:
        """
        Format a data event for OpenAI-compatible Server-Sent Events.

        Args:
            data: Event data dictionary

        Returns:
            Formatted SSE string
        """
        json_data = json.dumps(data, separators=(",", ":"))
        return f"data: {json_data}\n\n"

    @staticmethod
    def format_first_chunk(
        message_id: str, model: str, created: int, role: str = "assistant"
    ) -> str:
        """
        Format the first chunk with role and basic metadata.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            role: Role of the assistant

        Returns:
            Formatted SSE string
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": role},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_content_chunk(
        message_id: str, model: str, created: int, content: str, choice_index: int = 0
    ) -> str:
        """
        Format a content chunk with text delta.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            content: Text content to include in the delta
            choice_index: Index of the choice (usually 0)

        Returns:
            Formatted SSE string
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"content": content},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_tool_call_chunk(
        message_id: str,
        model: str,
        created: int,
        tool_call_id: str,
        function_name: str | None = None,
        function_arguments: str | None = None,
        tool_call_index: int = 0,
        choice_index: int = 0,
    ) -> str:
        """
        Format a tool call chunk.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            tool_call_id: ID of the tool call
            function_name: Name of the function being called
            function_arguments: Arguments for the function
            tool_call_index: Index of the tool call
            choice_index: Index of the choice (usually 0)

        Returns:
            Formatted SSE string
        """
        tool_call: dict[str, Any] = {
            "index": tool_call_index,
            "id": tool_call_id,
            "type": "function",
            "function": {},
        }

        if function_name is not None:
            tool_call["function"]["name"] = function_name

        if function_arguments is not None:
            tool_call["function"]["arguments"] = function_arguments

        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"tool_calls": [tool_call]},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_final_chunk(
        message_id: str,
        model: str,
        created: int,
        finish_reason: str = "stop",
        choice_index: int = 0,
        usage: dict[str, int] | None = None,
    ) -> str:
        """
        Format the final chunk with finish_reason.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            finish_reason: Reason for completion (stop, length, tool_calls, etc.)
            choice_index: Index of the choice (usually 0)
            usage: Optional usage information to include

        Returns:
            Formatted SSE string
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

        # Add usage if provided
        if usage:
            data["usage"] = usage

        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_error_chunk(
        message_id: str, model: str, created: int, error_type: str, error_message: str
    ) -> str:
        """
        Format an error chunk.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            error_type: Type of error
            error_message: Error message

        Returns:
            Formatted SSE string
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "error"}
            ],
            "error": {"type": error_type, "message": error_message},
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_done() -> str:
        """
        Format the final DONE event.

        Returns:
            Formatted SSE termination string
        """
        return "data: [DONE]\n\n"


async def stream_claude_response_openai(
    claude_response_iterator: AsyncGenerator[dict[str, Any], None],
    message_id: str,
    model: str,
    created: int | None = None,
    include_usage: bool = False,
    enable_text_chunking: bool = True,
    enable_tool_calls: bool = True,
    chunk_delay_ms: float = 10.0,
    chunk_size_words: int = 3,
) -> AsyncGenerator[str, None]:
    """
    Convert Claude SDK response to OpenAI-compatible streaming format.

    This is the unified OpenAI streaming function that replaces both the regular
    and simplified variants. All streaming behavior is now configurable.

    Args:
        claude_response_iterator: Async iterator of Claude response chunks
        message_id: Unique message identifier
        model: Model name being used
        created: Unix timestamp when the completion was created
        include_usage: Whether to include usage information
        enable_text_chunking: Whether to enable text chunking for streaming
        enable_tool_calls: Whether to enable tool call support
        chunk_delay_ms: Delay between text chunks in milliseconds
        chunk_size_words: Number of words per chunk

    Yields:
        Formatted OpenAI-compatible SSE strings
    """
    from ccproxy.services.stream_transformer import (
        OpenAIStreamTransformer,
        StreamingConfig,
    )

    # Configure streaming based on parameters
    config = StreamingConfig(
        enable_text_chunking=enable_text_chunking,
        enable_tool_calls=enable_tool_calls,
        enable_usage_info=include_usage,
        chunk_delay_ms=chunk_delay_ms,
        chunk_size_words=chunk_size_words,
    )

    # Create transformer
    transformer = OpenAIStreamTransformer.from_claude_sdk(
        claude_response_iterator,
        message_id=message_id,
        model=model,
        created=created,
        config=config,
    )

    # Transform and yield
    async for chunk in transformer.transform():
        yield chunk


async def stream_claude_response_openai_simple(
    claude_response_iterator: AsyncGenerator[dict[str, Any], None],
    message_id: str,
    model: str,
    created: int | None = None,
    include_usage: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Convert Claude SDK response to OpenAI-compatible streaming format (simplified).

    This is a convenience function that provides simplified streaming behavior
    without tool calls or text chunking for basic compatibility.

    Args:
        claude_response_iterator: Async iterator of Claude response chunks
        message_id: Unique message identifier
        model: Model name being used
        created: Unix timestamp when the completion was created
        include_usage: Whether to include usage information in streaming responses

    Yields:
        Formatted OpenAI-compatible SSE strings
    """
    async for chunk in stream_claude_response_openai(
        claude_response_iterator,
        message_id,
        model,
        created=created,
        include_usage=include_usage,
        enable_text_chunking=False,
        enable_tool_calls=False,
        chunk_delay_ms=0.0,
        chunk_size_words=1,
    ):
        yield chunk
