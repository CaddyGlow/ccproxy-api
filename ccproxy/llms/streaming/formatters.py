"""SSE formatting utilities for streaming responses.

This module provides Server-Sent Events (SSE) formatting classes for converting
streaming responses to different formats.
"""

import json
from typing import Any


class AnthropicSSEFormatter:
    """Formats streaming responses to match Anthropic's Messages API SSE format."""

    @staticmethod
    def format_event(event_type: str, data: dict[str, Any]) -> str:
        """Format an event for Anthropic Messages API Server-Sent Events.

        Args:
            event_type: Event type (e.g., 'message_start', 'content_block_delta')
            data: Event data dictionary

        Returns:
            Formatted SSE string with event and data lines
        """
        json_data = json.dumps(data, separators=(",", ":"))
        return f"event: {event_type}\ndata: {json_data}\n\n"

    @staticmethod
    def format_ping() -> str:
        """Format a ping event."""
        return 'event: ping\ndata: {"type": "ping"}\n\n'

    @staticmethod
    def format_done() -> str:
        """Format the final [DONE] event."""
        return "data: [DONE]\n\n"


class OpenAISSEFormatter:
    """Formats streaming responses to match OpenAI's SSE format."""

    @staticmethod
    def format_data_event(data: dict[str, Any]) -> str:
        """Format a data event for OpenAI-compatible Server-Sent Events.

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
        """Format the first chunk with role and basic metadata.

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
        return OpenAISSEFormatter.format_data_event(data)

    @staticmethod
    def format_content_chunk(
        message_id: str, model: str, created: int, content: str, choice_index: int = 0
    ) -> str:
        """Format a content chunk with text delta.

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
        return OpenAISSEFormatter.format_data_event(data)

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
        """Format a tool call chunk.

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
        return OpenAISSEFormatter.format_data_event(data)

    @staticmethod
    def format_final_chunk(
        message_id: str,
        model: str,
        created: int,
        finish_reason: str = "stop",
        choice_index: int = 0,
        usage: dict[str, int] | None = None,
    ) -> str:
        """Format the final chunk with finish_reason.

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

        return OpenAISSEFormatter.format_data_event(data)

    @staticmethod
    def format_error_chunk(
        message_id: str, model: str, created: int, error_type: str, error_message: str
    ) -> str:
        """Format an error chunk.

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
        return OpenAISSEFormatter.format_data_event(data)

    @staticmethod
    def format_done() -> str:
        """Format the final DONE event.

        Returns:
            Formatted SSE termination string
        """
        return "data: [DONE]\n\n"
