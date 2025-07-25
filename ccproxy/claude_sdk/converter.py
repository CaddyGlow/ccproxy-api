"""Message format converter for Claude SDK interactions."""

from typing import Any, cast

import structlog

from ccproxy.core.async_utils import patched_typing


logger = structlog.get_logger(__name__)

with patched_typing():
    from claude_code_sdk import (
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
    )


class MessageConverter:
    """
    Handles conversion between Anthropic API format and Claude SDK format.
    """

    @staticmethod
    def format_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        """
        Convert Anthropic messages format to a single prompt string.

        Args:
            messages: List of messages in Anthropic format

        Returns:
            Single prompt string formatted for Claude SDK
        """
        prompt_parts = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if isinstance(content, list):
                # Handle content blocks
                text_parts = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = " ".join(text_parts)

            if role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                # System messages are handled via options
                continue

        return "\n\n".join(prompt_parts)

    @staticmethod
    def convert_to_anthropic_response(
        assistant_message: AssistantMessage,
        result_message: ResultMessage,
        model: str,
    ) -> dict[str, Any]:
        """
        Convert Claude SDK messages to Anthropic API response format.

        Args:
            assistant_message: The assistant message from Claude SDK
            result_message: The result message from Claude SDK
            model: The model name used

        Returns:
            Response in Anthropic API format
        """
        # Extract token usage from result message
        # First try to get usage from the usage field (preferred method)
        usage = getattr(result_message, "usage", {})
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
        else:
            # Fallback to direct attributes
            input_tokens = getattr(result_message, "input_tokens", 0)
            output_tokens = getattr(result_message, "output_tokens", 0)
            cache_read_tokens = getattr(result_message, "cache_read_tokens", 0)
            cache_write_tokens = getattr(result_message, "cache_write_tokens", 0)

        # Log token extraction for debugging
        from structlog import get_logger

        logger = get_logger(__name__)

        logger.debug(
            "assistant_message_content",
            content_blocks=[
                type(block).__name__ for block in assistant_message.content
            ],
            content_count=len(assistant_message.content),
            first_block_text=(
                assistant_message.content[0].text[:100]
                if assistant_message.content
                and hasattr(assistant_message.content[0], "text")
                else None
            ),
        )

        logger.debug(
            "token_usage_extracted",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            source="claude_sdk",
        )

        # Calculate total tokens
        total_tokens = input_tokens + output_tokens

        # Build usage information
        usage_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "total_tokens": total_tokens,
        }

        # Add cost information if available
        total_cost_usd = getattr(result_message, "total_cost_usd", None)
        if total_cost_usd is not None:
            usage_info["cost_usd"] = total_cost_usd

        # Convert content blocks to Anthropic format, preserving thinking blocks
        content_blocks = []

        for block in assistant_message.content:
            if isinstance(block, TextBlock):
                # Parse text content for thinking blocks
                text = block.text

                # Check if the text contains thinking blocks
                import re

                thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'

                # Split the text by thinking blocks
                last_end = 0
                for match in re.finditer(thinking_pattern, text, re.DOTALL):
                    # Add any text before the thinking block
                    before_text = text[last_end : match.start()].strip()
                    if before_text:
                        content_blocks.append({"type": "text", "text": before_text})

                    # Add the thinking block
                    signature, thinking_text = match.groups()
                    content_blocks.append(
                        {
                            "type": "thinking",
                            "text": thinking_text,
                            "signature": signature,
                        }
                    )

                    last_end = match.end()

                # Add any remaining text after the last thinking block
                remaining_text = text[last_end:].strip()
                if remaining_text:
                    content_blocks.append({"type": "text", "text": remaining_text})

                # If no thinking blocks were found, add the entire text as a text block
                if last_end == 0 and text:
                    content_blocks.append({"type": "text", "text": text})

            elif isinstance(block, ToolUseBlock):
                tool_input = getattr(block, "input", {}) or {}
                content_blocks.append(
                    cast(
                        dict[str, Any],
                        {
                            "type": "tool_use_sdk",
                            "id": getattr(block, "id", f"tool_{id(block)}"),
                            "name": block.name,
                            "input": tool_input,
                            "source": "claude_code_sdk",
                        },
                    )
                )
            elif isinstance(block, ToolResultBlock):
                is_error = getattr(block, "is_error", None)
                tool_result_block: dict[str, Any] = {
                    "type": "tool_result_sdk",
                    "tool_use_id": getattr(block, "tool_use_id", ""),
                    "content": block.content if isinstance(block.content, str) else "",
                    "is_error": is_error if is_error is not None else False,
                    "source": "claude_code_sdk",
                }
                content_blocks.append(tool_result_block)

        return {
            "id": f"msg_{result_message.session_id}",
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": getattr(result_message, "stop_reason", "end_turn"),
            "stop_sequence": None,
            "usage": usage_info,
        }

    @staticmethod
    def create_streaming_start_chunks(
        message_id: str, model: str, input_tokens: int = 0
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Create the initial streaming chunks for Anthropic API format.

        Args:
            message_id: The message ID
            model: The model name
            input_tokens: Number of input tokens for the request

        Returns:
            List of tuples (event_type, chunk) for initial streaming chunks
        """
        return [
            # First, send message_start with event type
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": input_tokens,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 1,
                            "service_tier": "standard",
                        },
                    },
                },
            ),
        ]

    @staticmethod
    def create_streaming_delta_chunk(text: str) -> tuple[str, dict[str, Any]]:
        """
        Create a streaming delta chunk for Anthropic API format.

        Args:
            text: The text content to include

        Returns:
            Tuple of (event_type, chunk)
        """
        return (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )

    @staticmethod
    def create_streaming_end_chunks(
        stop_reason: str = "end_turn", stop_sequence: str | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Create the final streaming chunks for Anthropic API format.

        Args:
            stop_reason: The reason for stopping
            stop_sequence: The stop sequence used (if any)

        Returns:
            List of tuples (event_type, chunk) for final streaming chunks
        """
        return [
            # Then, send message_delta with stop reason and usage
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": stop_sequence,
                    },
                    "usage": {"output_tokens": 0},
                },
            ),
            # Finally, send message_stop
            ("message_stop", {"type": "message_stop"}),
        ]

    @staticmethod
    def create_ping_chunk() -> tuple[str, dict[str, Any]]:
        """
        Create a ping chunk for keeping the connection alive.

        Returns:
            Tuple of (event_type, chunk)
        """
        return ("ping", {"type": "ping"})

    @staticmethod
    def create_system_message_content_block(
        message_text: str, source: str = "claude_code_sdk"
    ) -> dict[str, Any]:
        """
        Create a system_message content block for non-streaming responses.

        Args:
            message_text: The system message text content
            source: The source of the system message (default: "claude_code_sdk")

        Returns:
            Content block dict for system message
        """
        return {
            "type": "system_message",
            "text": message_text,
            "source": source,
        }

    @staticmethod
    def create_system_message_chunks(
        message_text: str, index: int = 0
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Create streaming chunks for system messages using custom content block type.

        Args:
            message_text: The system message text content
            index: The content block index for the system message

        Returns:
            List of tuples (event_type, chunk) for system message streaming chunks
        """
        return [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "system_message",
                        "text": message_text,
                        "source": "claude_code_sdk",
                    },
                },
            ),
            (
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": index,
                },
            ),
        ]
