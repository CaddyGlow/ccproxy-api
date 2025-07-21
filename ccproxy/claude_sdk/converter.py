"""Message format converter for Claude SDK interactions."""

import json
from typing import Any, cast
from xml.sax.saxutils import escape

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
    def extract_text_from_content(
        content: TextBlock | ToolUseBlock | ToolResultBlock,
    ) -> str:
        """
        Extract text content from Claude SDK content blocks.

        Args:
            content: List of content blocks from Claude SDK

        Returns:
            Extracted text content
        """
        if isinstance(content, TextBlock):
            return content.text
        elif isinstance(content, ToolUseBlock):
            # Return full XML representation of ToolUseBlock
            tool_id = escape(str(getattr(content, "id", f"tool_{id(content)}")))
            tool_name = escape(content.name)
            tool_input = getattr(content, "input", {}) or {}
            # Convert input dict to JSON string and escape for XML
            input_json = escape(json.dumps(tool_input, ensure_ascii=False))
            return f'<tooluseblock id="{tool_id}" name="{tool_name}">{input_json}</tooluseblock>'
        elif isinstance(content, ToolResultBlock):
            # Return full XML representation of ToolResultBlock
            tool_use_id = escape(str(getattr(content, "tool_use_id", "")))
            result_content = content.content if isinstance(content.content, str) else ""
            escaped_content = escape(result_content)
            return f'<toolresultblock tool_use_id="{tool_use_id}">{escaped_content}</toolresultblock>'

    @staticmethod
    def extract_contents(
        contents: list[TextBlock | ToolUseBlock | ToolResultBlock],
    ) -> str:
        """
        Extract content from Claude SDK blocks, preserving custom blocks.

        Args:
            content: List of content blocks from Claude SDK

        Returns:
            Content with thinking blocks preserved
        """
        text_parts = []

        for block in contents:
            text_parts.append(MessageConverter.extract_text_from_content(block))

        return " ".join(text_parts)

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
                            "type": "tool_use",
                            "id": getattr(block, "id", f"tool_{id(block)}"),
                            "name": block.name,
                            "input": tool_input,
                        },
                    )
                )
            elif isinstance(block, ToolResultBlock):
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "tool_use_id", ""),
                        "content": block.content
                        if isinstance(block.content, str)
                        else "",
                    }
                )

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
    def create_streaming_start_chunk(message_id: str, model: str) -> dict[str, Any]:
        """
        Create the initial streaming chunk for Anthropic API format.

        Args:
            message_id: The message ID
            model: The model name

        Returns:
            Initial streaming chunk
        """
        return {
            "id": message_id,
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            },
        }

    @staticmethod
    def create_streaming_delta_chunk(text: str) -> dict[str, Any]:
        """
        Create a streaming delta chunk for Anthropic API format.

        Args:
            text: The text content to include

        Returns:
            Delta chunk
        """
        return {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }

    @staticmethod
    def create_streaming_end_chunk(stop_reason: str = "end_turn") -> dict[str, Any]:
        """
        Create the final streaming chunk for Anthropic API format.

        Args:
            stop_reason: The reason for stopping

        Returns:
            Final streaming chunk
        """
        return {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": 0},
        }
