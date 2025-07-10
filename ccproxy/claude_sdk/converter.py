"""Message format converter for Claude SDK interactions."""

from typing import Any

from ccproxy.utils.helper import patched_typing


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
        content: list[TextBlock | ToolUseBlock | ToolResultBlock],
    ) -> str:
        """
        Extract text content from Claude SDK content blocks.

        Args:
            content: List of content blocks from Claude SDK

        Returns:
            Extracted text content
        """
        text_parts = []

        for block in content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                # For tool use blocks, include the tool name
                text_parts.append(f"[Tool: {block.name}]")
            elif isinstance(block, ToolResultBlock) and isinstance(block.content, str):
                text_parts.append(block.content)

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
        return {
            "id": f"msg_{result_message.session_id}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": MessageConverter.extract_text_from_content(
                        assistant_message.content
                    ),
                }
            ],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,  # Claude Code SDK doesn't provide token counts
                "output_tokens": 0,
                "total_tokens": 0,
            },
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
