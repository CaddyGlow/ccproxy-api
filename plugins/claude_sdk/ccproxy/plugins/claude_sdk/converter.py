"""Message format converter for Claude SDK interactions."""

import html
import json
from collections.abc import Callable
from typing import Any

from ccproxy.core.async_utils import patched_typing
from ccproxy.core.logging import get_plugin_logger

from . import models as sdk_models
from .config import SDKMessageMode
from .models import MessageResponse


logger = get_plugin_logger()

with patched_typing():
    pass


class MessageConverter:
    """
    Handles conversion between Anthropic API format and Claude SDK format.
    """

    @staticmethod
    def _format_json_data(
        data: dict[str, Any],
        pretty_format: bool = True,
    ) -> str:
        """
        Format JSON data with optional indentation and newlines.

        Args:
            data: Dictionary to format as JSON
            pretty_format: Whether to use pretty formatting (indented JSON with spacing)

        Returns:
            Formatted JSON string
        """

        if pretty_format:
            # Pretty format with indentation and proper spacing
            return json.dumps(data, indent=2, separators=(", ", ": "))
        else:
            # Compact format without indentation or spacing
            return json.dumps(data, separators=(",", ":"))

    @staticmethod
    def _create_xml_formatted_text(
        data: dict[str, Any], tag_name: str, pretty_format: bool = True
    ) -> str:
        """
        Create XML-formatted text from data with consistent formatting.

        Args:
            data: Dictionary data to format as JSON and wrap in XML
            tag_name: XML tag name to wrap the content
            pretty_format: Whether to use pretty formatting

        Returns:
            Formatted XML string
        """
        formatted_json = MessageConverter._format_json_data(data, pretty_format)
        escaped_json = MessageConverter._escape_content_for_xml(
            formatted_json, pretty_format
        )

        if pretty_format:
            return f"<{tag_name}>\n{escaped_json}\n</{tag_name}>\n"
        else:
            return f"<{tag_name}>{escaped_json}</{tag_name}>"

    @staticmethod
    def _create_streaming_chunks_with_content(
        content_block: dict[str, Any],
        index: int,
        text_content: str | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Create streaming chunks with optional text delta content.

        Args:
            content_block: Content block for content_block_start
            index: Content block index
            text_content: Optional text content for content_block_delta

        Returns:
            List of streaming chunks
        """
        chunks = [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": content_block,
                },
            )
        ]

        if text_content is not None:
            chunks.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text_content},
                    },
                )
            )

        chunks.append(
            (
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": index,
                },
            )
        )

        return chunks

    @staticmethod
    def _escape_content_for_xml(content: str, pretty_format: bool = True) -> str:
        """
        Escape content for inclusion in XML tags.

        Args:
            content: Content to escape
            pretty_format: Whether to use pretty formatting (no escaping) or compact (escaped)

        Returns:
            Escaped or unescaped content based on formatting mode
        """
        if pretty_format:
            # Pretty format: no escaping, content as-is
            return content
        else:
            # Compact format: escape special XML characters

            return html.escape(content)

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
        assistant_message: sdk_models.AssistantMessage,
        result_message: sdk_models.ResultMessage,
        model: str,
        mode: SDKMessageMode = SDKMessageMode.FORWARD,
        pretty_format: bool = True,
    ) -> "MessageResponse":
        """
        Convert Claude SDK messages to Anthropic API response format.

        Args:
            assistant_message: The assistant message from Claude SDK
            result_message: The result message from Claude SDK
            model: The model name used
            mode: System message handling mode (forward, ignore, formatted)
            pretty_format: Whether to use pretty formatting (true: indented JSON with newlines, false: compact with escaped content)

        Returns:
            Response in Anthropic API format
        """
        # Extract token usage from result message
        usage = result_message.usage_model

        # Log token extraction for debugging
        # logger.debug(
        #     "assistant_message_content",
        #     content_blocks=[block.type for block in assistant_message.content],
        #     content_count=len(assistant_message.content),
        # )

        logger.debug(
            "token_usage_extracted",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens,
            source="claude_sdk",
        )

        # Build usage information
        usage_info = usage.model_dump(mode="json")

        # Add cost information if available
        if result_message.total_cost_usd is not None:
            usage_info["cost_usd"] = result_message.total_cost_usd

        # Convert content blocks to Anthropic format, preserving thinking blocks
        content_blocks = []

        for block in assistant_message.content:
            if isinstance(block, sdk_models.TextBlock):
                # Handle text content directly without thinking block parsing
                text = block.text
                if mode == SDKMessageMode.FORMATTED:
                    escaped_text = MessageConverter._escape_content_for_xml(
                        text, pretty_format
                    )
                    formatted_text = (
                        f"<text>\n{escaped_text}\n</text>\n"
                        if pretty_format
                        else f"<text>{escaped_text}</text>"
                    )
                    content_blocks.append({"type": "text", "text": formatted_text})
                else:
                    content_blocks.append({"type": "text", "text": text})

            elif isinstance(block, sdk_models.ToolUseBlock):
                if mode == SDKMessageMode.FORWARD:
                    content_blocks.append(block.to_sdk_block())
                elif mode == SDKMessageMode.FORMATTED:
                    tool_data = block.model_dump(mode="json")
                    formatted_json = MessageConverter._format_json_data(
                        tool_data, pretty_format
                    )
                    escaped_json = MessageConverter._escape_content_for_xml(
                        formatted_json, pretty_format
                    )
                    formatted_text = (
                        f"<tool_use_sdk>\n{escaped_json}\n</tool_use_sdk>\n"
                        if pretty_format
                        else f"<tool_use_sdk>{escaped_json}</tool_use_sdk>"
                    )
                    content_blocks.append({"type": "text", "text": formatted_text})

            elif isinstance(block, sdk_models.ToolResultBlock):
                if mode == SDKMessageMode.FORWARD:
                    content_blocks.append(block.to_sdk_block())
                elif mode == SDKMessageMode.FORMATTED:
                    tool_result_data = block.model_dump(mode="json")
                    formatted_json = MessageConverter._format_json_data(
                        tool_result_data, pretty_format
                    )
                    escaped_json = MessageConverter._escape_content_for_xml(
                        formatted_json, pretty_format
                    )
                    formatted_text = (
                        f"<tool_result_sdk>\n{escaped_json}\n</tool_result_sdk>\n"
                        if pretty_format
                        else f"<tool_result_sdk>{escaped_json}</tool_result_sdk>"
                    )
                    content_blocks.append({"type": "text", "text": formatted_text})

            elif isinstance(block, sdk_models.ThinkingBlock):
                if mode == SDKMessageMode.FORWARD:
                    thinking_block = {
                        "type": "thinking",
                        "thinking": block.thinking,
                    }
                    if block.signature is not None:
                        thinking_block["signature"] = block.signature
                    content_blocks.append(thinking_block)
                elif mode == SDKMessageMode.FORMATTED:
                    # Format thinking block with signature in XML tag attribute
                    signature_attr = (
                        f' signature="{block.signature}"' if block.signature else ""
                    )
                    if pretty_format:
                        escaped_text = MessageConverter._escape_content_for_xml(
                            block.thinking, pretty_format
                        )
                        formatted_text = (
                            f"<thinking{signature_attr}>\n{escaped_text}\n</thinking>\n"
                        )
                    else:
                        escaped_text = MessageConverter._escape_content_for_xml(
                            block.thinking, pretty_format
                        )
                        formatted_text = (
                            f"<thinking{signature_attr}>{escaped_text}</thinking>"
                        )
                    content_blocks.append({"type": "text", "text": formatted_text})

        return MessageResponse.model_validate(
            {
                "id": f"msg_{result_message.session_id}",
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": model,
                "stop_reason": result_message.stop_reason,
                "stop_sequence": None,
                "usage": usage_info,
            }
        )

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
    def _create_sdk_content_block(
        sdk_object: sdk_models.SystemMessage
        | sdk_models.ToolUseBlock
        | sdk_models.ToolResultBlock
        | sdk_models.ResultMessage,
        mode: SDKMessageMode = SDKMessageMode.FORWARD,
        pretty_format: bool = True,
        xml_tag: str = "sdk_block",
        forward_converter: Callable[[Any], dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """
        Generic method to create content blocks for SDK objects in non-streaming responses.

        Args:
            sdk_object: The SDK object to convert
            mode: System message handling mode
            pretty_format: Whether to use pretty formatting
            xml_tag: XML tag name for FORMATTED mode
            forward_converter: Optional converter function for FORWARD mode

        Returns:
            Content block dict for the SDK object, or None if mode is IGNORE
        """
        if mode == SDKMessageMode.IGNORE:
            return None
        elif mode == SDKMessageMode.FORWARD:
            if forward_converter:
                return forward_converter(sdk_object)
            else:
                return sdk_object.model_dump(mode="json")
        elif mode == SDKMessageMode.FORMATTED:
            object_data = sdk_object.model_dump(mode="json")
            formatted_json = MessageConverter._format_json_data(
                object_data, pretty_format
            )
            escaped_json = MessageConverter._escape_content_for_xml(
                formatted_json, pretty_format
            )
            formatted_text = (
                f"<{xml_tag}>\n{escaped_json}\n</{xml_tag}>\n"
                if pretty_format
                else f"<{xml_tag}>{escaped_json}</{xml_tag}>"
            )
            return {
                "type": "text",
                "text": formatted_text,
            }

    @staticmethod
    def _create_sdk_content_block_chunks(
        sdk_object: sdk_models.SystemMessage
        | sdk_models.ToolUseBlock
        | sdk_models.ToolResultBlock
        | sdk_models.ResultMessage,
        mode: SDKMessageMode = SDKMessageMode.FORWARD,
        index: int = 0,
        pretty_format: bool = True,
        xml_tag: str = "sdk_block",
        sdk_block_converter: Callable[[Any], dict[str, Any]] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Generic method to create streaming chunks for SDK content blocks.

        Args:
            sdk_object: The SDK object (SystemMessage, ToolUseBlock, or ToolResultBlock)
            mode: System message handling mode
            index: The content block index
            pretty_format: Whether to use pretty formatting
            xml_tag: XML tag name for FORMATTED mode
            sdk_block_converter: Optional converter function for FORWARD mode

        Returns:
            List of tuples (event_type, chunk) for streaming chunks
        """
        if mode == SDKMessageMode.IGNORE:
            return []
        elif mode == SDKMessageMode.FORWARD:
            content_block = (
                sdk_block_converter(sdk_object)
                if sdk_block_converter
                else sdk_object.model_dump(mode="json")
            )
            return MessageConverter._create_streaming_chunks_with_content(
                content_block=content_block,
                index=index,
            )
        elif mode == SDKMessageMode.FORMATTED:
            object_data = sdk_object.model_dump(mode="json")
            formatted_text = MessageConverter._create_xml_formatted_text(
                object_data, xml_tag, pretty_format
            )
            return MessageConverter._create_streaming_chunks_with_content(
                content_block={"type": "text", "text": ""},
                index=index,
                text_content=formatted_text,
            )
