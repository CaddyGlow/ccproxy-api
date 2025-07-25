"""Message format converter for Claude SDK interactions."""

import re
from typing import Any

import structlog

from ccproxy.claude_sdk import models as sdk_models
from ccproxy.config.claude import SystemMessageMode
from ccproxy.core.async_utils import patched_typing
from ccproxy.models.messages import MessageResponse


logger = structlog.get_logger(__name__)

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
        import json

        if pretty_format:
            # Pretty format with indentation and proper spacing
            return json.dumps(data, indent=2, separators=(", ", ": "))
        else:
            # Compact format without indentation or spacing
            return json.dumps(data, separators=(",", ":"))

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
            import html

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
        mode: SystemMessageMode = SystemMessageMode.FORWARD,
        pretty_format: bool = True,
    ) -> MessageResponse:
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
        usage = result_message.usage

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
        usage_info = usage.model_dump()

        # Add cost information if available
        if result_message.total_cost_usd is not None:
            usage_info["cost_usd"] = result_message.total_cost_usd

        # Convert content blocks to Anthropic format, preserving thinking blocks
        content_blocks = []

        for block in assistant_message.content:
            if isinstance(block, sdk_models.TextBlock):
                # Parse text content for thinking blocks
                text = block.text
                thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'
                last_end = 0
                for match in re.finditer(thinking_pattern, text, re.DOTALL):
                    before_text = text[last_end : match.start()].strip()
                    if before_text:
                        if mode == SystemMessageMode.FORMATTED:
                            escaped_text = MessageConverter._escape_content_for_xml(
                                before_text, pretty_format
                            )
                            formatted_text = (
                                f"<text>\n{escaped_text}\n</text>\n"
                                if pretty_format
                                else f"<text>{escaped_text}</text>"
                            )
                            content_blocks.append(
                                {"type": "text", "text": formatted_text}
                            )
                        else:
                            content_blocks.append({"type": "text", "text": before_text})

                    signature, thinking_text = match.groups()
                    content_blocks.append(
                        {
                            "type": "thinking",
                            "text": thinking_text,
                            "signature": signature,
                        }
                    )
                    last_end = match.end()

                remaining_text = text[last_end:].strip()
                if remaining_text:
                    if mode == SystemMessageMode.FORMATTED:
                        escaped_text = MessageConverter._escape_content_for_xml(
                            remaining_text, pretty_format
                        )
                        formatted_text = (
                            f"<text>\n{escaped_text}\n</text>\n"
                            if pretty_format
                            else f"<text>{escaped_text}</text>"
                        )
                        content_blocks.append({"type": "text", "text": formatted_text})
                    else:
                        content_blocks.append({"type": "text", "text": remaining_text})
                elif last_end == 0 and text:
                    if mode == SystemMessageMode.FORMATTED:
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
                if mode == SystemMessageMode.FORWARD:
                    content_blocks.append(
                        sdk_models.ToolUseSDKBlock(
                            **block.model_dump(),
                        ).model_dump(exclude_none=True)
                    )
                elif mode == SystemMessageMode.FORMATTED:
                    tool_data = block.model_dump()
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
                if mode == SystemMessageMode.FORWARD:
                    content_blocks.append(
                        sdk_models.ToolResultSDKBlock(
                            **block.model_dump(),
                        ).model_dump(exclude_none=True)
                    )
                elif mode == SystemMessageMode.FORMATTED:
                    tool_result_data = block.model_dump()
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
    def convert_sdk_messages_to_anthropic_response(
        sdk_messages: list[
            sdk_models.AssistantMessage
            | sdk_models.SystemMessage
            | sdk_models.ResultMessage
            | sdk_models.UserMessage
        ],
        model: str,
        mode: SystemMessageMode = SystemMessageMode.FORWARD,
        pretty_format: bool = True,
    ) -> MessageResponse:
        """
        Convert a full list of Claude SDK messages to Anthropic API response format.

        This method processes all SDK messages from a non-streaming response and
        consolidates them into a single Anthropic API response, handling system
        messages and other SDK-specific content based on the specified mode.

        Args:
            sdk_messages: List of SDK messages (AssistantMessage, SystemMessage, ResultMessage, etc.)
            model: The model name used
            mode: System message handling mode (forward, ignore, formatted)
            pretty_format: Whether to use pretty formatting for XML content

        Returns:
            Response in Anthropic API format
        """
        assistant_messages = [
            m for m in sdk_messages if isinstance(m, sdk_models.AssistantMessage)
        ]
        system_messages = [
            m for m in sdk_messages if isinstance(m, sdk_models.SystemMessage)
        ]
        result_message = next(
            (m for m in sdk_messages if isinstance(m, sdk_models.ResultMessage)), None
        )

        # If we have no assistant messages, create a minimal response
        if not assistant_messages:
            if result_message:
                return MessageResponse.model_validate(
                    {
                        "id": f"msg_{result_message.session_id}",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": result_message.stop_reason,
                        "stop_sequence": None,
                        "usage": result_message.usage.model_dump(),
                    }
                )
            else:
                return MessageResponse.model_validate(
                    {
                        "id": "msg_unknown",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {},
                    }
                )

        # Process the first assistant message as the main response
        main_assistant_message = assistant_messages[0]

        # If we have a result message, use the existing conversion method
        if result_message:
            response_data = MessageConverter.convert_to_anthropic_response(
                main_assistant_message, result_message, model, mode, pretty_format
            )
            response = MessageResponse.model_validate(response_data)
        else:
            # Fallback for cases without result message
            response = MessageResponse.model_validate(
                {
                    "id": "msg_unknown",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {},
                }
            )

        # Add system messages if mode is not IGNORE
        if mode != SystemMessageMode.IGNORE and system_messages:
            for system_message in system_messages:
                system_text = system_message.data.get("text", str(system_message.data))
                system_content_block = (
                    MessageConverter.create_system_message_content_block(
                        system_text,
                        mode,
                        source="claude_code_sdk",
                        pretty_format=pretty_format,
                    )
                )
                if system_content_block:
                    response.content.insert(
                        0,
                        sdk_models.SystemMessageBlock.model_validate(
                            system_content_block
                        ),
                    )

        # If we have multiple assistant messages, append their content
        if len(assistant_messages) > 1:
            for additional_message in assistant_messages[1:]:
                for block in additional_message.content:
                    if isinstance(block, sdk_models.TextBlock):
                        response.content.append(
                            sdk_models.TextBlock.model_validate(
                                {"type": "text", "text": block.text}
                            )
                        )

        return response

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
        message_text: str,
        mode: SystemMessageMode = SystemMessageMode.FORWARD,
        source: str = "claude_code_sdk",
        pretty_format: bool = True,
    ) -> dict[str, Any] | None:
        """
        Create a system_message content block for non-streaming responses.

        Args:
            message_text: The system message text content
            mode: System message handling mode
            source: The source of the system message (default: "claude_code_sdk")
            pretty_format: Whether to use pretty formatting

        Returns:
            Content block dict for system message, or None if mode is IGNORE
        """
        if mode == SystemMessageMode.IGNORE:
            return None
        elif mode == SystemMessageMode.FORWARD:
            return {
                "type": "system_message",
                "text": message_text,
                "source": source,
            }
        elif mode == SystemMessageMode.FORMATTED:
            system_data = {
                "text": message_text,
                "source": source,
            }
            formatted_json = MessageConverter._format_json_data(
                system_data, pretty_format
            )
            escaped_json = MessageConverter._escape_content_for_xml(
                formatted_json, pretty_format
            )
            if pretty_format:
                formatted_text = (
                    f"<system_message>\n{escaped_json}\n</system_message>\n"
                )
            else:
                formatted_text = f"<system_message>{escaped_json}</system_message>"
            return {
                "type": "text",
                "text": formatted_text,
            }

    @staticmethod
    def create_system_message_chunks(
        message_text: str,
        mode: SystemMessageMode = SystemMessageMode.FORWARD,
        index: int = 0,
        source: str = "claude_code_sdk",
        pretty_format: bool = True,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Create streaming chunks for system messages using specified mode.

        Args:
            message_text: The system message text content
            mode: System message handling mode
            index: The content block index for the system message
            source: The source of the system message

        Returns:
            List of tuples (event_type, chunk) for system message streaming chunks
        """
        if mode == SystemMessageMode.IGNORE:
            return []
        elif mode == SystemMessageMode.FORWARD:
            return [
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "system_message",
                            "text": message_text,
                            "source": source,
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
        elif mode == SystemMessageMode.FORMATTED:
            system_data = {
                "text": message_text,
                "source": source,
            }
            formatted_json = MessageConverter._format_json_data(
                system_data, pretty_format
            )
            escaped_json = MessageConverter._escape_content_for_xml(
                formatted_json, pretty_format
            )
            if pretty_format:
                formatted_text = (
                    f"<system_message>\n{escaped_json}\n</system_message>\n"
                )
            else:
                formatted_text = f"<system_message>{escaped_json}</system_message>"
            return [
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": formatted_text},
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

    @staticmethod
    def create_result_message_content_block(
        result_message: sdk_models.ResultMessage,
        mode: SystemMessageMode = SystemMessageMode.FORWARD,
        source: str = "claude_code_sdk",
        pretty_format: bool = True,
    ) -> dict[str, Any] | None:
        """
        Create a result_message content block for non-streaming responses.

        Args:
            result_message: The ResultMessage from Claude SDK
            mode: System message handling mode
            source: The source of the result message (default: "claude_code_sdk")

        Returns:
            Content block dict for result message, or None if mode is IGNORE
        """
        if mode == SystemMessageMode.IGNORE:
            return None

        result_data = {
            "session_id": result_message.session_id,
            "stop_reason": result_message.stop_reason,
            "usage": result_message.usage.model_dump(),
            "total_cost_usd": result_message.total_cost_usd,
            "source": source,
        }

        if mode == SystemMessageMode.FORWARD:
            return {
                "type": "result_message",
                "data": result_data,
                "source": source,
            }
        elif mode == SystemMessageMode.FORMATTED:
            formatted_json = MessageConverter._format_json_data(
                result_data, pretty_format
            )
            escaped_json = MessageConverter._escape_content_for_xml(
                formatted_json, pretty_format
            )
            formatted_text = (
                f"<result_message>\n{escaped_json}\n</result_message>\n"
                if pretty_format
                else f"<result_message>{escaped_json}</result_message>"
            )
            return {"type": "text", "text": formatted_text}

    @staticmethod
    def create_result_message_chunks(
        result_message: sdk_models.ResultMessage,
        mode: SystemMessageMode = SystemMessageMode.FORWARD,
        index: int = 0,
        source: str = "claude_code_sdk",
        pretty_format: bool = True,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Create streaming chunks for result messages using specified mode.

        Args:
            result_message: The ResultMessage from Claude SDK
            mode: System message handling mode
            index: The content block index for the result message
            source: The source of the result message

        Returns:
            List of tuples (event_type, chunk) for result message streaming chunks
        """
        if mode == SystemMessageMode.IGNORE:
            return []

        result_data = {
            "session_id": result_message.session_id,
            "stop_reason": result_message.stop_reason,
            "usage": result_message.usage.model_dump(),
            "total_cost_usd": result_message.total_cost_usd,
            "source": source,
        }

        if mode == SystemMessageMode.FORWARD:
            return [
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "result_message",
                            "data": result_data,
                            "source": source,
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
        elif mode == SystemMessageMode.FORMATTED:
            formatted_json = MessageConverter._format_json_data(
                result_data, pretty_format
            )
            escaped_json = MessageConverter._escape_content_for_xml(
                formatted_json, pretty_format
            )
            formatted_text = (
                f"<result_message>\n{escaped_json}\n</result_message>\n"
                if pretty_format
                else f"<result_message>{escaped_json}</result_message>"
            )
            return [
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": formatted_text},
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
