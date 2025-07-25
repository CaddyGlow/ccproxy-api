"""Claude SDK streaming processor for handling SDK message streams.

This module provides a dedicated processor for handling Claude SDK streaming responses,
extracting the complex streaming logic from the service layer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog
from claude_code_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from ccproxy.config.claude import SystemMessageMode
from ccproxy.observability.access_logger import log_request_access
from ccproxy.observability.context import RequestContext
from ccproxy.observability.metrics import PrometheusMetrics

from .converter import MessageConverter


logger = structlog.get_logger(__name__)


class ClaudeStreamProcessor:
    """Processes Claude SDK streaming responses into Anthropic API format."""

    def __init__(
        self,
        message_converter: MessageConverter,
        metrics: PrometheusMetrics | None = None,
    ) -> None:
        """Initialize the stream processor.

        Args:
            message_converter: Message converter for creating streaming chunks
            metrics: Optional metrics interface for logging
        """
        self.message_converter = message_converter
        self.metrics = metrics
        self.global_content_block_index = 0

    async def process_stream(
        self,
        sdk_stream: AsyncIterator[Any],
        model: str,
        request_id: str | None = None,
        ctx: RequestContext | None = None,
        sdk_message_mode: SystemMessageMode = SystemMessageMode.FORWARD,
        pretty_format: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process Claude SDK stream into Anthropic API streaming format.

        Args:
            sdk_stream: Claude SDK message stream
            model: Model name being used
            request_id: Optional request ID for logging
            ctx: Optional request context for metrics
            sdk_message_mode: How to handle SDK system messages
            pretty_format: Whether to use pretty formatting for XML content

        Yields:
            Streaming response chunks in Anthropic API format
        """
        first_chunk = True
        message_count = 0
        assistant_messages = []

        try:
            async for message in sdk_stream:
                message_count += 1
                logger.debug(
                    "streaming_message_received",
                    message_count=message_count,
                    message_type=type(message).__name__,
                    request_id=request_id,
                    message=message,
                )

                if first_chunk:
                    # Send initial message_start chunk
                    for (
                        event_type,
                        chunk_data,
                    ) in self.message_converter.create_streaming_start_chunks(
                        f"msg_{request_id}", model, 0
                    ):
                        yield {"event": event_type, "data": chunk_data}
                    first_chunk = False

                if isinstance(message, AssistantMessage):
                    assistant_messages.append(message)
                    async for chunk in self._process_assistant_message(
                        message, sdk_message_mode, pretty_format
                    ):
                        yield chunk

                elif isinstance(message, SystemMessage):
                    async for chunk in self._process_system_message(
                        message, sdk_message_mode, pretty_format
                    ):
                        yield chunk

                elif isinstance(message, UserMessage):
                    async for chunk in self._process_user_message(
                        message, sdk_message_mode, pretty_format
                    ):
                        yield chunk

                elif isinstance(message, ResultMessage):
                    async for chunk in self._process_result_message(
                        message, model, request_id, ctx, sdk_message_mode, pretty_format
                    ):
                        yield chunk
                    break  # ResultMessage is always the last message

        except Exception as e:
            logger.error(
                "streaming_completion_failed",
                error=str(e),
                error_type=type(e).__name__,
                request_id=request_id,
                exc_info=True,
            )
            raise

    async def _process_assistant_message(
        self,
        message: AssistantMessage,
        sdk_message_mode: SystemMessageMode,
        pretty_format: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process AssistantMessage content blocks."""
        for block in message.content:
            logger.debug("streaming_content_block", block=block)

            if isinstance(block, TextBlock) and getattr(block, "text", None):
                async for chunk in self._process_text_block(
                    block, sdk_message_mode, pretty_format
                ):
                    yield chunk

            elif isinstance(block, ToolUseBlock):
                async for chunk in self._process_tool_use_block(
                    block, sdk_message_mode, pretty_format
                ):
                    yield chunk

            elif isinstance(block, ToolResultBlock):
                async for chunk in self._process_tool_result_block(
                    block, sdk_message_mode, pretty_format
                ):
                    yield chunk
            else:
                logger.warning(
                    "streaming_content_block_unsupported_block_type",
                    block=block,
                )

    async def _process_text_block(
        self,
        block: TextBlock,
        sdk_message_mode: SystemMessageMode,
        pretty_format: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process TextBlock content."""
        text_content = block.text

        if sdk_message_mode == SystemMessageMode.FORMATTED:
            escaped_text = MessageConverter._escape_content_for_xml(
                block.text, pretty_format
            )
            if pretty_format:
                text_content = f"<text>\n{escaped_text}\n</text>\n"
            else:
                text_content = f"<text>{escaped_text}</text>"

        # Start text block
        yield {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": self.global_content_block_index,
                "content_block": {"type": "text", "text": ""},
            },
        }

        # Send text content
        yield {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": self.global_content_block_index,
                "delta": {"type": "text_delta", "text": text_content},
            },
        }

        # Stop text block
        yield {
            "event": "content_block_stop",
            "data": {
                "type": "content_block_stop",
                "index": self.global_content_block_index,
            },
        }
        self.global_content_block_index += 1

    async def _process_tool_use_block(
        self,
        block: ToolUseBlock,
        sdk_message_mode: SystemMessageMode,
        pretty_format: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process ToolUseBlock content."""
        if sdk_message_mode == SystemMessageMode.IGNORE:
            return

        tool_input = getattr(block, "input", {}) or {}
        tool_id = getattr(block, "id", f"tool_{id(block)}")

        if sdk_message_mode == SystemMessageMode.FORWARD:
            # Forward as tool_use_sdk content block
            yield {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": self.global_content_block_index,
                    "content_block": {
                        "type": "tool_use_sdk",
                        "id": tool_id,
                        "name": block.name,
                        "input": tool_input,
                        "source": "claude_code_sdk",
                    },
                },
            }
            yield {
                "event": "content_block_stop",
                "data": {
                    "type": "content_block_stop",
                    "index": self.global_content_block_index,
                },
            }

        elif sdk_message_mode == SystemMessageMode.FORMATTED:
            # Format as XML text
            tool_data = {
                "id": tool_id,
                "name": block.name,
                "input": tool_input,
            }
            formatted_json = MessageConverter._format_json_data(
                tool_data, pretty_format
            )
            escaped_json = MessageConverter._escape_content_for_xml(
                formatted_json, pretty_format
            )
            if pretty_format:
                formatted_text = f"<tool_use_sdk>\n{escaped_json}\n</tool_use_sdk>\n"
            else:
                formatted_text = f"<tool_use_sdk>{escaped_json}</tool_use_sdk>"

            async for chunk in self._send_text_content(formatted_text):
                yield chunk

        self.global_content_block_index += 1

    async def _process_tool_result_block(
        self,
        block: ToolResultBlock,
        sdk_message_mode: SystemMessageMode,
        pretty_format: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process ToolResultBlock content."""
        if sdk_message_mode == SystemMessageMode.IGNORE:
            return

        is_error = getattr(block, "is_error", None)

        if sdk_message_mode == SystemMessageMode.FORWARD:
            # Forward as tool_result_sdk content block
            yield {
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": self.global_content_block_index,
                    "content_block": {
                        "type": "tool_result_sdk",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": is_error if is_error is not None else False,
                        "source": "claude_code_sdk",
                    },
                },
            }
            yield {
                "event": "content_block_stop",
                "data": {
                    "type": "content_block_stop",
                    "index": self.global_content_block_index,
                },
            }

        elif sdk_message_mode == SystemMessageMode.FORMATTED:
            # Format as XML text
            tool_result_data = {
                "tool_use_id": block.tool_use_id,
                "content": block.content,
                "is_error": getattr(block, "is_error", False),
            }
            formatted_json = MessageConverter._format_json_data(
                tool_result_data, pretty_format
            )
            escaped_json = MessageConverter._escape_content_for_xml(
                formatted_json, pretty_format
            )
            if pretty_format:
                formatted_text = (
                    f"<tool_result_sdk>\n{escaped_json}\n</tool_result_sdk>\n"
                )
            else:
                formatted_text = f"<tool_result_sdk>{escaped_json}</tool_result_sdk>"

            async for chunk in self._send_text_content(formatted_text):
                yield chunk

        self.global_content_block_index += 1

    async def _process_system_message(
        self,
        message: SystemMessage,
        sdk_message_mode: SystemMessageMode,
        pretty_format: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process SystemMessage."""
        if sdk_message_mode == SystemMessageMode.IGNORE:
            logger.debug("Ignoring SystemMessage in streaming response (mode: ignore).")
            return

        # Extract text from system message data
        system_text = message.data.get("text", str(message.data))

        # Use message converter to create chunks
        system_chunks = self.message_converter.create_system_message_chunks(
            system_text,
            sdk_message_mode,
            index=self.global_content_block_index,
            source="claude_code_sdk",
            pretty_format=pretty_format,
        )

        for event_type, chunk_data in system_chunks:
            yield {"event": event_type, "data": chunk_data}

        self.global_content_block_index += 1

    async def _process_user_message(
        self,
        message: UserMessage,
        sdk_message_mode: SystemMessageMode,
        pretty_format: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process UserMessage (may contain ToolResultBlocks)."""
        if sdk_message_mode == SystemMessageMode.IGNORE:
            logger.debug("Ignoring UserMessage in streaming response (mode: ignore).")
            return

        # Process content blocks within UserMessage if they exist
        if (
            hasattr(message, "content")
            and hasattr(message.content, "__iter__")
            and not isinstance(message.content, str)
        ):
            for block in message.content:  # type: ignore[unreachable]
                if block.get("type") == "tool_result":
                    tool_result_block = ToolResultBlock(
                        tool_use_id=block.get("tool_use_id", ""),
                        content=block.get("content"),
                        is_error=block.get("is_error"),
                    )
                    async for chunk in self._process_tool_result_block(
                        tool_result_block, sdk_message_mode, pretty_format
                    ):
                        yield chunk

    async def _process_result_message(
        self,
        message: ResultMessage,
        model: str,
        request_id: str | None,
        ctx: RequestContext | None,
        sdk_message_mode: SystemMessageMode,
        pretty_format: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process ResultMessage (final message with usage info)."""
        # Handle ResultMessage based on mode
        if sdk_message_mode != SystemMessageMode.IGNORE:
            result_chunks = self.message_converter.create_result_message_chunks(
                message,
                sdk_message_mode,
                index=self.global_content_block_index,
                source="claude_code_sdk",
                pretty_format=pretty_format,
            )

            for event_type, chunk_data in result_chunks:
                yield {"event": event_type, "data": chunk_data}
            self.global_content_block_index += 1

        # Extract metrics from result message
        cost_usd = message.total_cost_usd
        tokens_input = tokens_output = cache_read_tokens = cache_write_tokens = None

        if message.usage:
            tokens_input = message.usage.get("input_tokens")
            tokens_output = message.usage.get("output_tokens")
            cache_read_tokens = message.usage.get("cache_read_input_tokens")
            cache_write_tokens = message.usage.get("cache_creation_input_tokens")

        logger.debug(
            "streaming_completion_completed",
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
        )

        # Update context with metrics
        if ctx:
            ctx.add_metadata(
                status_code=200,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cost_usd=cost_usd,
            )

            if self.metrics:
                await log_request_access(
                    context=ctx,
                    status_code=200,
                    method="POST",
                    metrics=self.metrics,
                    event_type="streaming_complete",
                )

        # Send final chunks with usage information
        final_chunks = self.message_converter.create_streaming_end_chunks(
            stop_reason=getattr(message, "stop_reason", "end_turn")
        )

        for event_type, chunk_data in final_chunks:
            if chunk_data.get("type") == "message_delta":
                # Add usage information to message_delta chunk
                usage_info = message.usage or {}
                if "usage" not in chunk_data:
                    chunk_data["usage"] = {}
                chunk_data["usage"].update(usage_info)

            yield {"event": event_type, "data": chunk_data}

        # Update input tokens in final message (workaround for complete usage data)
        if tokens_input:
            yield {
                "event": "message_delta",
                "data": {
                    "type": "message_delta",
                    "delta": {},
                    "usage": {"input_tokens": tokens_input},
                },
            }

    async def _send_text_content(self, text: str) -> AsyncIterator[dict[str, Any]]:
        """Helper to send text content as streaming chunks."""
        yield {
            "event": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": self.global_content_block_index,
                "content_block": {"type": "text", "text": ""},
            },
        }
        yield {
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": self.global_content_block_index,
                "delta": {"type": "text_delta", "text": text},
            },
        }
        yield {
            "event": "content_block_stop",
            "data": {
                "type": "content_block_stop",
                "index": self.global_content_block_index,
            },
        }


__all__ = ["ClaudeStreamProcessor"]
