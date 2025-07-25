"""Handles processing of Claude SDK streaming responses."""

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import structlog

from ccproxy.claude_sdk import models as sdk_models
from ccproxy.claude_sdk.converter import MessageConverter
from ccproxy.config.claude import SystemMessageMode
from ccproxy.observability.context import RequestContext
from ccproxy.observability.metrics import PrometheusMetrics


logger = structlog.get_logger(__name__)


class ClaudeStreamProcessor:
    """Processes streaming responses from the Claude SDK."""

    def __init__(
        self,
        message_converter: MessageConverter,
        metrics: PrometheusMetrics | None = None,
    ) -> None:
        """Initialize the stream processor.

        Args:
            message_converter: Converter for message formats.
            metrics: Prometheus metrics instance.
        """
        self.message_converter = message_converter
        self.metrics = metrics

    async def process_stream(
        self,
        sdk_stream: AsyncIterator[
            sdk_models.UserMessage
            | sdk_models.AssistantMessage
            | sdk_models.SystemMessage
            | sdk_models.ResultMessage
        ],
        model: str,
        request_id: str | None,
        ctx: RequestContext | None,
        sdk_message_mode: SystemMessageMode,
        pretty_format: bool,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process the SDK stream and yields Anthropic-compatible streaming chunks.

        Args:
            sdk_stream: The async iterator of Pydantic SDK messages.
            model: The model name.
            request_id: The request ID for correlation.
            ctx: The request context for observability.
            sdk_message_mode: The mode for handling system messages.
            pretty_format: Whether to format content prettily.

        Yields:
            Anthropic-compatible streaming chunks.
        """
        message_id = f"msg_{uuid4()}"
        content_block_index = 0
        input_tokens = 0  # Will be updated by ResultMessage

        # Yield start chunks
        start_chunks = self.message_converter.create_streaming_start_chunks(
            message_id, model, input_tokens
        )
        for _, chunk in start_chunks:
            yield chunk

        async for message in sdk_stream:
            if isinstance(message, sdk_models.SystemMessage):
                if sdk_message_mode != SystemMessageMode.IGNORE:
                    chunks = self.message_converter.create_system_message_chunks(
                        message_text=message.data.get("text", ""),
                        mode=sdk_message_mode,
                        index=content_block_index,
                        pretty_format=pretty_format,
                    )
                    for _, chunk in chunks:
                        yield chunk
                    content_block_index += 1

            elif isinstance(message, sdk_models.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk_models.TextBlock):
                        yield {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield self.message_converter.create_streaming_delta_chunk(
                            block.text
                        )[1]
                        yield {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        content_block_index += 1
                    # TODO: Handle other block types like ToolUseBlock in streaming

            elif isinstance(message, sdk_models.ResultMessage):
                # Final message, contains metrics
                if ctx:
                    ctx.add_metadata(
                        status_code=200,
                        tokens_input=message.usage.input_tokens,
                        tokens_output=message.usage.output_tokens,
                        cache_read_tokens=message.usage.cache_read_input_tokens,
                        cache_write_tokens=message.usage.cache_creation_input_tokens,
                        cost_usd=message.total_cost_usd,
                    )

                end_chunks = self.message_converter.create_streaming_end_chunks(
                    stop_reason=message.stop_reason
                )
                # Update usage in the delta chunk
                delta_chunk = end_chunks[0][1]
                delta_chunk["usage"] = {"output_tokens": message.usage.output_tokens}

                yield delta_chunk
                yield end_chunks[1][1]  # message_stop
                break  # End of stream

        logger.debug("claude_sdk_stream_processing_completed", request_id=request_id)
