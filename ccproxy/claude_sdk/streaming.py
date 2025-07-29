"""Handles processing of Claude SDK streaming responses."""

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import structlog

from ccproxy.claude_sdk.converter import MessageConverter
from ccproxy.config.claude import SDKMessageMode
from ccproxy.models import claude_sdk as sdk_models
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
        sdk_message_mode: SDKMessageMode,
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
            logger.debug(
                "sdk_message_received",
                message_type=type(message).__name__,
                request_id=request_id,
                message_content=message.model_dump()
                if hasattr(message, "model_dump")
                else str(message)[:200],
            )

            if isinstance(message, sdk_models.SystemMessage):
                logger.debug(
                    "sdk_system_message_processing",
                    mode=sdk_message_mode.value,
                    subtype=message.subtype,
                    request_id=request_id,
                )
                if sdk_message_mode != SDKMessageMode.IGNORE:
                    chunks = self.message_converter._create_sdk_content_block_chunks(
                        sdk_object=message,
                        mode=sdk_message_mode,
                        index=content_block_index,
                        pretty_format=pretty_format,
                        xml_tag="system_message",
                    )
                    for _, chunk in chunks:
                        yield chunk
                    content_block_index += 1

            elif isinstance(message, sdk_models.AssistantMessage):
                logger.debug(
                    "sdk_assistant_message_processing",
                    content_blocks_count=len(message.content),
                    block_types=[type(block).__name__ for block in message.content],
                    request_id=request_id,
                )
                for block in message.content:
                    if isinstance(block, sdk_models.TextBlock):
                        logger.debug(
                            "sdk_text_block_processing",
                            text_length=len(block.text),
                            text_preview=block.text[:50],
                            block_index=content_block_index,
                            request_id=request_id,
                        )
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
                    elif isinstance(block, sdk_models.ToolUseBlock):
                        logger.debug(
                            "sdk_tool_use_block_processing",
                            tool_id=block.id,
                            tool_name=block.name,
                            input_keys=list(block.input.keys()) if block.input else [],
                            block_index=content_block_index,
                            mode=sdk_message_mode.value,
                            request_id=request_id,
                        )
                        chunks = (
                            self.message_converter._create_sdk_content_block_chunks(
                                sdk_object=block,
                                mode=sdk_message_mode,
                                index=content_block_index,
                                pretty_format=pretty_format,
                                xml_tag="tool_use_sdk",
                                sdk_block_converter=lambda obj: obj.to_sdk_block(),
                            )
                        )
                        for _, chunk in chunks:
                            yield chunk
                        content_block_index += 1
                    elif isinstance(block, sdk_models.ToolResultBlock):
                        logger.debug(
                            "sdk_tool_result_block_processing",
                            tool_use_id=block.tool_use_id,
                            is_error=block.is_error,
                            content_type=type(block.content).__name__
                            if block.content
                            else "None",
                            content_preview=str(block.content)[:100]
                            if block.content
                            else None,
                            block_index=content_block_index,
                            mode=sdk_message_mode.value,
                            request_id=request_id,
                        )
                        chunks = (
                            self.message_converter._create_sdk_content_block_chunks(
                                sdk_object=block,
                                mode=sdk_message_mode,
                                index=content_block_index,
                                pretty_format=pretty_format,
                                xml_tag="tool_result_sdk",
                                sdk_block_converter=lambda obj: obj.to_sdk_block(),
                            )
                        )
                        for _, chunk in chunks:
                            yield chunk
                        content_block_index += 1

            elif isinstance(message, sdk_models.UserMessage):
                logger.debug(
                    "sdk_user_message_processing",
                    content_blocks_count=len(message.content),
                    block_types=[type(block).__name__ for block in message.content],
                    request_id=request_id,
                )
                for block in message.content:
                    if isinstance(block, sdk_models.ToolResultBlock):
                        logger.debug(
                            "sdk_tool_result_block_processing",
                            tool_use_id=block.tool_use_id,
                            is_error=block.is_error,
                            content_type=type(block.content).__name__
                            if block.content
                            else "None",
                            content_preview=str(block.content)[:100]
                            if block.content
                            else None,
                            block_index=content_block_index,
                            mode=sdk_message_mode.value,
                            request_id=request_id,
                        )
                        chunks = (
                            self.message_converter._create_sdk_content_block_chunks(
                                sdk_object=block,
                                mode=sdk_message_mode,
                                index=content_block_index,
                                pretty_format=pretty_format,
                                xml_tag="tool_result_sdk",
                                sdk_block_converter=lambda obj: obj.to_sdk_block(),
                            )
                        )
                        for _, chunk in chunks:
                            yield chunk
                        content_block_index += 1
                    # Handle other UserMessage content types if needed in the future
                    else:
                        logger.debug(
                            "sdk_user_message_unsupported_block",
                            block_type=type(block).__name__,
                            block_index=content_block_index,
                            request_id=request_id,
                        )

            elif isinstance(message, sdk_models.ResultMessage):
                logger.debug(
                    "sdk_result_message_processing",
                    session_id=message.session_id,
                    stop_reason=message.stop_reason,
                    is_error=message.is_error,
                    duration_ms=message.duration_ms,
                    num_turns=message.num_turns,
                    total_cost_usd=message.total_cost_usd,
                    usage_available=message.usage is not None,
                    mode=sdk_message_mode.value,
                    request_id=request_id,
                )
                if sdk_message_mode != SDKMessageMode.IGNORE:
                    chunks = self.message_converter._create_sdk_content_block_chunks(
                        sdk_object=message,
                        mode=sdk_message_mode,
                        index=content_block_index,
                        pretty_format=pretty_format,
                        xml_tag="system_message",
                    )
                    for _, chunk in chunks:
                        yield chunk
                    content_block_index += 1

                    if ctx:
                        usage_model = message.usage_model
                        ctx.add_metadata(
                            status_code=200,
                            tokens_input=usage_model.input_tokens,
                            tokens_output=usage_model.output_tokens,
                            cache_read_tokens=usage_model.cache_read_input_tokens,
                            cache_write_tokens=usage_model.cache_creation_input_tokens,
                            cost_usd=message.total_cost_usd,
                        )

                end_chunks = self.message_converter.create_streaming_end_chunks(
                    stop_reason=message.stop_reason
                )
                # Update usage in the delta chunk
                delta_chunk = end_chunks[0][1]
                delta_chunk["usage"] = {
                    "output_tokens": message.usage_model.output_tokens
                }

                yield delta_chunk
                yield end_chunks[1][1]  # message_stop
                break  # End of stream
            else:
                logger.warning(  # type: ignore[unreachable]
                    "sdk_unknown_message_type",
                    message_type=type(message).__name__,
                    message_content=str(message)[:200],
                    request_id=request_id,
                )
        # Final message, contains metrics
        if ctx:
            # For streaming requests, manually trigger access log since the context
            # has already exited by the time we get here
            try:
                from ccproxy.observability.access_logger import (
                    log_request_access,
                )

                await log_request_access(
                    context=ctx,
                    status_code=200,
                    metrics=self.metrics,
                )
            except Exception as e:
                logger.warning(
                    "streaming_access_log_failed",
                    error=str(e),
                    request_id=request_id,
                )

        logger.debug("claude_sdk_stream_processing_completed", request_id=request_id)
