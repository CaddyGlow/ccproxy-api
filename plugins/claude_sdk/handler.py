"""Claude SDK handler for orchestrating SDK operations.

This module contains the core business logic migrated from claude_sdk_service.py,
handling SDK operations while maintaining clean separation of concerns.
"""

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from claude_code_sdk import ClaudeCodeOptions

from ccproxy.auth.manager import AuthManager
from ccproxy.core.errors import ClaudeProxyError, ServiceUnavailableError
from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.messages import TextContentBlock
from ccproxy.observability.context import RequestContext
from ccproxy.observability.metrics import PrometheusMetrics
from ccproxy.utils.model_mapping import map_model_to_claude
from ccproxy.utils.simple_request_logger import append_streaming_log, write_request_log
from plugins.claude_sdk.exceptions import StreamTimeoutError
from plugins.claude_sdk.manager import SessionManager
from plugins.claude_sdk.options import OptionsHandler

from . import models as sdk_models
from .client import ClaudeSDKClient
from .config import ClaudeSDKSettings, SDKMessageMode
from .converter import MessageConverter
from .models import MessageResponse, SDKMessage, create_sdk_message
from .streaming import ClaudeStreamProcessor


logger = get_plugin_logger()


def _convert_sdk_message_mode(core_mode: Any) -> SDKMessageMode:
    """Convert core SDKMessageMode to plugin SDKMessageMode."""
    if hasattr(core_mode, "value"):
        # Convert enum value to plugin enum
        if core_mode.value == "forward":
            return SDKMessageMode.FORWARD
        elif core_mode.value == "ignore":
            return SDKMessageMode.IGNORE
        elif core_mode.value == "formatted":
            return SDKMessageMode.FORMATTED
    return SDKMessageMode.FORWARD  # Default fallback


class ClaudeSDKHandler:
    """
    Handler for Claude SDK operations orchestration.

    This class encapsulates the business logic for SDK operations,
    migrated from the original claude_sdk_service.py.
    """

    def __init__(
        self,
        config: ClaudeSDKSettings,
        sdk_client: ClaudeSDKClient | None = None,
        auth_manager: AuthManager | None = None,
        metrics: PrometheusMetrics | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        """Initialize Claude SDK handler."""
        self.config = config
        self.sdk_client = sdk_client or ClaudeSDKClient(
            config=config, session_manager=session_manager
        )
        self.auth_manager = auth_manager
        self.metrics = metrics
        self.message_converter = MessageConverter()
        self.options_handler = OptionsHandler(config=config)
        self.stream_processor = ClaudeStreamProcessor(
            message_converter=self.message_converter,
            metrics=self.metrics,
        )

    def _convert_messages_to_sdk_message(
        self, messages: list[dict[str, Any]], session_id: str | None = None
    ) -> SDKMessage:
        """Convert list of Anthropic messages to single SDKMessage."""
        # Find the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg
                break

        if not last_user_message:
            raise ClaudeProxyError(
                message="No user message found in messages list",
                error_type="invalid_request_error",
                status_code=400,
            )

        # Extract text content from the message
        content = last_user_message.get("content", "")
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "\n".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)

        return create_sdk_message(content=content, session_id=session_id)

    async def _capture_session_metadata(
        self,
        ctx: RequestContext,
        session_id: str | None,
        options: ClaudeCodeOptions,
    ) -> None:
        """Capture session metadata for access logging."""
        if (
            session_id
            and hasattr(self.sdk_client, "_session_manager")
            and self.sdk_client._session_manager
        ):
            try:
                session_client = (
                    await self.sdk_client._session_manager.get_session_client(
                        session_id, options
                    )
                )
                if session_client:
                    # Determine if session pool is enabled
                    session_pool_enabled = (
                        hasattr(self.sdk_client._session_manager, "session_pool")
                        and self.sdk_client._session_manager.session_pool is not None
                        and hasattr(
                            self.sdk_client._session_manager.session_pool, "config"
                        )
                        and self.sdk_client._session_manager.session_pool.config.enabled
                    )

                    # Add session metadata to context
                    ctx.add_metadata(
                        session_type="session_pool"
                        if session_pool_enabled
                        else "direct",
                        session_status=session_client.status.value,
                        session_age_seconds=session_client.metrics.age_seconds,
                        session_message_count=session_client.metrics.message_count,
                        session_client_id=session_client.client_id,
                        session_pool_enabled=session_pool_enabled,
                        session_idle_seconds=session_client.metrics.idle_seconds,
                        session_error_count=session_client.metrics.error_count,
                        session_is_new=session_client.is_newly_created,
                    )
            except Exception as e:
                logger.warning(
                    "failed_to_capture_session_metadata",
                    session_id=session_id,
                    error=str(e),
                    exc_info=e,
                )
        else:
            # Add basic session metadata for direct connections
            ctx.add_metadata(
                session_type="direct",
                session_pool_enabled=False,
                session_is_new=True,
            )

    async def create_completion(
        self,
        request_context: RequestContext,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> MessageResponse | AsyncIterator[dict[str, Any]]:
        """Create a completion using Claude SDK with business logic orchestration."""
        # Extract system message and create options
        system_message = self.options_handler.extract_system_message(messages)

        # Map model to Claude model
        model = map_model_to_claude(model)

        options = self.options_handler.create_options(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            session_id=session_id,
            **kwargs,
        )

        # Use existing context
        ctx = request_context
        metadata = {
            "endpoint": "messages",
            "model": model,
            "streaming": stream,
        }
        if session_id:
            metadata["session_id"] = session_id
        ctx.add_metadata(**metadata)
        request_id = ctx.request_id

        try:
            # Log SDK request parameters
            timestamp = ctx.get_log_timestamp_prefix() if ctx else None
            await self._log_sdk_request(
                request_id, messages, options, model, stream, session_id, timestamp
            )

            if stream:
                return self._stream_completion(
                    ctx, messages, options, model, session_id, timestamp
                )
            else:
                result = await self._complete_non_streaming(
                    ctx, messages, options, model, session_id, timestamp
                )
                return result
        except (ClaudeProxyError, ServiceUnavailableError) as e:
            ctx.add_metadata(error_message=str(e), error_type=type(e).__name__)
            raise

    async def _complete_non_streaming(
        self,
        ctx: RequestContext,
        messages: list[dict[str, Any]],
        options: ClaudeCodeOptions,
        model: str,
        session_id: str | None = None,
        timestamp: str | None = None,
    ) -> MessageResponse:
        """Complete a non-streaming request with business logic."""
        request_id = ctx.request_id
        logger.debug("completion_start", request_id=request_id)

        # Convert messages to single SDKMessage
        sdk_message = self._convert_messages_to_sdk_message(messages, session_id)

        # Get stream handle
        stream_handle = await self.sdk_client.query_completion(
            sdk_message, options, request_id, session_id
        )

        # Capture session metadata
        await self._capture_session_metadata(ctx, session_id, options)

        # Create a listener and collect all messages
        sdk_messages = []
        async for m in stream_handle.create_listener():
            sdk_messages.append(m)

        result_message = next(
            (m for m in sdk_messages if isinstance(m, sdk_models.ResultMessage)), None
        )
        assistant_message = next(
            (m for m in sdk_messages if isinstance(m, sdk_models.AssistantMessage)),
            None,
        )

        if result_message is None:
            raise ClaudeProxyError(
                message="No result message received from Claude SDK",
                error_type="internal_server_error",
                status_code=500,
            )

        if assistant_message is None:
            raise ClaudeProxyError(
                message="No assistant response received from Claude SDK",
                error_type="internal_server_error",
                status_code=500,
            )

        logger.debug("completion_received")
        mode = (
            _convert_sdk_message_mode(self.config.sdk_message_mode)
            if self.config
            else SDKMessageMode.FORWARD
        )
        pretty_format = self.config.pretty_format if self.config else True

        response = self.message_converter.convert_to_anthropic_response(
            assistant_message, result_message, model, mode, pretty_format
        )

        # Add other message types to the content block
        all_messages = [
            m
            for m in sdk_messages
            if not isinstance(m, sdk_models.AssistantMessage | sdk_models.ResultMessage)
        ]

        if mode != SDKMessageMode.IGNORE and response.content:
            for message in all_messages:
                if isinstance(message, sdk_models.SystemMessage):
                    content_block = self.message_converter._create_sdk_content_block(
                        sdk_object=message,
                        mode=mode,
                        pretty_format=pretty_format,
                        xml_tag="system_message",
                        forward_converter=lambda obj: {
                            "type": "system_message",
                            "text": obj.model_dump_json(),
                        },
                    )
                    if content_block:
                        if content_block.get("type") == "system_message":
                            response.content.append(
                                sdk_models.SDKMessageMode.model_validate(content_block)
                            )
                        else:
                            if content_block.get("type") == "text":
                                # Convert SDK TextBlock to core TextContentBlock
                                response.content.append(
                                    TextContentBlock(
                                        type="text", text=content_block["text"]
                                    )
                                )
                            else:
                                logger.warning(
                                    "unknown_content_block_type",
                                    content_block_type=content_block.get("type"),
                                )
                elif isinstance(message, sdk_models.UserMessage):
                    for block in message.content:
                        if isinstance(block, sdk_models.ToolResultBlock):
                            # Convert SDK ToolResultBlock to ToolResultSDKBlock
                            response.content.append(
                                sdk_models.ToolResultSDKBlock(
                                    type="tool_result_sdk",
                                    tool_use_id=block.tool_use_id,
                                    content=block.content,
                                    is_error=block.is_error,
                                    source="claude_code_sdk",
                                )
                            )

        cost_usd = result_message.total_cost_usd
        usage = result_message.usage_model

        logger.debug(
            "claude_sdk_completion_completed",
            model=model,
            tokens_input=usage.input_tokens,
            tokens_output=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
        )

        ctx.add_metadata(
            status_code=200,
            tokens_input=usage.input_tokens,
            tokens_output=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens,
            cost_usd=cost_usd,
            session_id=result_message.session_id,
            num_turns=result_message.num_turns,
        )

        # Log SDK response
        if request_id:
            await self._log_sdk_response(request_id, response, timestamp)

        return response

    async def _stream_completion(
        self,
        ctx: RequestContext,
        messages: list[dict[str, Any]],
        options: ClaudeCodeOptions,
        model: str,
        session_id: str | None = None,
        timestamp: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream completion responses with business logic."""
        request_id = ctx.request_id
        sdk_message_mode = (
            _convert_sdk_message_mode(self.config.sdk_message_mode)
            if self.config
            else SDKMessageMode.FORWARD
        )
        pretty_format = self.config.pretty_format if self.config else True

        # Convert messages to single SDKMessage
        sdk_message = self._convert_messages_to_sdk_message(messages, session_id)

        # Get stream handle
        stream_handle = await self.sdk_client.query_completion(
            sdk_message, options, request_id, session_id
        )

        # Store handle in session client if available
        if (
            session_id
            and hasattr(self.sdk_client, "_session_manager")
            and self.sdk_client._session_manager
        ):
            try:
                session_client = (
                    await self.sdk_client._session_manager.get_session_client(
                        session_id, options
                    )
                )
                if session_client:
                    session_client.active_stream_handle = stream_handle
            except Exception as e:
                logger.warning(
                    "failed_to_store_stream_handle",
                    session_id=session_id,
                    error=str(e),
                    exc_info=e,
                )

        # Capture session metadata
        await self._capture_session_metadata(ctx, session_id, options)

        # Create a listener for this stream
        sdk_stream = stream_handle.create_listener()

        try:
            async for chunk in self.stream_processor.process_stream(
                sdk_stream=sdk_stream,
                model=model,
                request_id=request_id,
                ctx=ctx,
                sdk_message_mode=sdk_message_mode,
                pretty_format=pretty_format,
            ):
                # Log streaming chunk
                if request_id:
                    await self._log_sdk_streaming_chunk(request_id, chunk, timestamp)
                yield chunk
        except GeneratorExit:
            logger.debug(
                "claude_sdk_handler_client_disconnected",
                request_id=request_id,
                session_id=session_id,
            )
            raise
        except StreamTimeoutError as e:
            # Send error events to the client
            logger.error(
                "stream_timeout_error",
                message=str(e),
                session_id=e.session_id,
                timeout_seconds=e.timeout_seconds,
                request_id=request_id,
            )

            error_message_id = f"msg_error_{uuid4()}"

            # Yield error events
            yield {
                "type": "message_start",
                "message": {
                    "id": error_message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": "error",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }

            yield {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }

            error_text = f"Error: {e}"
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": error_text},
            }

            yield {"type": "content_block_stop", "index": 0}

            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "error", "stop_sequence": None},
                "usage": {"output_tokens": len(error_text.split())},
            }

            yield {"type": "message_stop"}

            ctx.add_metadata(
                status_code=504,
                error_message=str(e),
                error_type="stream_timeout",
                session_id=e.session_id,
            )

    async def _log_sdk_request(
        self,
        request_id: str,
        messages: list[dict[str, Any]],
        options: ClaudeCodeOptions,
        model: str,
        stream: bool,
        session_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Log SDK input parameters as JSON dump."""
        sdk_request_data = {
            "messages": messages,
            "options": options,
            "stream": stream,
            "request_id": request_id,
        }
        if session_id:
            sdk_request_data["session_id"] = session_id

        await write_request_log(
            request_id=request_id,
            log_type="sdk_request",
            data=sdk_request_data,
            timestamp=timestamp,
        )

    async def _log_sdk_response(
        self,
        request_id: str,
        result: Any,
        timestamp: str | None = None,
    ) -> None:
        """Log SDK response result as JSON dump."""
        sdk_response_data = {
            "result": result.model_dump()
            if hasattr(result, "model_dump")
            else str(result),
        }

        await write_request_log(
            request_id=request_id,
            log_type="sdk_response",
            data=sdk_response_data,
            timestamp=timestamp,
        )

    async def _log_sdk_streaming_chunk(
        self,
        request_id: str,
        chunk: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """Log streaming chunk as JSON dump."""
        import json

        chunk_data = json.dumps(chunk, default=str) + "\n"
        await append_streaming_log(
            request_id=request_id,
            log_type="sdk_streaming",
            data=chunk_data.encode("utf-8"),
            timestamp=timestamp,
        )

    async def validate_health(self) -> bool:
        """Validate that the handler is healthy."""
        try:
            return await self.sdk_client.validate_health()
        except Exception as e:
            logger.error(
                "health_check_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            return False

    async def interrupt_session(self, session_id: str) -> bool:
        """Interrupt a Claude session due to client disconnection."""
        return await self.sdk_client.interrupt_session(session_id)

    async def close(self) -> None:
        """Close the handler and cleanup resources."""
        await self.sdk_client.close()
