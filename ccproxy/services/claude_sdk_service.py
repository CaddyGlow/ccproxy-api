"""Claude SDK service orchestration for business logic."""

from collections.abc import AsyncIterator
from typing import Any

import structlog
from claude_code_sdk import ClaudeCodeOptions

from ccproxy.auth.manager import AuthManager
from ccproxy.claude_sdk.client import ClaudeSDKClient
from ccproxy.claude_sdk.converter import MessageConverter
from ccproxy.claude_sdk.manager import PoolManager
from ccproxy.claude_sdk.options import OptionsHandler
from ccproxy.claude_sdk.streaming import ClaudeStreamProcessor
from ccproxy.config.claude import SDKMessageMode
from ccproxy.config.settings import Settings
from ccproxy.core.errors import (
    ClaudeProxyError,
    ServiceUnavailableError,
)
from ccproxy.models import claude_sdk as sdk_models
from ccproxy.models.messages import MessageResponse
from ccproxy.observability.context import RequestContext
from ccproxy.observability.metrics import PrometheusMetrics
from ccproxy.utils.model_mapping import map_model_to_claude
from ccproxy.utils.simple_request_logger import write_request_log


logger = structlog.get_logger(__name__)


class ClaudeSDKService:
    """
    Service layer for Claude SDK operations orchestration.

    This class handles business logic coordination between the pure SDK client,
    authentication, metrics, and format conversion while maintaining clean
    separation of concerns.
    """

    def __init__(
        self,
        sdk_client: ClaudeSDKClient | None = None,
        auth_manager: AuthManager | None = None,
        metrics: PrometheusMetrics | None = None,
        settings: Settings | None = None,
        use_pool: bool = False,
        pool_manager: PoolManager | None = None,
    ) -> None:
        """
        Initialize Claude SDK service.

        Args:
            sdk_client: Claude SDK client instance
            auth_manager: Authentication manager (optional)
            metrics: Prometheus metrics instance (optional)
            settings: Application settings (optional)
            use_pool: Whether to use connection pooling for improved performance
            pool_manager: Pool manager for dependency injection (optional)
        """
        self.sdk_client = sdk_client or ClaudeSDKClient(
            use_pool=use_pool, settings=settings, pool_manager=pool_manager
        )
        self.auth_manager = auth_manager
        self.metrics = metrics
        self.settings = settings
        self.message_converter = MessageConverter()
        self.options_handler = OptionsHandler(settings=settings)
        self.stream_processor = ClaudeStreamProcessor(
            message_converter=self.message_converter,
            metrics=self.metrics,
        )

    async def create_completion(
        self,
        request_context: RequestContext,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> MessageResponse | AsyncIterator[dict[str, Any]]:
        """
        Create a completion using Claude SDK with business logic orchestration.

        Args:
            messages: List of messages in Anthropic format
            model: The model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            stream: Whether to stream responses
            request_context: Existing request context to use instead of creating new one
            **kwargs: Additional arguments

        Returns:
            Response dict or async iterator of response chunks if streaming

        Raises:
            ClaudeProxyError: If request fails
            ServiceUnavailableError: If service is unavailable
        """

        # Extract system message and create options
        system_message = self.options_handler.extract_system_message(messages)

        # Map model to Claude model
        model = map_model_to_claude(model)

        options = self.options_handler.create_options(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            **kwargs,
        )

        # Convert messages to prompt format
        prompt = self.message_converter.format_messages_to_prompt(messages)

        # Use existing context, but update metadata for this service (preserve original service_type)
        ctx = request_context
        ctx.add_metadata(
            endpoint="messages",
            model=model,
            streaming=stream,
        )
        # Use existing request ID from context
        request_id = ctx.request_id

        try:
            # Log SDK request parameters
            timestamp = ctx.get_log_timestamp_prefix() if ctx else None
            await self._log_sdk_request(
                request_id, prompt, options, model, stream, timestamp
            )

            if stream:
                # For streaming, return the async iterator directly
                # Access logging will be handled by the stream processor when ResultMessage is received
                return self._stream_completion(ctx, prompt, options, model, timestamp)
            else:
                result = await self._complete_non_streaming(
                    ctx, prompt, options, model, timestamp
                )
                return result
        except (ClaudeProxyError, ServiceUnavailableError) as e:
            # Add error info to context for automatic access logging
            ctx.add_metadata(error_message=str(e), error_type=type(e).__name__)
            raise

    async def _complete_non_streaming(
        self,
        ctx: RequestContext,
        prompt: str,
        options: "ClaudeCodeOptions",
        model: str,
        timestamp: str | None = None,
    ) -> MessageResponse:
        """
        Complete a non-streaming request with business logic.

        Args:
            prompt: The formatted prompt
            options: Claude SDK options
            model: The model being used

        Returns:
            Response in Anthropic format

        Raises:
            ClaudeProxyError: If completion fails
        """
        request_id = ctx.request_id
        logger.debug("claude_sdk_completion_start", request_id=request_id)

        messages = [
            m
            async for m in self.sdk_client.query_completion(prompt, options, request_id)
        ]

        result_message = next(
            (m for m in messages if isinstance(m, sdk_models.ResultMessage)), None
        )
        assistant_message = next(
            (m for m in messages if isinstance(m, sdk_models.AssistantMessage)), None
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

        logger.debug("claude_sdk_completion_received")
        mode = (
            self.settings.claude.sdk_message_mode
            if self.settings
            else SDKMessageMode.FORWARD
        )
        pretty_format = self.settings.claude.pretty_format if self.settings else True

        response = self.message_converter.convert_to_anthropic_response(
            assistant_message, result_message, model, mode, pretty_format
        )

        # Add other message types to the content block
        all_messages = [
            m
            for m in messages
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
                        # Only validate as SDKMessageMode if it's a system_message type
                        if content_block.get("type") == "system_message":
                            response.content.append(
                                sdk_models.SDKMessageMode.model_validate(content_block)
                            )
                        else:
                            # For other types (like text blocks in FORMATTED mode), create appropriate content block
                            if content_block.get("type") == "text":
                                response.content.append(
                                    sdk_models.TextBlock.model_validate(content_block)
                                )
                            else:
                                # Fallback for other content block types
                                logger.warning(
                                    "unknown_content_block_type",
                                    content_block_type=content_block.get("type"),
                                )
                elif isinstance(message, sdk_models.UserMessage):
                    for block in message.content:
                        if isinstance(block, sdk_models.ToolResultBlock):
                            response.content.append(block)

        cost_usd = result_message.total_cost_usd
        usage = result_message.usage_model

        # if cost_usd is not None and response.usage:
        #     response.usage.cost_usd = cost_usd

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
        )
        # Add success status to context for automatic access logging
        ctx.add_metadata(status_code=200)

        # Log SDK response
        if request_id:
            await self._log_sdk_response(request_id, response, timestamp)

        return response

    async def _stream_completion(
        self,
        ctx: RequestContext,
        prompt: str,
        options: "ClaudeCodeOptions",
        model: str,
        timestamp: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream completion responses with business logic.

        Args:
            prompt: The formatted prompt
            options: Claude SDK options
            model: The model being used
            ctx: Optional request context for metrics

        Yields:
            Response chunks in Anthropic format
        """
        request_id = ctx.request_id
        sdk_message_mode = (
            self.settings.claude.sdk_message_mode
            if self.settings
            else SDKMessageMode.FORWARD
        )
        pretty_format = self.settings.claude.pretty_format if self.settings else True

        sdk_stream = self.sdk_client.query_completion(prompt, options, request_id)

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

    async def _log_sdk_request(
        self,
        request_id: str,
        prompt: str,
        options: "ClaudeCodeOptions",
        model: str,
        stream: bool,
        timestamp: str | None = None,
    ) -> None:
        """Log SDK input parameters as JSON dump.

        Args:
            request_id: Request identifier
            prompt: The formatted prompt
            options: Claude SDK options
            model: The model being used
            stream: Whether streaming is enabled
            timestamp: Optional timestamp prefix
        """
        # timestamp is already provided from context, no need for fallback

        # JSON dump of the parameters passed to SDK completion
        sdk_request_data = {
            "prompt": prompt,
            "options": options.model_dump()
            if hasattr(options, "model_dump")
            else str(options),
            "model": model,
            "stream": stream,
            "request_id": request_id,
        }

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
        """Log SDK response result as JSON dump.

        Args:
            request_id: Request identifier
            result: The result from _complete_non_streaming
            timestamp: Optional timestamp prefix
        """
        # timestamp is already provided from context, no need for fallback

        # JSON dump of the result from _complete_non_streaming
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
        """Log streaming chunk as JSON dump.

        Args:
            request_id: Request identifier
            chunk: The streaming chunk from process_stream
            timestamp: Optional timestamp prefix
        """
        # timestamp is already provided from context, no need for fallback

        # Append streaming chunk as JSON to raw file
        import json

        from ccproxy.utils.simple_request_logger import append_streaming_log

        chunk_data = json.dumps(chunk, default=str) + "\n"
        await append_streaming_log(
            request_id=request_id,
            log_type="sdk_streaming",
            data=chunk_data.encode("utf-8"),
            timestamp=timestamp,
        )

    async def validate_health(self) -> bool:
        """
        Validate that the service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            return await self.sdk_client.validate_health()
        except Exception as e:
            logger.error(
                "health_check_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return False

    async def close(self) -> None:
        """Close the service and cleanup resources."""
        await self.sdk_client.close()

    async def __aenter__(self) -> "ClaudeSDKService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
