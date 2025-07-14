"""Claude SDK service orchestration for business logic."""

from collections.abc import AsyncIterator
from typing import Any

import structlog
from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, ResultMessage

from ccproxy.auth.manager import AuthManager
from ccproxy.claude_sdk.client import ClaudeSDKClient
from ccproxy.claude_sdk.converter import MessageConverter
from ccproxy.claude_sdk.options import OptionsHandler
from ccproxy.core.errors import (
    ClaudeProxyError,
    ServiceUnavailableError,
)
from ccproxy.observability.context import request_context
from ccproxy.observability.metrics import PrometheusMetrics


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
    ) -> None:
        """
        Initialize Claude SDK service.

        Args:
            sdk_client: Claude SDK client instance
            auth_manager: Authentication manager (optional)
            metrics: Prometheus metrics instance (optional)
        """
        self.sdk_client = sdk_client or ClaudeSDKClient()
        self.auth_manager = auth_manager
        self.metrics = metrics
        self.message_converter = MessageConverter()
        self.options_handler = OptionsHandler()

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """
        Create a completion using Claude SDK with business logic orchestration.

        Args:
            messages: List of messages in Anthropic format
            model: The model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            stream: Whether to stream responses
            user_id: User identifier for auth/metrics
            **kwargs: Additional arguments

        Returns:
            Response dict or async iterator of response chunks if streaming

        Raises:
            ClaudeProxyError: If request fails
            ServiceUnavailableError: If service is unavailable
        """
        # Validate authentication if auth manager is configured
        if self.auth_manager and user_id:
            try:
                await self._validate_user_auth(user_id)
            except Exception as e:
                logger.error(
                    "authentication_failed",
                    user_id=user_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        # Extract system message and create options
        system_message = self.options_handler.extract_system_message(messages)
        options = self.options_handler.create_options(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            **kwargs,
        )

        # Validate model
        if not self.options_handler.validate_model(model):
            raise ClaudeProxyError(
                message=f"Unsupported model: {model}",
                error_type="invalid_request_error",
                status_code=400,
            )

        # Convert messages to prompt format
        prompt = self.message_converter.format_messages_to_prompt(messages)

        # Generate request ID for correlation
        from uuid import uuid4

        request_id = str(uuid4())

        # Use request context for observability
        endpoint = "messages"  # Claude SDK uses messages endpoint
        async with request_context(
            method="POST",
            path=f"/sdk/v1/{endpoint}",
            endpoint=endpoint,
            model=model,
            streaming=stream,
            service_type="claude_sdk_service",
        ) as ctx:
            # Record active request start
            if self.metrics:
                self.metrics.inc_active_requests()

            try:
                if stream:
                    # For streaming, return the async iterator directly
                    # Response time will be handled by the async context manager
                    return self._stream_completion(prompt, options, model, request_id)
                else:
                    result = await self._complete_non_streaming(
                        prompt, options, model, request_id
                    )
                    # Record response time after completion
                    if self.metrics:
                        self.metrics.record_response_time(
                            ctx.duration_seconds, model, endpoint, "claude_sdk_service"
                        )
                    return result

            except Exception as e:
                # Record error metrics if available
                if self.metrics:
                    self.metrics.record_error(
                        type(e).__name__, "messages", model, "claude_sdk_service"
                    )
                raise
            finally:
                # Record active request end
                if self.metrics:
                    self.metrics.dec_active_requests()

    async def _complete_non_streaming(
        self,
        prompt: str,
        options: ClaudeCodeOptions,
        model: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Complete a non-streaming request with business logic.

        Args:
            prompt: The formatted prompt
            options: Claude SDK options
            model: The model being used
            request_id: The request ID for metrics correlation

        Returns:
            Response in Anthropic format

        Raises:
            ClaudeProxyError: If completion fails
        """
        messages = []
        result_message = None
        assistant_message = None

        async for message in self.sdk_client.query_completion(
            prompt, options, request_id
        ):
            messages.append(message)
            if isinstance(message, AssistantMessage):
                assistant_message = message
            elif isinstance(message, ResultMessage):
                result_message = message

        # Get Claude API call timing
        claude_api_call_ms = self.sdk_client.get_last_api_call_time_ms()

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

        logger.info("claude_sdk_completion", result_message)
        # Convert to Anthropic format
        response = self.message_converter.convert_to_anthropic_response(
            assistant_message, result_message, model
        )

        # Extract token usage and cost from result message using direct access
        cost_usd = result_message.total_cost_usd
        if result_message.usage:
            tokens_input = result_message.usage.get("input_tokens")
            tokens_output = result_message.usage.get("output_tokens")
            cache_read_tokens = result_message.usage.get("cache_read_input_tokens")
            cache_write_tokens = result_message.usage.get("cache_creation_input_tokens")
        else:
            tokens_input = tokens_output = cache_read_tokens = cache_write_tokens = None

        # Add cost to response usage section if available
        if cost_usd is not None and "usage" in response:
            response["usage"]["cost_usd"] = cost_usd

        # Log metrics for observability
        logger.info(
            "claude_sdk_completion",
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
        )

        # Record Prometheus metrics if available
        if self.metrics:
            self.metrics.record_request(
                "POST", "messages", model, "200", "claude_sdk_service"
            )

            if tokens_input:
                self.metrics.record_tokens(
                    tokens_input, "input", model, "claude_sdk_service"
                )
            if tokens_output:
                self.metrics.record_tokens(
                    tokens_output, "output", model, "claude_sdk_service"
                )
            if cache_read_tokens:
                self.metrics.record_tokens(
                    cache_read_tokens, "cache_read", model, "claude_sdk_service"
                )
            if cache_write_tokens:
                self.metrics.record_tokens(
                    cache_write_tokens, "cache_write", model, "claude_sdk_service"
                )
            if cost_usd:
                self.metrics.record_cost(cost_usd, model, "total", "claude_sdk_service")

        return response

    async def _stream_completion(
        self,
        prompt: str,
        options: ClaudeCodeOptions,
        model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream completion responses with business logic.

        Args:
            prompt: The formatted prompt
            options: Claude SDK options
            model: The model being used

        Yields:
            Response chunks in Anthropic format
        """
        import asyncio

        first_chunk = True
        message_count = 0
        assistant_messages = []

        try:
            async for message in self.sdk_client.query_completion(
                prompt, options, request_id
            ):
                message_count += 1
                logger.debug(
                    "claude_sdk_streaming_message",
                    message_count=message_count,
                    message_type=type(message).__name__,
                    request_id=request_id,
                )

                if isinstance(message, AssistantMessage):
                    assistant_messages.append(message)

                    if first_chunk:
                        # Send initial chunk
                        yield self.message_converter.create_streaming_start_chunk(
                            f"msg_{id(message)}", model
                        )
                        first_chunk = False

                    # Send content delta
                    text_content = self.message_converter.extract_text_from_content(
                        message.content
                    )
                    if text_content:
                        yield self.message_converter.create_streaming_delta_chunk(
                            text_content
                        )

                elif isinstance(message, ResultMessage):
                    # Get Claude API call timing
                    claude_api_call_ms = self.sdk_client.get_last_api_call_time_ms()

                    # Extract cost and tokens from result message using direct access
                    cost_usd = message.total_cost_usd
                    if message.usage:
                        tokens_input = message.usage.get("input_tokens")
                        tokens_output = message.usage.get("output_tokens")
                        cache_read_tokens = message.usage.get("cache_read_input_tokens")
                        cache_write_tokens = message.usage.get(
                            "cache_creation_input_tokens"
                        )
                    else:
                        tokens_input = tokens_output = cache_read_tokens = (
                            cache_write_tokens
                        ) = None

                    # Log streaming completion metrics
                    logger.info(
                        "claude_sdk_streaming_completion",
                        model=model,
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        cost_usd=cost_usd,
                        message_count=message_count,
                        request_id=request_id,
                    )

                    # Record Prometheus metrics if available
                    if self.metrics:
                        self.metrics.record_request(
                            "POST", "messages", model, "200", "claude_sdk_service"
                        )

                        if tokens_input:
                            self.metrics.record_tokens(
                                tokens_input, "input", model, "claude_sdk_service"
                            )
                        if tokens_output:
                            self.metrics.record_tokens(
                                tokens_output, "output", model, "claude_sdk_service"
                            )
                        if cache_read_tokens:
                            self.metrics.record_tokens(
                                cache_read_tokens,
                                "cache_read",
                                model,
                                "claude_sdk_service",
                            )
                        if cache_write_tokens:
                            self.metrics.record_tokens(
                                cache_write_tokens,
                                "cache_write",
                                model,
                                "claude_sdk_service",
                            )
                        if cost_usd:
                            self.metrics.record_cost(
                                cost_usd, model, "total", "claude_sdk_service"
                            )

                    # Send final chunk with usage and cost information
                    final_chunk = self.message_converter.create_streaming_end_chunk()

                    # Add usage information to final chunk
                    if tokens_input or tokens_output or cost_usd:
                        usage_info = {}
                        if tokens_input:
                            usage_info["input_tokens"] = tokens_input
                        if tokens_output:
                            usage_info["output_tokens"] = tokens_output
                        if cost_usd is not None:
                            usage_info["cost_usd"] = cost_usd

                        # Update the usage in the final chunk
                        final_chunk["usage"].update(usage_info)

                    yield final_chunk

                    break

        except asyncio.CancelledError:
            logger.info("stream_completion_cancelled", request_id=request_id)
            raise
        except Exception as e:
            logger.error(
                "stream_completion_error",
                error=str(e),
                error_type=type(e).__name__,
                request_id=request_id,
            )
            yield self.message_converter.create_streaming_end_chunk("error")
            raise

    async def _validate_user_auth(self, user_id: str) -> None:
        """
        Validate user authentication.

        Args:
            user_id: User identifier

        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.auth_manager:
            return

        # Implement authentication validation logic
        # This is a placeholder for future auth integration
        logger.debug("validating_user_auth", user_id=user_id)

    def _calculate_cost(
        self,
        tokens_input: int | None,
        tokens_output: int | None,
        model: str | None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> float | None:
        """
        Calculate cost in USD for the given token usage including cache tokens.

        Note: This method is provided for consistency, but the Claude SDK already
        provides accurate cost calculation in ResultMessage.total_cost_usd which
        should be preferred when available.

        Args:
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            model: Model name for pricing lookup
            cache_read_tokens: Number of cache read tokens
            cache_write_tokens: Number of cache write tokens

        Returns:
            Cost in USD or None if calculation not possible
        """
        from ccproxy.utils.cost_calculator import calculate_token_cost

        return calculate_token_cost(
            tokens_input, tokens_output, model, cache_read_tokens, cache_write_tokens
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List available Claude models.

        Returns:
            List of available models in Anthropic format
        """
        supported_models = self.options_handler.get_supported_models()
        models = []

        for model_id in supported_models:
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 1677610602,  # Static timestamp
                    "owned_by": "anthropic",
                }
            )

        return models

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
                "claude_sdk_service_health_check_failed",
                error=str(e),
                error_type=type(e).__name__,
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
