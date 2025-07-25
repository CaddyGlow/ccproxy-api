"""Claude SDK service orchestration for business logic."""

from collections.abc import AsyncIterator
from typing import Any

import structlog
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    SystemMessage,
)

from ccproxy.adapters.openai import adapter
from ccproxy.auth.manager import AuthManager
from ccproxy.claude_sdk.client import ClaudeSDKClient
from ccproxy.claude_sdk.converter import MessageConverter
from ccproxy.claude_sdk.options import OptionsHandler
from ccproxy.config.settings import Settings
from ccproxy.core.errors import (
    AuthenticationError,
    ClaudeProxyError,
    ServiceUnavailableError,
)
from ccproxy.observability.access_logger import log_request_access
from ccproxy.observability.context import RequestContext, request_context
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
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize Claude SDK service.

        Args:
            sdk_client: Claude SDK client instance
            auth_manager: Authentication manager (optional)
            metrics: Prometheus metrics instance (optional)
            settings: Application settings (optional)
        """
        self.sdk_client = sdk_client or ClaudeSDKClient()
        self.auth_manager = auth_manager
        self.metrics = metrics
        self.settings = settings
        self.message_converter = MessageConverter()
        self.options_handler = OptionsHandler(settings=settings)

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
                    exc_info=True,
                )
                raise

        # Extract system message and create options
        system_message = self.options_handler.extract_system_message(messages)

        # Map model to Claude model
        model = adapter.map_openai_model_to_claude(model)

        options = self.options_handler.create_options(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            **kwargs,
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
            metrics=self.metrics,  # Pass metrics for active request tracking
        ) as ctx:
            try:
                if stream:
                    # For streaming, return the async iterator directly
                    # Pass context to streaming method
                    return self._stream_completion(
                        prompt, options, model, request_id, ctx
                    )
                else:
                    result = await self._complete_non_streaming(
                        prompt, options, model, request_id, ctx
                    )
                    return result

            except (ClaudeProxyError, ServiceUnavailableError) as e:
                # Log error via access logger (includes metrics)
                await log_request_access(
                    context=ctx,
                    method="POST",
                    error_message=str(e),
                    metrics=self.metrics,
                    error_type=type(e).__name__,
                )
                raise
            except AuthenticationError as e:
                logger.error(
                    "authentication_failed",
                    user_id=user_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                raise

    async def _complete_non_streaming(
        self,
        prompt: str,
        options: ClaudeCodeOptions,
        model: str,
        request_id: str | None = None,
        ctx: RequestContext | None = None,
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
        system_messages = []

        async for message in self.sdk_client.query_completion(
            prompt, options, request_id
        ):
            messages.append(message)
            if isinstance(message, AssistantMessage):
                assistant_message = message
            elif isinstance(message, ResultMessage):
                result_message = message
            elif isinstance(message, SystemMessage):
                # Collect SystemMessages for processing if configured
                if (
                    self.settings
                    and self.settings.claude.include_system_messages_in_stream
                ):
                    system_messages.append(message)
                else:
                    logger.debug(
                        "Ignoring SystemMessage in non-streaming response (configured to ignore)."
                    )

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

        logger.debug("claude_sdk_completion_received")
        # Convert to Anthropic format
        response = self.message_converter.convert_to_anthropic_response(
            assistant_message, result_message, model
        )

        # Add SystemMessages to response content if any were collected
        if system_messages and "content" in response:
            for system_message in system_messages:
                system_content_block = (
                    self.message_converter.create_system_message_content_block(
                        f"{system_message.subtype}: {system_message.data}"
                    )
                )
                response["content"].append(system_content_block)

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
        logger.debug(
            "claude_sdk_completion_completed",
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
        )

        # Update context with metrics if available
        if ctx:
            ctx.add_metadata(
                status_code=200,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cost_usd=cost_usd,
            )

            # Log comprehensive access log (includes Prometheus metrics)
            await log_request_access(
                context=ctx,
                status_code=200,
                method="POST",
                metrics=self.metrics,
            )

        return response

    async def _stream_completion(
        self,
        prompt: str,
        options: ClaudeCodeOptions,
        model: str,
        request_id: str | None = None,
        ctx: RequestContext | None = None,
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

        from claude_code_sdk import TextBlock, ToolResultBlock, ToolUseBlock

        first_chunk = True
        message_count = 0
        assistant_messages = []

        try:
            async for message in self.sdk_client.query_completion(
                prompt, options, request_id
            ):
                message_count += 1
                logger.debug(
                    "streaming_message_received",
                    message_count=message_count,
                    message_type=type(message).__name__,
                    request_id=request_id,
                )

                if first_chunk:
                    # Send initial message_start chunk
                    # We don't have input tokens here, so we pass 0.
                    # The SDK does not provide it at this stage.
                    # The input tokens will be updated in a final message_delta event.
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

                    # Iterate through content blocks and yield structured chunks for each
                    for i, block in enumerate(message.content):
                        # Handle Text Blocks
                        if isinstance(block, TextBlock) and block.text:
                            # Start a new text block
                            yield {
                                "event": "content_block_start",
                                "data": {
                                    "type": "content_block_start",
                                    "index": i,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            }
                            # Send the text content in a delta
                            yield {
                                "event": "content_block_delta",
                                "data": {
                                    "type": "content_block_delta",
                                    "index": i,
                                    "delta": {"type": "text_delta", "text": block.text},
                                },
                            }
                            # Stop the text block
                            yield {
                                "event": "content_block_stop",
                                "data": {"type": "content_block_stop", "index": i},
                            }

                        # --- Handle Tool Use Blocks ---
                        elif isinstance(block, ToolUseBlock):
                            tool_input = getattr(block, "input", {}) or {}
                            # Start a new tool_use_sdk block with all its data
                            yield {
                                "event": "content_block_start",
                                "data": {
                                    "type": "content_block_start",
                                    "index": i,
                                    "content_block": {
                                        "type": "tool_use_sdk",
                                        "id": getattr(block, "id", f"tool_{id(block)}"),
                                        "name": block.name,
                                        "input": tool_input,
                                        "source": "claude_code_sdk",
                                    },
                                },
                            }
                            # 2. Immediately stop the block (since we get it all at once)
                            yield {
                                "event": "content_block_stop",
                                "data": {"type": "content_block_stop", "index": i},
                            }

                        # --- Handle Tool Result Blocks ---
                        elif isinstance(block, ToolResultBlock):
                            is_error = getattr(block, "is_error", None)
                            # 1. Start a new tool_result_sdk block
                            yield {
                                "event": "content_block_start",
                                "data": {
                                    "type": "content_block_start",
                                    "index": i,
                                    "content_block": {
                                        "type": "tool_result_sdk",
                                        "tool_use_id": block.tool_use_id,
                                        "content": block.content,
                                        "is_error": is_error
                                        if is_error is not None
                                        else False,
                                        "source": "claude_code_sdk",
                                    },
                                },
                            }
                            # 2. Immediately stop the block
                            yield {
                                "event": "content_block_stop",
                                "data": {"type": "content_block_stop", "index": i},
                            }

                elif isinstance(message, SystemMessage):
                    # Handle SystemMessage based on configuration
                    if (
                        self.settings
                        and self.settings.claude.include_system_messages_in_stream
                    ):
                        # Process SystemMessage as a special text block with SDK indicator
                        system_text = f"{message.subtype}: {message.data}"
                        index = (
                            len(
                                [
                                    block
                                    for am in assistant_messages
                                    for block in getattr(am, "content", [])
                                ]
                            ),
                        )[0]

                        yield {
                            "event": "content_block_start",
                            "data": {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        }
                        # 2. Send the text content in a delta
                        yield {
                            "event": "content_block_delta",
                            "data": {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "text_delta", "text": system_text},
                            },
                        }
                        # 3. Stop the text block
                        yield {
                            "event": "content_block_stop",
                            "data": {"type": "content_block_stop", "index": index},
                        }

                        # system_chunks = (
                        #     self.message_converter.create_system_message_chunks(
                        #         system_text,
                        #         index=len(
                        #             [
                        #                 block
                        #                 for am in assistant_messages
                        #                 for block in getattr(am, "content", [])
                        #             ]
                        #         ),
                        #     )
                        # )

                        # Yield the system message chunks
                        # for event_type, chunk_data in system_chunks:
                        #     yield {"event": event_type, "data": chunk_data}
                    else:
                        # Legacy behavior: ignore with debug logging
                        logger.debug(
                            "Ignoring SystemMessage in streaming response (configured to ignore)."
                        )
                    continue

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
                    logger.debug(
                        "streaming_completion_completed",
                        model=model,
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                        cache_read_tokens=cache_read_tokens,
                        cache_write_tokens=cache_write_tokens,
                        cost_usd=cost_usd,
                        message_count=message_count,
                        request_id=request_id,
                    )

                    # Update context with metrics if available
                    if ctx:
                        ctx.add_metadata(
                            status_code=200,
                            tokens_input=tokens_input,
                            tokens_output=tokens_output,
                            cache_read_tokens=cache_read_tokens,
                            cache_write_tokens=cache_write_tokens,
                            cost_usd=cost_usd,
                        )

                        # Log comprehensive access log for streaming completion
                        await log_request_access(
                            context=ctx,
                            status_code=200,
                            method="POST",
                            metrics=self.metrics,
                            event_type="streaming_complete",
                        )

                    # Send final chunks with usage and cost information
                    final_chunks = self.message_converter.create_streaming_end_chunks(
                        stop_reason=getattr(message, "stop_reason", "end_turn")
                    )

                    # Add usage information to message_delta chunk
                    for event_type, chunk_data in final_chunks:
                        if chunk_data.get("type") == "message_delta":
                            usage_info = {}
                            if tokens_output:
                                usage_info["output_tokens"] = tokens_output
                            if cost_usd is not None:
                                usage_info["cost_usd"] = cost_usd

                            # Update the usage in the message_delta chunk
                            if "usage" not in chunk_data:
                                chunk_data["usage"] = {}
                            chunk_data["usage"].update(usage_info)

                        yield {"event": event_type, "data": chunk_data}

                    # Update the input tokens in the initial message_start message
                    # This is a workaround to provide complete usage data.
                    if tokens_input:
                        yield {
                            "event": "message_delta",
                            "data": {
                                "type": "message_delta",
                                "delta": {},
                                "usage": {"input_tokens": tokens_input},
                            },
                        }

                    break

        except asyncio.CancelledError:
            logger.debug("streaming_completion_cancelled", request_id=request_id)
            raise
        except Exception as e:
            logger.error(
                "streaming_completion_failed",
                error=str(e),
                error_type=type(e).__name__,
                request_id=request_id,
                exc_info=True,
            )
            # Don't yield error chunk - let exception propagate for proper HTTP error response
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
        logger.debug("user_auth_validation_start", user_id=user_id)

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

    async def list_models(self) -> dict[str, Any]:
        """
        List available Claude models and recent OpenAI models.

        Returns:
            Dictionary with combined list of models in mixed format
        """
        # Get Claude models
        supported_models = self.options_handler.get_supported_models()

        # Create Anthropic-style model entries
        anthropic_models = []
        for model_id in supported_models:
            anthropic_models.append(
                {
                    "type": "model",
                    "id": model_id,
                    "display_name": self._get_display_name(model_id),
                    "created_at": self._get_created_timestamp(model_id),
                }
            )

        # Add recent OpenAI models (GPT-4 variants and O1 models)
        openai_models = [
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 1715367049,
                "owned_by": "openai",
            },
            {
                "id": "gpt-4o-mini",
                "object": "model",
                "created": 1721172741,
                "owned_by": "openai",
            },
            {
                "id": "gpt-4-turbo",
                "object": "model",
                "created": 1712361441,
                "owned_by": "openai",
            },
            {
                "id": "gpt-4-turbo-preview",
                "object": "model",
                "created": 1706037777,
                "owned_by": "openai",
            },
            {
                "id": "o1",
                "object": "model",
                "created": 1734375816,
                "owned_by": "openai",
            },
            {
                "id": "o1-mini",
                "object": "model",
                "created": 1725649008,
                "owned_by": "openai",
            },
            {
                "id": "o1-preview",
                "object": "model",
                "created": 1725648897,
                "owned_by": "openai",
            },
            {
                "id": "o3",
                "object": "model",
                "created": 1744225308,
                "owned_by": "openai",
            },
            {
                "id": "o3-mini",
                "object": "model",
                "created": 1737146383,
                "owned_by": "openai",
            },
        ]

        # Return combined response in mixed format
        return {
            "data": anthropic_models + openai_models,
            "has_more": False,
            "object": "list",
        }

    def _get_display_name(self, model_id: str) -> str:
        """Get display name for a model ID."""
        display_names = {
            "claude-opus-4-20250514": "Claude Opus 4",
            "claude-sonnet-4-20250514": "Claude Sonnet 4",
            "claude-3-7-sonnet-20250219": "Claude Sonnet 3.7",
            "claude-3-5-sonnet-20241022": "Claude Sonnet 3.5 (New)",
            "claude-3-5-haiku-20241022": "Claude Haiku 3.5",
            "claude-3-5-haiku-latest": "Claude Haiku 3.5",
            "claude-3-5-sonnet-20240620": "Claude Sonnet 3.5 (Old)",
            "claude-3-haiku-20240307": "Claude Haiku 3",
            "claude-3-opus-20240229": "Claude Opus 3",
        }
        return display_names.get(model_id, model_id)

    def _get_created_timestamp(self, model_id: str) -> int:
        """Get created timestamp for a model ID."""
        timestamps = {
            "claude-opus-4-20250514": 1747526400,  # 2025-05-22
            "claude-sonnet-4-20250514": 1747526400,  # 2025-05-22
            "claude-3-7-sonnet-20250219": 1740268800,  # 2025-02-24
            "claude-3-5-sonnet-20241022": 1729555200,  # 2024-10-22
            "claude-3-5-haiku-20241022": 1729555200,  # 2024-10-22
            "claude-3-5-haiku-latest": 1729555200,  # 2024-10-22
            "claude-3-5-sonnet-20240620": 1718841600,  # 2024-06-20
            "claude-3-haiku-20240307": 1709769600,  # 2024-03-07
            "claude-3-opus-20240229": 1709164800,  # 2024-02-29
        }
        return timestamps.get(model_id, 1677610602)  # Default timestamp

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
