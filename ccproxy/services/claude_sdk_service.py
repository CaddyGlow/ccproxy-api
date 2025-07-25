"""Claude SDK service orchestration for business logic."""

from collections.abc import AsyncIterator
from typing import Any

import structlog
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    SystemMessage,
    ToolResultBlock,
    UserMessage,
)

from ccproxy.adapters.openai import adapter
from ccproxy.auth.manager import AuthManager
from ccproxy.claude_sdk.client import ClaudeSDKClient
from ccproxy.claude_sdk.converter import MessageConverter
from ccproxy.claude_sdk.options import OptionsHandler
from ccproxy.config.claude import SystemMessageMode
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
        user_messages = []

        async for message in self.sdk_client.query_completion(
            prompt, options, request_id
        ):
            messages.append(message)

            if isinstance(message, AssistantMessage):
                assistant_message = message
            elif isinstance(message, ResultMessage):
                result_message = message
            elif isinstance(message, SystemMessage):
                # Collect SystemMessages for processing based on mode
                mode = (
                    self.settings.claude.system_message_mode
                    if self.settings
                    else SystemMessageMode.FORWARD
                )
                if mode != SystemMessageMode.IGNORE:
                    system_messages.append(message)
                else:
                    logger.debug(
                        "Ignoring SystemMessage in non-streaming response (mode: ignore)."
                    )
            elif isinstance(message, UserMessage):
                # Collect UserMessages (which contain ToolResultBlocks)
                mode = (
                    self.settings.claude.system_message_mode
                    if self.settings
                    else SystemMessageMode.FORWARD
                )
                if mode != SystemMessageMode.IGNORE:
                    user_messages.append(message)
                else:
                    logger.debug(
                        "Ignoring UserMessage in non-streaming response (mode: ignore)."
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
        # Get system message mode
        mode = (
            self.settings.claude.system_message_mode
            if self.settings
            else SystemMessageMode.FORWARD
        )

        # Get formatting settings
        pretty_format = self.settings.claude.pretty_format if self.settings else True

        # Convert to Anthropic format with mode and formatting
        response = self.message_converter.convert_to_anthropic_response(
            assistant_message, result_message, model, mode, pretty_format
        )

        # Add SystemMessages to response content if any were collected
        if system_messages and "content" in response:
            for system_message in system_messages:
                # Extract text from system message data
                system_text = system_message.data.get("text", str(system_message.data))
                system_content_block = (
                    self.message_converter.create_system_message_content_block(
                        system_text,
                        mode,
                        "claude_code_sdk",
                        pretty_format,
                    )
                )
                if (
                    system_content_block is not None
                ):  # Handle IGNORE mode returning None
                    response["content"].append(system_content_block)

        # Add UserMessages (containing ToolResultBlocks) to response content if any were collected
        if mode != SystemMessageMode.IGNORE and user_messages and "content" in response:
            for user_message in user_messages:
                # Process content blocks within UserMessage
                if (
                    hasattr(user_message, "content")
                    and hasattr(user_message.content, "__iter__")
                    and not isinstance(user_message.content, str)
                ):
                    for block in user_message.content:  # type: ignore[unreachable]
                        if isinstance(block, ToolResultBlock):
                            if mode == SystemMessageMode.FORWARD:
                                is_error = getattr(block, "is_error", None)
                                tool_result_content_block = {
                                    "type": "tool_result_sdk",
                                    "tool_use_id": block.tool_use_id,
                                    "content": block.content
                                    if isinstance(block.content, str)
                                    else "",
                                    "is_error": is_error
                                    if is_error is not None
                                    else False,
                                    "source": "claude_code_sdk",
                                }
                                response["content"].append(tool_result_content_block)
                            elif mode == SystemMessageMode.FORMATTED:
                                tool_result_data = {
                                    "tool_use_id": block.tool_use_id,
                                    "content": block.content
                                    if isinstance(block.content, str)
                                    else "",
                                    "is_error": getattr(block, "is_error", False),
                                }
                                formatted_json = MessageConverter._format_json_data(
                                    tool_result_data, pretty_format
                                )
                                escaped_json = MessageConverter._escape_content_for_xml(
                                    formatted_json, pretty_format
                                )
                                if pretty_format:
                                    formatted_text = f"<tool_result_sdk>\n{escaped_json}\n</tool_result_sdk>\n"
                                else:
                                    formatted_text = f"<tool_result_sdk>{escaped_json}</tool_result_sdk>"
                                response["content"].append(
                                    {"type": "text", "text": formatted_text}
                                )

        # Add ResultMessage to response content based on mode
        if mode != SystemMessageMode.IGNORE and "content" in response:
            result_content_block = (
                self.message_converter.create_result_message_content_block(
                    result_message, mode, "claude_code_sdk", pretty_format
                )
            )
            if result_content_block is not None:  # Handle IGNORE mode returning None
                response["content"].append(result_content_block)

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
        global_content_block_index = (
            0  # Track global content block index across all messages
        )

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
                    message=message,
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
                    assistant_message = message
                    assistant_messages.append(assistant_message)

                    # Iterate through content blocks and yield structured chunks for each
                    for block in assistant_message.content:
                        logger.debug("streaming_content_block", block=block)

                        # Handle Text Blocks
                        if isinstance(block, TextBlock) and getattr(
                            block, "text", None
                        ):
                            # Get system message mode for text formatting
                            mode = (
                                self.settings.claude.system_message_mode
                                if self.settings
                                else SystemMessageMode.FORWARD
                            )

                            # Format text content based on mode
                            text_block = block
                            text_content = text_block.text
                            if mode == SystemMessageMode.FORMATTED:
                                # Get formatting settings
                                pretty_format = (
                                    self.settings.claude.pretty_format
                                    if self.settings
                                    else True
                                )
                                escaped_text = MessageConverter._escape_content_for_xml(
                                    text_block.text, pretty_format
                                )
                                if pretty_format:
                                    text_content = f"<text>\n{escaped_text}\n</text>\n"
                                else:
                                    text_content = f"<text>{escaped_text}</text>"

                            # Start a new text block
                            yield {
                                "event": "content_block_start",
                                "data": {
                                    "type": "content_block_start",
                                    "index": global_content_block_index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            }
                            # Send the text content in a delta
                            yield {
                                "event": "content_block_delta",
                                "data": {
                                    "type": "content_block_delta",
                                    "index": global_content_block_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": text_content,
                                    },
                                },
                            }
                            # Stop the text block
                            yield {
                                "event": "content_block_stop",
                                "data": {
                                    "type": "content_block_stop",
                                    "index": global_content_block_index,
                                },
                            }
                            global_content_block_index += 1

                        # --- Handle Tool Use Blocks ---
                        elif isinstance(block, ToolUseBlock):
                            mode = (
                                self.settings.claude.system_message_mode
                                if self.settings
                                else SystemMessageMode.FORWARD
                            )

                            if mode == SystemMessageMode.FORWARD:
                                # Handle ToolUseBlock
                                tool_use_block = block
                                tool_input = getattr(tool_use_block, "input", {}) or {}
                                # Start a new tool_use_sdk block with all its data
                                yield {
                                    "event": "content_block_start",
                                    "data": {
                                        "type": "content_block_start",
                                        "index": global_content_block_index,
                                        "content_block": {
                                            "type": "tool_use_sdk",
                                            "id": getattr(
                                                tool_use_block,
                                                "id",
                                                f"tool_{id(tool_use_block)}",
                                            ),
                                            "name": tool_use_block.name,
                                            "input": tool_input,
                                            "source": "claude_code_sdk",
                                        },
                                    },
                                }
                                # 2. Immediately stop the block (since we get it all at once)
                                yield {
                                    "event": "content_block_stop",
                                    "data": {
                                        "type": "content_block_stop",
                                        "index": global_content_block_index,
                                    },
                                }
                                global_content_block_index += 1
                            elif mode == SystemMessageMode.FORMATTED:
                                # Get formatting settings
                                pretty_format = (
                                    self.settings.claude.pretty_format
                                    if self.settings
                                    else True
                                )

                                # Handle ToolUseBlock for FORMATTED mode
                                tool_use_block = block
                                tool_data = {
                                    "id": getattr(
                                        tool_use_block,
                                        "id",
                                        f"tool_{id(tool_use_block)}",
                                    ),
                                    "name": tool_use_block.name,
                                    "input": getattr(tool_use_block, "input", {}) or {},
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
                                    formatted_text = (
                                        f"<tool_use_sdk>{escaped_json}</tool_use_sdk>"
                                    )
                                # Send as text delta
                                yield {
                                    "event": "content_block_start",
                                    "data": {
                                        "type": "content_block_start",
                                        "index": global_content_block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                }
                                yield {
                                    "event": "content_block_delta",
                                    "data": {
                                        "type": "content_block_delta",
                                        "index": global_content_block_index,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": formatted_text,
                                        },
                                    },
                                }
                                yield {
                                    "event": "content_block_stop",
                                    "data": {
                                        "type": "content_block_stop",
                                        "index": global_content_block_index,
                                    },
                                }
                                global_content_block_index += 1
                            # mode == SystemMessageMode.IGNORE: skip entirely

                        # --- Handle Tool Result Blocks ---
                        elif isinstance(block, ToolResultBlock):
                            mode = (
                                self.settings.claude.system_message_mode
                                if self.settings
                                else SystemMessageMode.FORWARD
                            )

                            if mode == SystemMessageMode.FORWARD:
                                # Handle ToolResultBlock
                                tool_result_block = block
                                is_error = getattr(tool_result_block, "is_error", None)
                                # 1. Start a new tool_result_sdk block
                                yield {
                                    "event": "content_block_start",
                                    "data": {
                                        "type": "content_block_start",
                                        "index": global_content_block_index,
                                        "content_block": {
                                            "type": "tool_result_sdk",
                                            "tool_use_id": tool_result_block.tool_use_id,
                                            "content": tool_result_block.content,
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
                                    "data": {
                                        "type": "content_block_stop",
                                        "index": global_content_block_index,
                                    },
                                }
                                global_content_block_index += 1
                            elif mode == SystemMessageMode.FORMATTED:
                                # Get formatting settings
                                pretty_format = (
                                    self.settings.claude.pretty_format
                                    if self.settings
                                    else True
                                )

                                # Handle ToolResultBlock for FORMATTED mode
                                tool_result_block = block
                                tool_result_data = {
                                    "tool_use_id": tool_result_block.tool_use_id,
                                    "content": tool_result_block.content,
                                    "is_error": getattr(
                                        tool_result_block, "is_error", False
                                    ),
                                }
                                formatted_json = MessageConverter._format_json_data(
                                    tool_result_data, pretty_format
                                )
                                escaped_json = MessageConverter._escape_content_for_xml(
                                    formatted_json, pretty_format
                                )
                                if pretty_format:
                                    formatted_text = f"<tool_result_sdk>\n{escaped_json}\n</tool_result_sdk>\n"
                                else:
                                    formatted_text = f"<tool_result_sdk>{escaped_json}</tool_result_sdk>"
                                # Send as text delta
                                yield {
                                    "event": "content_block_start",
                                    "data": {
                                        "type": "content_block_start",
                                        "index": global_content_block_index,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                }
                                yield {
                                    "event": "content_block_delta",
                                    "data": {
                                        "type": "content_block_delta",
                                        "index": global_content_block_index,
                                        "delta": {
                                            "type": "text_delta",
                                            "text": formatted_text,
                                        },
                                    },
                                }
                                yield {
                                    "event": "content_block_stop",
                                    "data": {
                                        "type": "content_block_stop",
                                        "index": global_content_block_index,
                                    },
                                }
                                global_content_block_index += 1
                            # mode == SystemMessageMode.IGNORE: skip entirely
                        else:
                            logger.warning(
                                "streaming_content_block_unsupported_block_type",
                                block=block,
                            )

                elif isinstance(message, SystemMessage):
                    # Handle SystemMessage based on mode
                    mode = (
                        self.settings.claude.system_message_mode
                        if self.settings
                        else SystemMessageMode.FORWARD
                    )

                    if mode == SystemMessageMode.IGNORE:
                        logger.debug(
                            "Ignoring SystemMessage in streaming response (mode: ignore)."
                        )
                    else:
                        # Process using the converter
                        system_message = message
                        # Extract text from system message data
                        system_text = system_message.data.get(
                            "text", str(system_message.data)
                        )

                        # Get formatting settings
                        pretty_format = (
                            self.settings.claude.pretty_format
                            if self.settings
                            else True
                        )

                        # Use the message converter to create chunks based on mode
                        system_chunks = (
                            self.message_converter.create_system_message_chunks(
                                system_text,
                                mode,
                                index=global_content_block_index,
                                source="claude_code_sdk",
                                pretty_format=pretty_format,
                            )
                        )

                        # Yield the system message chunks
                        for event_type, chunk_data in system_chunks:
                            yield {"event": event_type, "data": chunk_data}
                        global_content_block_index += 1
                    continue

                elif isinstance(message, UserMessage):
                    # Handle UserMessage (which contains ToolResultBlocks)
                    mode = (
                        self.settings.claude.system_message_mode
                        if self.settings
                        else SystemMessageMode.FORWARD
                    )

                    if mode == SystemMessageMode.IGNORE:
                        logger.debug(
                            "Ignoring UserMessage in streaming response (mode: ignore)."
                        )
                    else:
                        # UserMessage content should be checked differently
                        # UserMessage may have tool results in a different structure
                        if (
                            hasattr(message, "content")
                            and hasattr(message.content, "__iter__")
                            and not isinstance(message.content, str)
                        ):
                            # Process content blocks within UserMessage if they exist
                            for block in message.content:  # type: ignore[unreachable]
                                if block.get("type") == "tool_result":
                                    tool_result_block = ToolResultBlock(
                                        tool_use_id=block.get("tool_use_id", ""),
                                        content=block.get("content"),
                                        is_error=block.get("is_error"),
                                    )
                                    # if isinstance(block, ToolResultBlock):
                                    if mode == SystemMessageMode.FORWARD:
                                        # Handle ToolResultBlock in FORWARD mode
                                        is_error = getattr(
                                            tool_result_block, "is_error", None
                                        )
                                        # 1. Start a new tool_result_sdk tool_result_block
                                        yield {
                                            "event": "content_block_start",
                                            "data": {
                                                "type": "content_block_start",
                                                "index": global_content_block_index,
                                                "content_block": {
                                                    "type": "tool_result_sdk",
                                                    "tool_use_id": tool_result_block.tool_use_id,
                                                    "content": tool_result_block.content,
                                                    "is_error": is_error
                                                    if is_error is not None
                                                    else False,
                                                    "source": "claude_code_sdk",
                                                },
                                            },
                                        }
                                        # 2. Immediately stop the tool_result_block
                                        yield {
                                            "event": "content_block_stop",
                                            "data": {
                                                "type": "content_block_stop",
                                                "index": global_content_block_index,
                                            },
                                        }
                                        global_content_block_index += 1
                                    elif mode == SystemMessageMode.FORMATTED:
                                        # Get formatting settings
                                        pretty_format = (
                                            self.settings.claude.pretty_format
                                            if self.settings
                                            else True
                                        )

                                        # Handle ToolResultBlock for FORMATTED mode
                                        tool_result_data = {
                                            "tool_use_id": tool_result_block.tool_use_id,
                                            "content": tool_result_block.content,
                                            "is_error": getattr(
                                                tool_result_block, "is_error", False
                                            ),
                                        }
                                        formatted_json = (
                                            MessageConverter._format_json_data(
                                                tool_result_data, pretty_format
                                            )
                                        )
                                        escaped_json = (
                                            MessageConverter._escape_content_for_xml(
                                                formatted_json, pretty_format
                                            )
                                        )
                                        if pretty_format:
                                            formatted_text = f"<tool_result_sdk>\n{escaped_json}\n</tool_result_sdk>\n"
                                        else:
                                            formatted_text = f"<tool_result_sdk>{escaped_json}</tool_result_sdk>"
                                        # Send as text delta
                                        yield {
                                            "event": "content_block_start",
                                            "data": {
                                                "type": "content_block_start",
                                                "index": global_content_block_index,
                                                "content_block": {
                                                    "type": "text",
                                                    "text": "",
                                                },
                                            },
                                        }
                                        yield {
                                            "event": "content_block_delta",
                                            "data": {
                                                "type": "content_block_delta",
                                                "index": global_content_block_index,
                                                "delta": {
                                                    "type": "text_delta",
                                                    "text": formatted_text,
                                                },
                                            },
                                        }
                                        yield {
                                            "event": "content_block_stop",
                                            "data": {
                                                "type": "content_block_stop",
                                                "index": global_content_block_index,
                                            },
                                        }
                                        global_content_block_index += 1
                                    # mode == SystemMessageMode.IGNORE: skip entirely
                    continue

                elif isinstance(message, ResultMessage):
                    # Process ResultMessage
                    result_message = message

                    # Get Claude API call timing
                    claude_api_call_ms = self.sdk_client.get_last_api_call_time_ms()

                    # Handle ResultMessage based on mode (before processing usage)
                    mode = (
                        self.settings.claude.system_message_mode
                        if self.settings
                        else SystemMessageMode.FORWARD
                    )

                    if mode != SystemMessageMode.IGNORE:
                        # Get formatting settings
                        pretty_format = (
                            self.settings.claude.pretty_format
                            if self.settings
                            else True
                        )

                        # Use the message converter to create chunks based on mode
                        result_chunks = (
                            self.message_converter.create_result_message_chunks(
                                result_message,
                                mode,
                                index=global_content_block_index,
                                source="claude_code_sdk",
                                pretty_format=pretty_format,
                            )
                        )

                        # Yield the result message chunks
                        for event_type, chunk_data in result_chunks:
                            yield {"event": event_type, "data": chunk_data}
                        global_content_block_index += 1

                    # Extract cost and tokens from result message using direct access
                    cost_usd = result_message.total_cost_usd
                    if result_message.usage:
                        tokens_input = result_message.usage.get("input_tokens")
                        tokens_output = result_message.usage.get("output_tokens")
                        cache_read_tokens = result_message.usage.get(
                            "cache_read_input_tokens"
                        )
                        cache_write_tokens = result_message.usage.get(
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
                        stop_reason=getattr(result_message, "stop_reason", "end_turn")
                    )

                    # Add usage information to message_delta chunk
                    for event_type, chunk_data in final_chunks:
                        if chunk_data.get("type") == "message_delta":
                            # usage_info = {}
                            usage_info = result_message.usage

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
