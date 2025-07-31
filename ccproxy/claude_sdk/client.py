"""Claude SDK client wrapper for handling core Claude Code SDK interactions."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, TypeVar, cast

import structlog
from pydantic import BaseModel

from ccproxy.claude_sdk.exceptions import ClaudeSDKError, StreamTimeoutError
from ccproxy.claude_sdk.manager import PoolManager
from ccproxy.claude_sdk.pool import PoolConfig
from ccproxy.config.settings import Settings
from ccproxy.core.async_utils import patched_typing
from ccproxy.core.errors import ClaudeProxyError, ServiceUnavailableError
from ccproxy.models import claude_sdk as sdk_models
from ccproxy.models.claude_sdk import SDKMessage
from ccproxy.observability import timed_operation


with patched_typing():
    from claude_code_sdk import (
        AssistantMessage as SDKAssistantMessage,
    )
    from claude_code_sdk import (
        ClaudeCodeOptions,
        CLIConnectionError,
        CLIJSONDecodeError,
        CLINotFoundError,
        ProcessError,
    )
    from claude_code_sdk import (
        ClaudeSDKClient as ImportedClaudeSDKClient,
    )
    from claude_code_sdk import (
        ResultMessage as SDKResultMessage,
    )
    from claude_code_sdk import (
        SystemMessage as SDKSystemMessage,
    )
    from claude_code_sdk import (
        UserMessage as SDKUserMessage,
    )


logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ClaudeSDKClient:
    """
    Minimal Claude SDK client wrapper that handles core SDK interactions.

    This class provides a clean interface to the Claude Code SDK while handling
    error translation and basic query execution. Supports both stateless query()
    calls and pooled connection reuse for improved performance.
    """

    # Class constants
    FIRST_CHUNK_TIMEOUT = 4.0  # Standard timeout for all streaming methods
    MESSAGE_TYPE_MAP: dict[type[Any], type[BaseModel]] = {
        SDKUserMessage: sdk_models.UserMessage,
        SDKAssistantMessage: sdk_models.AssistantMessage,
        SDKSystemMessage: sdk_models.SystemMessage,
        SDKResultMessage: sdk_models.ResultMessage,
    }

    def __init__(
        self,
        use_pool: bool = False,
        settings: Settings | None = None,
        pool_manager: PoolManager | None = None,
    ) -> None:
        """Initialize the Claude SDK client.

        Args:
            use_pool: Whether to use connection pooling for better performance
            settings: Application settings for pool configuration
            pool_manager: Optional PoolManager instance for dependency injection
        """
        self._last_api_call_time_ms: float = 0.0
        self._use_pool = use_pool
        self._settings = settings
        self._pool_manager = pool_manager

    @contextlib.asynccontextmanager
    async def _handle_sdk_exceptions(
        self, operation: str, request_id: str | None = None
    ) -> AsyncIterator[None]:
        """Context manager for common SDK error handling."""
        try:
            yield
        except (CLINotFoundError, CLIConnectionError) as e:
            logger.error(
                "claude_sdk_connection_failed",
                error=str(e),
                error_type=type(e).__name__,
                operation=operation,
                request_id=request_id,
            )
            raise ServiceUnavailableError(f"Claude CLI not available: {str(e)}") from e
        except (ProcessError, CLIJSONDecodeError) as e:
            logger.error(
                "claude_sdk_process_failed",
                error=str(e),
                error_type=type(e).__name__,
                operation=operation,
                request_id=request_id,
            )
            raise ClaudeProxyError(
                message=f"Claude process error: {str(e)}",
                error_type="service_unavailable_error",
                status_code=503,
            ) from e
        except StreamTimeoutError:
            # Re-raise StreamTimeoutError for service layer to handle
            raise
        except Exception as e:
            logger.error(
                "claude_sdk_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                operation=operation,
                request_id=request_id,
            )
            raise ClaudeProxyError(
                message=f"Unexpected error: {str(e)}",
                error_type="internal_server_error",
                status_code=500,
            ) from e

    async def _create_message_iterator(
        self, message: SDKMessage
    ) -> AsyncIterator[dict[str, Any]]:
        """Convert SDKMessage to iterator format expected by Claude SDK."""
        message_dict = message.model_dump()
        logger.debug("sending_sdk_message", message=message_dict)
        yield message_dict

    async def _get_configured_pool(self) -> Any:
        """Get pool with configuration from settings."""
        if not self._pool_manager:
            raise ClaudeProxyError("No pool manager available")

        pool_config = None
        if (
            self._settings
            and hasattr(self._settings, "claude")
            and self._settings.claude.use_client_pool
        ):
            pool_settings = self._settings.claude.pool_settings
            pool_config = PoolConfig(
                pool_size=pool_settings.pool_size,
                max_pool_size=pool_settings.max_pool_size,
                connection_timeout=pool_settings.connection_timeout,
                idle_timeout=pool_settings.idle_timeout,
                health_check_interval=pool_settings.health_check_interval,
                enable_health_checks=pool_settings.enable_health_checks,
            )

        return await self._pool_manager.get_pool(config=pool_config)

    async def _execute_with_client(
        self,
        client: ImportedClaudeSDKClient,  # Claude SDK client (ImportedClaudeSDKClient)
        message: SDKMessage,
        session_id: str | None,
        request_id: str | None,
        session_client: Any = None,  # SessionClient for session pool
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """Execute query with standard 4-second first chunk timeout."""
        # Send message
        message_iter = self._create_message_iterator(message)
        if session_id:
            await client.query(message_iter, session_id=session_id)
        else:
            await client.query(message_iter)

        # Get response with 4s timeout on first chunk
        response_iterator = client.receive_response()
        first_message, remaining_iterator = await self._wait_for_first_chunk(
            response_iterator,
            self.FIRST_CHUNK_TIMEOUT,  # 4 seconds for all methods
            session_id,
            request_id,
        )

        # Chain first message with remaining
        async def message_chain() -> AsyncIterator[Any]:
            yield first_message
            async for msg in remaining_iterator:
                yield msg

        # Process messages
        async for converted_message in self._process_message_stream(
            message_chain(), request_id, session_id, session_client
        ):
            yield converted_message

    def _convert_anthropic_messages_to_sdk(
        self, messages: list[dict[str, Any]]
    ) -> list[sdk_models.UserMessage]:
        """Convert Anthropic API messages to Claude SDK UserMessage format.

        Args:
            messages: List of Anthropic API messages

        Returns:
            List of Claude SDK UserMessage objects
        """
        sdk_messages = []

        for msg in messages:
            if msg.get("role") == "user":
                # Convert content to SDK format
                content_blocks: list[sdk_models.ContentBlock] = []

                if isinstance(msg.get("content"), str):
                    # Simple text content
                    content_blocks.append(
                        sdk_models.TextBlock(type="text", text=msg["content"])
                    )
                elif isinstance(msg.get("content"), list):
                    # List of content blocks
                    for block in msg["content"]:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                content_blocks.append(
                                    sdk_models.TextBlock(
                                        type="text", text=block.get("text", "")
                                    )
                                )
                            elif block.get("type") == "tool_result":
                                content_blocks.append(
                                    sdk_models.ToolResultBlock(
                                        type="tool_result",
                                        tool_use_id=block.get("tool_use_id", ""),
                                        content=block.get("content"),
                                        is_error=block.get("is_error", False),
                                    )
                                )
                            # Add other block types as needed

                if content_blocks:
                    sdk_messages.append(sdk_models.UserMessage(content=content_blocks))

        return sdk_messages

    def _should_use_session_pool(self, session_id: str | None) -> bool:
        """Determine if session pool should be used for this request."""
        if not session_id or not self._pool_manager:
            return False

        # Check settings using optional chaining pattern
        try:
            return bool(
                self._settings
                and self._settings.claude
                and self._settings.claude.session_pool
                and self._settings.claude.session_pool.enabled
            )
        except AttributeError:
            return False

    async def query_completion(
        self,
        message: SDKMessage,
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """
        Execute a query using the Claude Code SDK and yields strongly-typed Pydantic models.

        Args:
            message: SDKMessage to send to Claude SDK
            options: Claude Code options configuration
            request_id: Optional request ID for correlation
            session_id: Optional session ID for conversation continuity

        Yields:
            Strongly-typed Pydantic messages from ccproxy.claude_sdk.models

        Raises:
            ClaudeSDKError: If the query fails
        """
        # Determine routing strategy
        if self._should_use_session_pool(session_id):
            query_method = self._query_with_session_pool
        elif self._use_pool:
            query_method = self._query_with_pool
        else:
            query_method = self._query

        # Execute query with selected method
        async for msg in query_method(message, options, request_id, session_id):
            yield msg

    async def _query(
        self,
        message: SDKMessage,
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """Execute query using direct connection (no pool)."""
        async with (
            timed_operation("claude_sdk_query_direct", request_id) as op,
            self._handle_sdk_exceptions("direct_query", request_id),
        ):
            client = ImportedClaudeSDKClient(options)
            try:
                await client.connect()

                message_count = 0
                async for msg in self._execute_with_client(
                    client, message, session_id, request_id
                ):
                    message_count += 1
                    yield msg

                op["message_count"] = message_count
                self._last_api_call_time_ms = op.get("duration_ms", 0.0)

            finally:
                # Critical: Always disconnect non-session clients to prevent reuse
                try:
                    await client.disconnect()
                except Exception as e:
                    logger.warning(
                        "claude_sdk_disconnect_failed",
                        error=str(e),
                        request_id=request_id,
                    )

    async def _query_with_pool(
        self,
        message: SDKMessage,
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        if not session_id:
            session_id = "default"  # str(uuid4())

        """Execute query using pooled connection."""
        async with timed_operation("claude_sdk_query_pooled", request_id) as op:
            try:
                pool = await self._get_configured_pool()

                async with pool.acquire_client(options) as client:
                    message_count = 0
                    async for msg in self._execute_with_client(
                        client, message, session_id, request_id
                    ):
                        message_count += 1
                        yield msg

                    op["message_count"] = message_count
                    self._last_api_call_time_ms = op.get("duration_ms", 0.0)

            except StreamTimeoutError:
                # Re-raise timeout errors
                raise
            except Exception as e:
                logger.error(
                    "claude_sdk_pooled_query_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Fall back to direct connection on pool errors
                logger.info("claude_sdk_fallback_to_direct")
                async for msg in self._query(message, options, request_id, session_id):
                    yield msg

    async def _query_with_session_pool(
        self,
        message: SDKMessage,
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """Execute query using session-aware pooled connection."""
        async with timed_operation("claude_sdk_query_session_pool", request_id) as op:
            try:
                if not session_id:
                    raise ClaudeSDKError("Session ID required for session pool")

                if not self._pool_manager:
                    raise ClaudeSDKError("No pool manager available")

                # Enable continue conversation for session pool
                # so conversation is possible to resume based on session_id
                options.continue_conversation = True

                session_client = await self._pool_manager.get_session_client(
                    session_id, options
                )

                async with session_client.lock:  # Prevent concurrent access
                    session_client.update_usage()

                    # Ensure client is connected
                    if not session_client.claude_client:
                        logger.error(
                            "session_client_not_connected",
                            session_id=session_id,
                            status=session_client.status,
                        )
                        raise ClaudeSDKError(
                            f"Session client not connected for session {session_id}"
                        )

                    message_count = 0
                    async for msg in self._execute_with_client(
                        session_client.claude_client,
                        message,
                        session_id,
                        request_id,
                        session_client=session_client,
                    ):
                        message_count += 1
                        yield msg

                    op["message_count"] = message_count
                    op["session_id"] = session_id
                    self._last_api_call_time_ms = op.get("duration_ms", 0.0)

            except StreamTimeoutError:
                raise  # Let service layer handle
            except Exception as e:
                logger.error(
                    "claude_sdk_session_pool_query_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    session_id=session_id,
                    exc_info=True,
                )
                # Fall back to regular pool
                logger.info(
                    "claude_sdk_fallback_to_regular_pool", session_id=session_id
                )
                async for msg in self._query_with_pool(
                    message, options, request_id, session_id
                ):
                    yield msg

    async def _wait_for_first_chunk(
        self,
        message_iterator: AsyncIterator[Any],
        timeout_seconds: float = 5.0,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> tuple[Any, AsyncIterator[Any]]:
        """
        Wait for the first chunk from an async iterator with timeout.

        Args:
            message_iterator: The async iterator to get messages from
            timeout_seconds: Timeout in seconds (default 5.0)
            session_id: Optional session ID for logging
            request_id: Optional request ID for logging

        Returns:
            Tuple of (first_message, remaining_iterator)

        Raises:
            StreamTimeoutError: If no chunk is received within timeout
        """
        try:
            # Wait for the first chunk with timeout - don't care about message type
            first_message = await asyncio.wait_for(
                anext(message_iterator), timeout=timeout_seconds
            )
            return first_message, message_iterator
        except TimeoutError:
            logger.error(
                "first_chunk_timeout",
                session_id=session_id,
                request_id=request_id,
                timeout=timeout_seconds,
                message="No chunk received within timeout, interrupting session",
            )
            # Interrupt the session if we have a session_id and pool manager
            if session_id and self._pool_manager:
                try:
                    await self._pool_manager.interrupt_session(session_id)
                except Exception as e:
                    logger.error(
                        "failed_to_interrupt_stuck_session",
                        session_id=session_id,
                        error=str(e),
                    )

            # Raise a custom exception with error details
            raise StreamTimeoutError(
                message=f"Stream timeout: No response received within {timeout_seconds} seconds. The command may not be supported or the session may be stuck.",
                session_id=session_id or "unknown",
                timeout_seconds=timeout_seconds,
            ) from None

    async def _process_message_stream(
        self,
        message_iterator: AsyncIterator[Any],
        request_id: str | None = None,
        session_id: str | None = None,
        session_client: Any = None,  # SessionClient for session pool
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """
        Process messages from an async iterator, converting them to Pydantic models.

        Args:
            message_iterator: The async iterator of SDK messages
            request_id: Optional request ID for logging
            session_id: Optional session ID for logging
            session_client: Optional session context for session pool operations

        Yields:
            Converted Pydantic model messages
        """
        async for sdk_msg in message_iterator:
            # Find matching type and convert
            for sdk_type, model_type in self.MESSAGE_TYPE_MAP.items():
                if isinstance(sdk_msg, sdk_type):
                    try:
                        converted_message = cast(
                            sdk_models.UserMessage
                            | sdk_models.AssistantMessage
                            | sdk_models.SystemMessage
                            | sdk_models.ResultMessage,
                            self._convert_message(sdk_msg, model_type),
                        )

                        # Special handling for ResultMessage
                        if session_client and isinstance(
                            converted_message, sdk_models.ResultMessage
                        ):
                            session_client.sdk_session_id = converted_message.session_id

                        yield converted_message
                    except Exception as e:
                        logger.warning(
                            "claude_sdk_message_conversion_failed",
                            message_type=type(sdk_msg).__name__,
                            error=str(e),
                            request_id=request_id,
                            session_id=session_id,
                        )
                    break
            else:
                # No matching type found
                logger.warning(
                    "claude_sdk_unknown_message_type",
                    message_type=type(sdk_msg).__name__,
                    request_id=request_id,
                    session_id=session_id,
                )

    def _convert_message(self, message: Any, model_class: type[T]) -> T:
        """Convert SDK message to Pydantic model."""
        if hasattr(message, "__dict__"):
            return model_class.model_validate(vars(message))
        else:
            # For dataclass objects, use dataclass.asdict equivalent
            message_dict = {}
            if hasattr(message, "__dataclass_fields__"):
                message_dict = {
                    field: getattr(message, field)
                    for field in message.__dataclass_fields__
                }
            else:
                # Try to extract common attributes
                for attr in [
                    "content",
                    "subtype",
                    "data",
                    "session_id",
                    "stop_reason",
                    "usage",
                    "total_cost_usd",
                ]:
                    if hasattr(message, attr):
                        message_dict[attr] = getattr(message, attr)

            return model_class.model_validate(message_dict)

    def get_last_api_call_time_ms(self) -> float:
        """
        Get the duration of the last Claude API call in milliseconds.

        Returns:
            Duration in milliseconds, or 0.0 if no call has been made yet
        """
        return self._last_api_call_time_ms

    async def validate_health(self) -> bool:
        """
        Validate that the Claude SDK is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            logger.debug("health_check_start", component="claude_sdk")

            # Simple health check - the SDK is available if we can import it
            # More sophisticated checks could be added here
            is_healthy = True

            logger.debug(
                "health_check_completed", component="claude_sdk", healthy=is_healthy
            )
            return is_healthy
        except Exception as e:
            logger.error(
                "health_check_failed",
                component="claude_sdk",
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def interrupt_session(self, session_id: str) -> bool:
        """Interrupt a specific session due to client disconnection.

        Args:
            session_id: The session ID to interrupt

        Returns:
            True if session was found and interrupted, False otherwise
        """
        logger.debug("sdk_client_interrupt_session_started", session_id=session_id)
        if self._pool_manager:
            logger.info(
                "client_interrupt_session_requested",
                session_id=session_id,
                has_pool_manager=True,
            )
            return await self._pool_manager.interrupt_session(session_id)
        else:
            logger.warning(
                "client_interrupt_session_no_pool_manager",
                session_id=session_id,
            )
            return False

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        # Claude Code SDK doesn't require explicit cleanup
        pass

    async def __aenter__(self) -> "ClaudeSDKClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
