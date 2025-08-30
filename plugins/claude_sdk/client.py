"""Claude SDK client wrapper for handling core Claude Code SDK interactions."""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from ccproxy.core.async_utils import patched_typing
from ccproxy.core.errors import ClaudeProxyError, ServiceUnavailableError
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.request_context import timed_operation

from . import models as sdk_models
from .config import ClaudeSDKSettings, SessionPoolSettings
from .exceptions import ClaudeSDKError, StreamTimeoutError
from .manager import SessionManager
from .models import SDKMessage
from .stream_handle import StreamHandle


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


logger = get_plugin_logger()

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
        config: ClaudeSDKSettings,
        session_manager: SessionManager | None = None,
    ) -> None:
        """Initialize the Claude SDK client.

        Args:
            config: Plugin-specific configuration for Claude SDK
            session_manager: Optional SessionManager instance for dependency injection
        """
        self._last_api_call_time_ms: float = 0.0
        self.config = config
        self._session_manager = session_manager

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
                exc_info=e,
            )
            raise ClaudeProxyError(
                message=f"Unexpected error: {str(e)}",
                error_type="internal_server_error",
                status_code=500,
            ) from e

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
        message_dict = message.model_dump()
        logger.debug("sending_sdk_message", message=message_dict)

        async def message_iter() -> AsyncIterator[dict[str, Any]]:
            yield message_dict

        if session_id:
            await client.query(message_iter(), session_id=session_id)
        else:
            await client.query(message_iter())

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
        if not session_id or not self._session_manager:
            return False

        # Check settings using safe attribute chaining
        if not self.config:
            return False

        pool_settings = getattr(self.config, "sdk_session_pool", None)
        if not pool_settings:
            return False

        return bool(getattr(pool_settings, "enabled", False))

    async def query_completion(
        self,
        message: SDKMessage,
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> StreamHandle:
        """
        Execute a query using the Claude Code SDK and return a StreamHandle.

        Args:
            message: SDKMessage to send to Claude SDK
            options: Claude Code options configuration
            request_id: Optional request ID for correlation
            session_id: Optional session ID for conversation continuity

        Returns:
            StreamHandle that can create listeners for the stream

        Raises:
            ClaudeSDKError: If the query fails
        """
        # Determine routing strategy
        if self._should_use_session_pool(session_id):
            return await self._create_session_pool_stream_handle(
                message, options, request_id, session_id
            )
        else:
            return await self._create_direct_stream_handle(
                message, options, request_id, session_id
            )

    async def _create_direct_stream_handle(
        self,
        message: SDKMessage,
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> StreamHandle:
        """Create stream handle for direct query (no session pool)."""
        message_iterator = self._query(message, options, request_id, session_id)

        # Convert core settings to plugin settings if available
        plugin_session_config = None
        if self.config and self.config.sdk_session_pool:
            core_pool_settings = self.config.sdk_session_pool
            plugin_session_config = SessionPoolSettings(
                enabled=core_pool_settings.enabled,
                session_ttl=core_pool_settings.session_ttl,
                max_sessions=core_pool_settings.max_sessions,
                cleanup_interval=getattr(core_pool_settings, "cleanup_interval", 300),
                idle_threshold=getattr(core_pool_settings, "idle_threshold", 300),
                connection_recovery=getattr(
                    core_pool_settings, "connection_recovery", True
                ),
                stream_first_chunk_timeout=getattr(
                    core_pool_settings, "stream_first_chunk_timeout", 8
                ),
                stream_ongoing_timeout=getattr(
                    core_pool_settings, "stream_ongoing_timeout", 60
                ),
            )

        return StreamHandle(
            message_iterator=message_iterator,
            session_id=session_id,
            request_id=request_id,
            session_client=None,
            session_config=plugin_session_config,  # StreamHandle will use defaults if None
        )

    async def _create_session_pool_stream_handle(
        self,
        message: SDKMessage,
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> StreamHandle:
        """Create stream handle for session pool query."""
        if not session_id:
            raise ClaudeSDKError("Session ID required for session pool")
        if not self._session_manager:
            raise ClaudeSDKError("No session manager available")

        # Enable continue conversation for session pool
        options.continue_conversation = True
        session_client = await self._session_manager.get_session_client(
            session_id, options
        )

        message_iterator = self._query_with_session_pool(
            message, options, request_id, session_id
        )

        # Get session config from session manager
        session_config = None
        if (
            self._session_manager
            and hasattr(self._session_manager, "_session_pool")
            and self._session_manager._session_pool
        ):
            session_config = self._session_manager._session_pool.config

        stream_handle = StreamHandle(
            message_iterator=message_iterator,
            session_id=session_id,
            request_id=request_id,
            session_client=session_client,
            session_config=session_config,
        )

        # Set the active stream handle on the session client for proper cleanup
        session_client.active_stream_handle = stream_handle

        return stream_handle

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
                        exc_info=e,
                    )

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

                if not self._session_manager:
                    raise ClaudeSDKError("No session manager available")

                # Enable continue conversation for session pool
                # so conversation is possible to resume based on session_id
                options.continue_conversation = True

                session_client = await self._session_manager.get_session_client(
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

                    # Mark session as having active stream
                    session_client.has_active_stream = True

                    # Create wrapped stream generator
                    async def stream_with_cleanup() -> AsyncIterator[
                        sdk_models.UserMessage
                        | sdk_models.AssistantMessage
                        | sdk_models.SystemMessage
                        | sdk_models.ResultMessage
                    ]:
                        stream_iterator = None
                        try:
                            message_count = 0
                            if not session_client.claude_client:
                                raise ClaudeSDKError("Session client not connected")

                            stream_iterator = self._execute_with_client(
                                session_client.claude_client,
                                message,
                                session_id,
                                request_id,
                                session_client=session_client,
                            )

                            async for msg in stream_iterator:
                                message_count += 1
                                yield msg

                            op["message_count"] = message_count
                            op["session_id"] = session_id
                            self._last_api_call_time_ms = op.get("duration_ms", 0.0)

                        except GeneratorExit:
                            # Client disconnected - mark session for drain
                            logger.warning(
                                "claude_sdk_session_stream_interrupted",
                                session_id=session_id,
                                request_id=request_id,
                                message="Client disconnected, session will drain stream on next interrupt",
                            )

                            # Just mark that stream needs draining
                            # The SessionClient.interrupt() will handle the actual draining
                            session_client.has_active_stream = True
                            raise
                        finally:
                            # Clean up if stream completed normally
                            if not session_client.has_active_stream:
                                session_client.has_active_stream = False

                    # Yield from the wrapped generator
                    async for msg in stream_with_cleanup():
                        yield msg

            except StreamTimeoutError:
                raise  # Let service layer handle
            except Exception as e:
                logger.error(
                    "claude_sdk_session_pool_query_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    session_id=session_id,
                    exc_info=e,
                )
                # Fall back to direct query
                logger.debug(
                    "claude_sdk_fallback_to_direct_query", session_id=session_id
                )
                async for msg in self._query(message, options, request_id, session_id):
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
            logger.debug(
                "waiting_for_first_chunk", timeout=timeout_seconds, category="streaming"
            )
            first_message = await asyncio.wait_for(
                anext(message_iterator), timeout=timeout_seconds
            )
            return first_message, message_iterator
        except TimeoutError:
            # Check if session pool is enabled - if so, let it handle the timeout
            has_session_pool = (
                self._session_manager and await self._session_manager.has_session_pool()
            )

            if has_session_pool:
                logger.error(
                    "first_chunk_timeout",
                    session_id=session_id,
                    request_id=request_id,
                    timeout=timeout_seconds,
                    message="No chunk received within timeout, session pool will handle cleanup",
                )
            else:
                logger.error(
                    "first_chunk_timeout",
                    session_id=session_id,
                    request_id=request_id,
                    timeout=timeout_seconds,
                    message="No chunk received within timeout, interrupting session",
                )
                # Interrupt the session if we have a session_id and session manager (no session pool)
                if session_id and self._session_manager:
                    try:
                        await self._session_manager.interrupt_session(session_id)
                    except Exception as e:
                        logger.error(
                            "failed_to_interrupt_stuck_session",
                            session_id=session_id,
                            error=str(e),
                            exc_info=e,
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
        drain_mode: bool = False,  # If True, consume but don't yield
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
            drain_mode: If True, consume messages without yielding (for cleanup)

        Yields:
            Converted Pydantic model messages (unless drain_mode is True)
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

                        # Only yield if not in drain mode
                        if not drain_mode:
                            yield converted_message
                        else:
                            logger.debug(
                                "claude_sdk_draining_message",
                                message_type=type(converted_message).__name__,
                                request_id=request_id,
                                session_id=session_id,
                            )
                    except Exception as e:
                        logger.warning(
                            "claude_sdk_message_conversion_failed",
                            message_type=type(sdk_msg).__name__,
                            error=str(e),
                            request_id=request_id,
                            session_id=session_id,
                            exc_info=e,
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

    async def _create_drain_task(
        self,
        message_iterator: AsyncIterator[Any],
        session_client: Any,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> asyncio.Task[None]:
        """Create a background task to drain remaining messages from stream.

        Args:
            message_iterator: The message iterator to drain
            session_client: Session client to update stream status
            request_id: Optional request ID for logging
            session_id: Optional session ID for logging

        Returns:
            Task that completes when stream is drained
        """

        async def drain_stream() -> None:
            try:
                logger.trace(
                    "claude_sdk_starting_stream_drain",
                    session_id=session_id,
                    request_id=request_id,
                )

                message_count = 0
                async for _ in self._process_message_stream(
                    message_iterator,
                    request_id=request_id,
                    session_id=session_id,
                    session_client=session_client,
                    drain_mode=True,
                ):
                    message_count += 1

                logger.trace(
                    "claude_sdk_stream_drained",
                    session_id=session_id,
                    request_id=request_id,
                    drained_messages=message_count,
                )
            except Exception as e:
                logger.error(
                    "claude_sdk_stream_drain_error",
                    session_id=session_id,
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=e,
                )
            finally:
                if session_client:
                    session_client.has_active_stream = False
                    session_client.active_stream_task = None

        return asyncio.create_task(drain_stream())

    def _convert_message(self, message: Any, model_class: type[T]) -> T:
        """Convert SDK message to Pydantic model."""
        # Try standard object attribute extraction first
        if hasattr(message, "__dict__"):
            return model_class.model_validate(vars(message))

        # Handle dataclass objects
        if hasattr(message, "__dataclass_fields__"):
            message_dict = {
                field: getattr(message, field) for field in message.__dataclass_fields__
            }
            return model_class.model_validate(message_dict)

        # Fallback: extract common attributes
        message_dict = {}
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
                exc_info=e,
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
        if self._session_manager:
            logger.debug(
                "client_interrupt_session_requested",
                session_id=session_id,
                has_session_manager=True,
            )
            return await self._session_manager.interrupt_session(session_id)
        else:
            logger.warning(
                "client_interrupt_session_no_session_manager",
                session_id=session_id,
            )
            return False

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        # Claude Code SDK doesn't require explicit cleanup
        pass
