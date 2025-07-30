"""Claude SDK client wrapper for handling core Claude Code SDK interactions."""

from collections.abc import AsyncIterator
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel

from ccproxy.claude_sdk.manager import PoolManager
from ccproxy.claude_sdk.pool import PoolConfig
from ccproxy.config.settings import Settings
from ccproxy.core.async_utils import patched_typing
from ccproxy.core.errors import ClaudeProxyError, ServiceUnavailableError
from ccproxy.models import claude_sdk as sdk_models
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
        query,
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


class ClaudeSDKError(Exception):
    """Base exception for Claude SDK errors."""


class ClaudeSDKConnectionError(ClaudeSDKError):
    """Raised when unable to connect to Claude Code."""


class ClaudeSDKProcessError(ClaudeSDKError):
    """Raised when Claude Code process fails."""


class ClaudeSDKClient:
    """
    Minimal Claude SDK client wrapper that handles core SDK interactions.

    This class provides a clean interface to the Claude Code SDK while handling
    error translation and basic query execution. Supports both stateless query()
    calls and pooled connection reuse for improved performance.
    """

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
        logger.debug(
            "session_pool_routing_check",
            session_id=session_id,
            has_session_id=bool(session_id),
            has_pool_manager=bool(self._pool_manager),
            has_settings=bool(self._settings),
        )

        # Must have session_id
        if not session_id:
            logger.debug(
                "session_pool_routing_decision",
                decision="no_session_id",
                use_session_pool=False,
            )
            return False

        # Must have pool manager
        if not self._pool_manager:
            logger.debug(
                "session_pool_routing_decision",
                decision="no_pool_manager",
                use_session_pool=False,
            )
            return False

        # Must have settings with session pool enabled
        if not self._settings or not hasattr(self._settings, "claude"):
            logger.debug(
                "session_pool_routing_decision",
                decision="no_settings",
                use_session_pool=False,
            )
            return False

        session_pool_settings = getattr(self._settings.claude, "session_pool", None)
        if not session_pool_settings:
            logger.debug(
                "session_pool_routing_decision",
                decision="no_session_pool_settings",
                use_session_pool=False,
            )
            return False

        enabled = getattr(session_pool_settings, "enabled", False)
        logger.debug(
            "session_pool_routing_decision",
            decision="session_pool_enabled_check",
            session_pool_enabled=enabled,
            use_session_pool=enabled,
        )
        return enabled

    async def query_completion(
        self,
        messages: list[dict[str, Any]],
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
            messages: List of Anthropic API messages to convert and send
            options: Claude Code options configuration
            request_id: Optional request ID for correlation
            session_id: Optional session ID for conversation continuity

        Yields:
            Strongly-typed Pydantic messages from ccproxy.claude_sdk.models

        Raises:
            ClaudeSDKError: If the query fails
        """
        logger.debug(
            "query_completion_start",
            request_id=request_id,
            session_id=session_id,
            messages_count=len(messages),
            use_pool=self._use_pool,
            pool_manager_available=bool(self._pool_manager),
        )

        # Route to session pool if session_id provided and session pool available
        if self._should_use_session_pool(session_id):
            logger.debug(
                "query_completion_routing",
                request_id=request_id,
                session_id=session_id,
                route="session_pool",
            )
            async for message in self._query_with_session_pool(
                messages, options, request_id, session_id
            ):
                yield message
        elif self._use_pool:
            logger.debug(
                "query_completion_routing",
                request_id=request_id,
                session_id=session_id,
                route="regular_pool",
            )
            async for message in self._query_with_pool(
                messages, options, request_id, session_id
            ):
                yield message
        else:
            logger.debug(
                "query_completion_routing",
                request_id=request_id,
                session_id=session_id,
                route="direct_connection",
            )
            async for message in self._query(messages, options, request_id, session_id):
                yield message

    async def _query(
        self,
        messages: list[dict[str, Any]],
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """Execute query using direct connect approach (same as pool but without pool)."""
        async with timed_operation("claude_sdk_query_stateless", request_id) as op:
            try:
                logger.debug(
                    "claude_sdk_query_start",
                    messages_count=len(messages),
                    mode="stateless_connect",
                    session_id=session_id,
                )

                # Use ClaudeSDKClient with direct connect (no pool)
                client = ImportedClaudeSDKClient(options)

                try:
                    # Connect to Claude CLI
                    await client.connect()

                    # Convert Anthropic messages to SDK format
                    sdk_messages = self._convert_anthropic_messages_to_sdk(messages)

                    # Convert SDK messages to Claude SDK expected format
                    async def message_iter() -> AsyncIterator[dict[str, Any]]:
                        for msg in sdk_messages[-1:]:
                            # Convert to Claude SDK expected format with proper content handling
                            content: str | list[dict[str, Any]] = ""
                            if msg.content:
                                if len(msg.content) == 1 and hasattr(
                                    msg.content[0], "text"
                                ):
                                    # Simple text message
                                    content = msg.content[0].text
                                else:
                                    # Complex message with multiple blocks - convert to list format
                                    content = [
                                        block.model_dump() for block in msg.content
                                    ]

                            message_dict = {
                                "type": "user",
                                "message": {"role": "user", "content": content},
                                "parent_tool_use_id": None,
                                "session_id": session_id,
                            }
                            logger.debug("sending_sdk_message", message=message_dict)
                            yield message_dict

                    # Send query with session_id support
                    if session_id:
                        await client.query(message_iter(), session_id=session_id)
                    else:
                        await client.query(message_iter())

                    message_count = 0
                    # Receive and process all messages (same as pool)
                    async for message in client.receive_response():
                        message_count += 1

                        logger.debug(
                            "claude_sdk_raw_message_received",
                            message_type=type(message).__name__,
                            message_count=message_count,
                            request_id=request_id,
                            has_content=hasattr(message, "content")
                            and bool(getattr(message, "content", None)),
                            content_preview=str(message)[:150],
                        )

                        # Skip unknown message types early
                        if not isinstance(
                            message,
                            SDKUserMessage
                            | SDKAssistantMessage
                            | SDKSystemMessage
                            | SDKResultMessage,
                        ):
                            logger.warning(  # type: ignore[unreachable]
                                "claude_sdk_unknown_message_type",
                                message_type=type(message).__name__,
                                request_id=request_id,
                            )
                            continue

                        # Convert SDK message to our Pydantic model (same logic as pool)
                        try:
                            converted_message: (
                                sdk_models.UserMessage
                                | sdk_models.AssistantMessage
                                | sdk_models.SystemMessage
                                | sdk_models.ResultMessage
                            )
                            if isinstance(message, SDKUserMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.UserMessage
                                )
                            elif isinstance(message, SDKAssistantMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.AssistantMessage
                                )
                            elif isinstance(message, SDKSystemMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.SystemMessage
                                )
                            else:  # SDKResultMessage
                                converted_message = self._convert_message(
                                    message, sdk_models.ResultMessage
                                )

                            logger.debug(
                                "claude_sdk_message_converted_successfully",
                                original_type=type(message).__name__,
                                converted_type=type(converted_message).__name__,
                                message_count=message_count,
                                request_id=request_id,
                            )
                            yield converted_message
                        except Exception as e:
                            logger.warning(
                                "claude_sdk_message_conversion_failed",
                                message_type=type(message).__name__,
                                error=str(e),
                            )
                            # Skip invalid messages rather than crashing
                            continue

                finally:
                    # Always disconnect the client
                    try:
                        await client.disconnect()
                    except Exception as e:
                        logger.warning(
                            "claude_sdk_disconnect_failed",
                            error=str(e),
                            request_id=request_id,
                        )

                # Store final metrics
                op["message_count"] = message_count
                self._last_api_call_time_ms = op.get("duration_ms", 0.0)

                logger.debug(
                    "claude_sdk_query_completed",
                    message_count=message_count,
                    duration_ms=op.get("duration_ms"),
                    mode="stateless_connect",
                )

            except (CLINotFoundError, CLIConnectionError) as e:
                logger.error(
                    "claude_sdk_connection_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ServiceUnavailableError(
                    f"Claude CLI not available: {str(e)}"
                ) from e
            except (ProcessError, CLIJSONDecodeError) as e:
                logger.error(
                    "claude_sdk_process_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ClaudeProxyError(
                    message=f"Claude process error: {str(e)}",
                    error_type="service_unavailable_error",
                    status_code=503,
                ) from e
            except Exception as e:
                logger.error(
                    "claude_sdk_unexpected_error_occurred",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ClaudeProxyError(
                    message=f"Unexpected error: {str(e)}",
                    error_type="internal_server_error",
                    status_code=500,
                ) from e

    async def _query_stateless(
        self,
        messages: list[dict[str, Any]],
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """Execute query using the basic query() function from Claude SDK."""
        async with timed_operation(
            "claude_sdk_query_stateless_basic", request_id
        ) as op:
            try:
                logger.debug(
                    "claude_sdk_query_start",
                    messages_count=len(messages),
                    mode="stateless_basic",
                    session_id=session_id,
                )

                # Convert Anthropic messages to a simple prompt string
                # Take the last user message as the prompt
                prompt = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            prompt = content
                        elif isinstance(content, list):
                            # Extract text from content blocks
                            text_parts = []
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    text_parts.append(block.get("text", ""))
                            prompt = "\n".join(text_parts)

                if not prompt:
                    prompt = "Hello"  # Default prompt if none found

                message_count = 0

                # Use the basic query function from Claude SDK
                async for message in query(prompt=prompt):
                    message_count += 1

                    logger.debug(
                        "claude_sdk_raw_message_received",
                        message_type=type(message).__name__,
                        message_count=message_count,
                        request_id=request_id,
                        has_content=hasattr(message, "content")
                        and bool(getattr(message, "content", None)),
                        content_preview=str(message)[:150],
                    )

                    # Skip unknown message types early
                    if not isinstance(
                        message,
                        SDKUserMessage
                        | SDKAssistantMessage
                        | SDKSystemMessage
                        | SDKResultMessage,
                    ):
                        logger.warning(  # type: ignore[unreachable]
                            "claude_sdk_unknown_message_type",
                            message_type=type(message).__name__,
                            request_id=request_id,
                        )
                        continue

                    # Convert SDK message to our Pydantic model
                    try:
                        converted_message: (
                            sdk_models.UserMessage
                            | sdk_models.AssistantMessage
                            | sdk_models.SystemMessage
                            | sdk_models.ResultMessage
                        )
                        if isinstance(message, SDKUserMessage):
                            converted_message = self._convert_message(
                                message, sdk_models.UserMessage
                            )
                        elif isinstance(message, SDKAssistantMessage):
                            converted_message = self._convert_message(
                                message, sdk_models.AssistantMessage
                            )
                        elif isinstance(message, SDKSystemMessage):
                            converted_message = self._convert_message(
                                message, sdk_models.SystemMessage
                            )
                        else:  # SDKResultMessage
                            converted_message = self._convert_message(
                                message, sdk_models.ResultMessage
                            )

                        logger.debug(
                            "claude_sdk_message_converted_successfully",
                            original_type=type(message).__name__,
                            converted_type=type(converted_message).__name__,
                            message_count=message_count,
                            request_id=request_id,
                        )
                        yield converted_message
                    except Exception as e:
                        logger.warning(
                            "claude_sdk_message_conversion_failed",
                            message_type=type(message).__name__,
                            error=str(e),
                        )
                        # Skip invalid messages rather than crashing
                        continue

                # Store final metrics
                op["message_count"] = message_count
                self._last_api_call_time_ms = op.get("duration_ms", 0.0)

                logger.debug(
                    "claude_sdk_query_completed",
                    message_count=message_count,
                    duration_ms=op.get("duration_ms"),
                    mode="stateless_basic",
                )

            except (CLINotFoundError, CLIConnectionError) as e:
                logger.error(
                    "claude_sdk_connection_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ServiceUnavailableError(
                    f"Claude CLI not available: {str(e)}"
                ) from e
            except (ProcessError, CLIJSONDecodeError) as e:
                logger.error(
                    "claude_sdk_process_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ClaudeProxyError(
                    message=f"Claude process error: {str(e)}",
                    error_type="service_unavailable_error",
                    status_code=503,
                ) from e
            except Exception as e:
                logger.error(
                    "claude_sdk_unexpected_error_occurred",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ClaudeProxyError(
                    message=f"Unexpected error: {str(e)}",
                    error_type="internal_server_error",
                    status_code=500,
                ) from e

    async def _query_with_pool(
        self,
        messages: list[dict[str, Any]],
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """Execute query using pooled connection approach."""

        async with timed_operation("claude_sdk_query_pooled", request_id) as op:
            try:
                logger.debug(
                    "claude_sdk_query_start",
                    messages_count=len(messages),
                    mode="pooled",
                    session_id=session_id,
                )

                # Create pool config from settings if available
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

                # Use injected manager
                if self._pool_manager is None:
                    raise ClaudeProxyError("No pool manager available")

                manager = self._pool_manager
                pool = await manager.get_pool(config=pool_config)

                message_count = 0

                async with pool.acquire_client(options) as client:
                    # Send the query to the pooled client with options
                    # Use session_id for conversation continuity
                    # Convert Anthropic messages to SDK format
                    sdk_messages = self._convert_anthropic_messages_to_sdk(messages)

                    # Convert SDK messages to Claude SDK expected format
                    async def message_iter() -> AsyncIterator[dict[str, Any]]:
                        for msg in sdk_messages:
                            # Convert to Claude SDK expected format with proper content handling
                            content: str | list[dict[str, Any]] = ""
                            if msg.content:
                                if len(msg.content) == 1 and hasattr(
                                    msg.content[0], "text"
                                ):
                                    # Simple text message
                                    content = msg.content[0].text
                                else:
                                    # Complex message with multiple blocks - convert to list format
                                    content = [
                                        block.model_dump() for block in msg.content
                                    ]

                            yield {
                                "type": "user",
                                "message": {"role": "user", "content": content},
                                "parent_tool_use_id": None,
                                "session_id": session_id,
                            }

                    if session_id:
                        await client.query(message_iter(), session_id=session_id)
                    else:
                        await client.query(message_iter())

                    # Receive and process all messages
                    async for message in client.receive_response():
                        message_count += 1

                        logger.debug(
                            "claude_sdk_raw_message_received",
                            message_type=type(message).__name__,
                            message_count=message_count,
                            request_id=request_id,
                            has_content=hasattr(message, "content")
                            and bool(getattr(message, "content", None)),
                            content_preview=str(message)[:150],
                        )

                        # Skip unknown message types early
                        if not isinstance(
                            message,
                            SDKUserMessage
                            | SDKAssistantMessage
                            | SDKSystemMessage
                            | SDKResultMessage,
                        ):
                            logger.warning(  # type: ignore[unreachable]
                                "claude_sdk_unknown_message_type",
                                message_type=type(message).__name__,
                                request_id=request_id,
                            )
                            continue

                        # Convert SDK message to our Pydantic model (same logic as stateless)
                        try:
                            converted_message: (
                                sdk_models.UserMessage
                                | sdk_models.AssistantMessage
                                | sdk_models.SystemMessage
                                | sdk_models.ResultMessage
                            )
                            if isinstance(message, SDKUserMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.UserMessage
                                )
                            elif isinstance(message, SDKAssistantMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.AssistantMessage
                                )
                            elif isinstance(message, SDKSystemMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.SystemMessage
                                )
                            else:  # SDKResultMessage
                                converted_message = self._convert_message(
                                    message, sdk_models.ResultMessage
                                )

                            logger.debug(
                                "claude_sdk_message_converted_successfully",
                                original_type=type(message).__name__,
                                converted_type=type(converted_message).__name__,
                                message_count=message_count,
                                request_id=request_id,
                            )
                            yield converted_message
                        except Exception as e:
                            logger.warning(
                                "claude_sdk_message_conversion_failed",
                                message_type=type(message).__name__,
                                error=str(e),
                            )
                            # Skip invalid messages rather than crashing
                            continue

                # Store final metrics
                op["message_count"] = message_count
                self._last_api_call_time_ms = op.get("duration_ms", 0.0)

                logger.debug(
                    "claude_sdk_query_completed",
                    message_count=message_count,
                    duration_ms=op.get("duration_ms"),
                    mode="pooled",
                )

            except Exception as e:
                logger.error(
                    "claude_sdk_pooled_query_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Fall back to stateless mode on pool errors
                logger.info("claude_sdk_nopool_mode")
                async for converted_message in self._query(
                    messages, options, request_id, session_id
                ):
                    yield converted_message

    async def _query_with_session_pool(
        self,
        messages: list[dict[str, Any]],
        options: ClaudeCodeOptions,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[
        sdk_models.UserMessage
        | sdk_models.AssistantMessage
        | sdk_models.SystemMessage
        | sdk_models.ResultMessage
    ]:
        """Execute query using session-aware pooled connection approach."""

        async with timed_operation("claude_sdk_query_session_pool", request_id) as op:
            try:
                logger.debug(
                    "claude_sdk_query_start",
                    messages_count=len(messages),
                    mode="session_pool",
                    session_id=session_id,
                )

                # Get pool manager
                if self._pool_manager is None:
                    raise ClaudeSDKError("No pool manager available")

                # Get session context (session_id is guaranteed to be non-None here)
                if not session_id:
                    raise ClaudeSDKError("Session ID required for session pool")

                session_ctx = await self._pool_manager.get_session_client(
                    session_id, options
                )

                message_count = 0

                async with (
                    session_ctx.lock
                ):  # Prevent concurrent access to same session
                    logger.debug(
                        "session_pool_lock_acquired",
                        session_id=session_id,
                        request_id=request_id,
                        session_status=session_ctx.status,
                    )

                    # Update session usage
                    session_ctx.update_usage()

                    # Convert Anthropic messages to SDK format
                    sdk_messages = self._convert_anthropic_messages_to_sdk(messages)

                    # Convert SDK messages to Claude SDK expected format
                    async def message_iter() -> AsyncIterator[dict[str, Any]]:
                        last_message = sdk_messages[-1]
                        logger.debug(
                            "last_message",
                            last_message_type=last_message.__module__,
                            last_message=last_message,
                        )
                        if isinstance(last_message, sdk_models.UserMessage):
                            content = last_message.content[0]
                            if isinstance(content, sdk_models.TextBlock):
                                message = {
                                    "type": "user",
                                    "message": {
                                        "role": "user",
                                        "content": content.text,
                                    },
                                    # "parent_tool_use_id": None,
                                    # "session_id": session_id,
                                }
                                logger.info("sending_sdk_message", message=message)
                                yield message
                            else:
                                raise ClaudeSDKError("Invalid message content")
                        else:
                            raise ClaudeSDKError("Invalid message")

                    # Send query to persistent client
                    logger.debug(
                        "session_pool_sending_query",
                        session_id=session_id,
                        request_id=request_id,
                        has_claude_client=bool(session_ctx.claude_client),
                    )

                    if session_id:
                        await session_ctx.claude_client.query(
                            message_iter(), session_id=session_id
                        )
                    else:
                        await session_ctx.claude_client.query(message_iter())

                    logger.debug(
                        "session_pool_query_sent_receiving_response",
                        session_id=session_id,
                        request_id=request_id,
                    )

                    # Receive and process all messages
                    async for message in session_ctx.claude_client.receive_response():
                        message_count += 1

                        logger.debug(
                            "claude_sdk_raw_message_received",
                            message_type=type(message).__name__,
                            message_count=message_count,
                            request_id=request_id,
                            session_id=session_id,
                            has_content=hasattr(message, "content")
                            and bool(getattr(message, "content", None)),
                            content_preview=str(message)[:150],
                        )

                        # Skip unknown message types early
                        if not isinstance(
                            message,
                            SDKUserMessage
                            | SDKAssistantMessage
                            | SDKSystemMessage
                            | SDKResultMessage,
                        ):
                            logger.warning(
                                "claude_sdk_unknown_message_type",
                                message_type=type(message).__name__,
                                request_id=request_id,
                                session_id=session_id,
                            )
                            continue

                        # Convert SDK message to our Pydantic model
                        try:
                            converted_message: (
                                sdk_models.UserMessage
                                | sdk_models.AssistantMessage
                                | sdk_models.SystemMessage
                                | sdk_models.ResultMessage
                            )
                            if isinstance(message, SDKUserMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.UserMessage
                                )
                            elif isinstance(message, SDKAssistantMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.AssistantMessage
                                )
                            elif isinstance(message, SDKSystemMessage):
                                converted_message = self._convert_message(
                                    message, sdk_models.SystemMessage
                                )
                            else:  # SDKResultMessage
                                converted_message = self._convert_message(
                                    message, sdk_models.ResultMessage
                                )
                                # Capture SDK session ID from result message
                                if isinstance(
                                    converted_message, sdk_models.ResultMessage
                                ):
                                    session_ctx.sdk_session_id = (
                                        converted_message.session_id
                                    )

                            logger.debug(
                                "claude_sdk_message_converted_successfully",
                                original_type=type(message).__name__,
                                converted_type=type(converted_message).__name__,
                                message_count=message_count,
                                request_id=request_id,
                                session_id=session_id,
                            )
                            yield converted_message

                        except Exception as e:
                            logger.warning(
                                "claude_sdk_message_conversion_failed",
                                message_type=type(message).__name__,
                                error=str(e),
                                session_id=session_id,
                            )
                            # Skip invalid messages rather than crashing
                            continue

                # Store final metrics
                op["message_count"] = message_count
                op["session_id"] = session_id
                self._last_api_call_time_ms = op.get("duration_ms", 0.0)

                logger.debug(
                    "claude_sdk_query_completed",
                    message_count=message_count,
                    duration_ms=op.get("duration_ms"),
                    mode="session_pool",
                    session_id=session_id,
                )

            except Exception as e:
                logger.error(
                    "claude_sdk_session_pool_query_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    session_id=session_id,
                    exc_info=True,
                )
                # Fall back to regular pool on session pool errors
                logger.info(
                    "claude_sdk_fallback_to_regular_pool", session_id=session_id
                )
                async for converted_message in self._query_with_pool(
                    messages, options, request_id, session_id
                ):
                    yield converted_message

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
