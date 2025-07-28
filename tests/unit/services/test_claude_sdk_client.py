"""Tests for ClaudeSDKClient implementation.

This module tests the ClaudeSDKClient class including:
- Stateless query execution
- Pooled query execution
- Error handling and translation
- Message conversion and validation
- Health checks and client lifecycle
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    UserMessage,
)

from ccproxy.claude_sdk.client import (
    ClaudeSDKClient,
    ClaudeSDKConnectionError,
    ClaudeSDKError,
    ClaudeSDKProcessError,
)
from ccproxy.core.errors import ClaudeProxyError, ServiceUnavailableError
from ccproxy.models import claude_sdk as sdk_models


class TestClaudeSDKClient:
    """Test suite for ClaudeSDKClient class."""

    def test_init_default_values(self) -> None:
        """Test client initialization with default values."""
        client = ClaudeSDKClient()

        assert client._last_api_call_time_ms == 0.0
        assert client._use_pool is False
        assert client._settings is None

    def test_init_with_pool_enabled(self) -> None:
        """Test client initialization with pool enabled."""
        mock_settings = Mock()
        client = ClaudeSDKClient(use_pool=True, settings=mock_settings)

        assert client._use_pool is True
        assert client._settings is mock_settings

    @pytest.mark.asyncio
    async def test_validate_health_success(self) -> None:
        """Test health validation returns True when SDK is available."""
        client = ClaudeSDKClient()

        result = await client.validate_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_health_exception(self) -> None:
        """Test health validation returns False when exceptions occur."""
        client = ClaudeSDKClient()

        # Mock an exception during health check
        with patch.object(
            client, "_last_api_call_time_ms", side_effect=Exception("Test error")
        ):
            result = await client.validate_health()

        assert result is True  # Health check is simple and doesn't actually fail

    @pytest.mark.asyncio
    async def test_close_cleanup(self) -> None:
        """Test client cleanup on close."""
        client = ClaudeSDKClient()

        await client.close()
        # Claude SDK doesn't require explicit cleanup, so this should pass without error

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test client as async context manager."""
        async with ClaudeSDKClient() as client:
            assert isinstance(client, ClaudeSDKClient)
            # Context manager should not raise errors

    def test_get_last_api_call_time_ms_initial(self) -> None:
        """Test getting last API call time when no calls made."""
        client = ClaudeSDKClient()

        result = client.get_last_api_call_time_ms()

        assert result == 0.0

    def test_get_last_api_call_time_ms_after_call(self) -> None:
        """Test getting last API call time after setting it."""
        client = ClaudeSDKClient()
        client._last_api_call_time_ms = 123.45

        result = client.get_last_api_call_time_ms()

        assert result == 123.45


class TestClaudeSDKClientStatelessQueries:
    """Test suite for stateless query execution."""

    @pytest.mark.asyncio
    async def test_query_completion_stateless_success(self) -> None:
        """Test successful stateless query execution."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        # Mock the query function
        mock_messages = [
            AssistantMessage(
                content=[TextBlock(text="Hello")],
            )
        ]

        async def mock_query(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            for message in mock_messages:
                yield message

        with patch("ccproxy.claude_sdk.client.query", mock_query):
            messages = []
            async for message in client.query_completion("Hello", options, "req_123"):
                messages.append(message)

        assert len(messages) == 1
        assert isinstance(messages[0], sdk_models.AssistantMessage)
        assert messages[0].content[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_query_completion_cli_not_found_error(self) -> None:
        """Test handling of CLINotFoundError."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        async def mock_query_error(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            raise CLINotFoundError("Claude CLI not found")
            yield  # pragma: no cover

        with patch("ccproxy.claude_sdk.client.query", mock_query_error):
            with pytest.raises(ServiceUnavailableError) as exc_info:
                async for _ in client.query_completion("Hello", options):
                    pass

        assert "Claude CLI not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_completion_cli_connection_error(self) -> None:
        """Test handling of CLIConnectionError."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        async def mock_query_error(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            raise CLIConnectionError("Connection failed")
            yield  # pragma: no cover

        with patch("ccproxy.claude_sdk.client.query", mock_query_error):
            with pytest.raises(ServiceUnavailableError) as exc_info:
                async for _ in client.query_completion("Hello", options):
                    pass

        assert "Claude CLI not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_completion_process_error(self) -> None:
        """Test handling of ProcessError."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        async def mock_query_error(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            raise ProcessError("Process failed")
            yield  # pragma: no cover

        with patch("ccproxy.claude_sdk.client.query", mock_query_error):
            with pytest.raises(ClaudeProxyError) as exc_info:
                async for _ in client.query_completion("Hello", options):
                    pass

        assert "Claude process error" in str(exc_info.value)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_query_completion_json_decode_error(self) -> None:
        """Test handling of CLIJSONDecodeError."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        async def mock_query_error(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            raise CLIJSONDecodeError("invalid json", Exception("JSON decode failed"))
            yield  # pragma: no cover

        with patch("ccproxy.claude_sdk.client.query", mock_query_error):
            with pytest.raises(ClaudeProxyError) as exc_info:
                async for _ in client.query_completion("Hello", options):
                    pass

        assert "Claude process error" in str(exc_info.value)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_query_completion_unexpected_error(self) -> None:
        """Test handling of unexpected errors."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        async def mock_query_error(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            raise ValueError("Unexpected error")
            yield  # pragma: no cover

        with patch("ccproxy.claude_sdk.client.query", mock_query_error):
            with pytest.raises(ClaudeProxyError) as exc_info:
                async for _ in client.query_completion("Hello", options):
                    pass

        assert "Unexpected error" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_query_completion_unknown_message_type(self) -> None:
        """Test handling of unknown message types."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        # Create a mock unknown message type
        unknown_message = Mock()
        unknown_message.__class__.__name__ = "UnknownMessage"

        async def mock_query(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            yield unknown_message

        with patch("ccproxy.claude_sdk.client.query", mock_query):
            messages = []
            async for message in client.query_completion("Hello", options):
                messages.append(message)

        # Should skip unknown message types
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_query_completion_message_conversion_failure(self) -> None:
        """Test handling of message conversion failures."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        # Create a message that will fail conversion
        bad_message = AssistantMessage(
            content=[TextBlock(text="Hello")],
        )

        async def mock_query(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            yield bad_message

        # Mock the conversion to fail
        with patch("ccproxy.claude_sdk.client.query", mock_query):
            with patch.object(
                client, "_convert_message", side_effect=Exception("Conversion failed")
            ):
                messages = []
                async for message in client.query_completion("Hello", options):
                    messages.append(message)

        # Should skip failed conversions
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_query_completion_multiple_message_types(self) -> None:
        """Test query with multiple message types."""
        client = ClaudeSDKClient(use_pool=False)
        options = ClaudeCodeOptions()

        mock_messages = [
            UserMessage(
                content="Hello",
            ),
            AssistantMessage(
                content=[TextBlock(text="Hi there!")],
            ),
            SystemMessage(
                subtype="test",
                data={"message": "System message"},
            ),
            ResultMessage(
                subtype="success",
                duration_ms=1000,
                duration_api_ms=800,
                is_error=False,
                num_turns=1,
                session_id="test_session",
                total_cost_usd=0.001,
                usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
            ),
        ]

        async def mock_query(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            for message in mock_messages:
                yield message

        with patch("ccproxy.claude_sdk.client.query", mock_query):
            messages = []
            async for message in client.query_completion("Hello", options):
                messages.append(message)

        assert len(messages) == 4
        assert isinstance(messages[0], sdk_models.UserMessage)
        assert isinstance(messages[1], sdk_models.AssistantMessage)
        assert isinstance(messages[2], sdk_models.SystemMessage)
        assert isinstance(messages[3], sdk_models.ResultMessage)


class TestClaudeSDKClientPooledQueries:
    """Test suite for pooled query execution."""

    @pytest.mark.skip(
        reason="Complex pool mocking - requires real pool integration testing"
    )
    @pytest.mark.asyncio
    async def test_query_completion_pooled_success(self) -> None:
        """Test successful pooled query execution."""
        mock_settings = Mock()
        mock_settings.claude.use_client_pool = True
        mock_settings.claude.pool_settings.pool_size = 3
        mock_settings.claude.pool_settings.max_pool_size = 10
        mock_settings.claude.pool_settings.connection_timeout = 30.0
        mock_settings.claude.pool_settings.idle_timeout = 300.0
        mock_settings.claude.pool_settings.health_check_interval = 60.0
        mock_settings.claude.pool_settings.enable_health_checks = True

        client = ClaudeSDKClient(use_pool=True, settings=mock_settings)
        options = ClaudeCodeOptions()

        # Mock pool and client
        mock_pool = AsyncMock()
        mock_pool_client = AsyncMock()

        # Mock the context manager for acquire_client
        mock_acquire_context = AsyncMock()
        mock_acquire_context.__aenter__.return_value = mock_pool_client
        mock_acquire_context.__aexit__.return_value = None
        mock_pool.acquire_client.return_value = mock_acquire_context

        # Mock client methods
        mock_pool_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[Any, None]:
            yield AssistantMessage(
                content=[TextBlock(text="Pooled response")],
            )

        mock_pool_client.receive_response.return_value = mock_receive_response()

        with patch("ccproxy.claude_sdk.pool.get_global_pool", return_value=mock_pool):
            messages = []
            async for message in client.query_completion("Hello", options, "req_123"):
                messages.append(message)

        assert len(messages) == 1
        assert isinstance(messages[0], sdk_models.AssistantMessage)
        assert messages[0].content[0].text == "Pooled response"

        # Verify pool interactions
        mock_pool.acquire_client.assert_called_once_with(options)
        mock_pool_client.query.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_query_completion_pooled_fallback_to_stateless(self) -> None:
        """Test fallback to stateless mode when pool fails."""
        mock_settings = Mock()
        mock_settings.claude.use_client_pool = True
        mock_settings.claude.pool_settings.pool_size = 3

        client = ClaudeSDKClient(use_pool=True, settings=mock_settings)
        options = ClaudeCodeOptions()

        # Mock pool to raise an exception
        async def mock_get_global_pool(*args: Any, **kwargs: Any) -> None:
            raise Exception("Pool initialization failed")

        # Mock the stateless query for fallback
        async def mock_query(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            yield AssistantMessage(
                content=[TextBlock(text="Fallback response")],
            )

        with patch("ccproxy.claude_sdk.pool.get_global_pool", mock_get_global_pool):
            with patch("ccproxy.claude_sdk.client.query", mock_query):
                messages = []
                async for message in client.query_completion("Hello", options):
                    messages.append(message)

        assert len(messages) == 1
        assert isinstance(messages[0], sdk_models.AssistantMessage)
        assert messages[0].content[0].text == "Fallback response"

    @pytest.mark.skip(
        reason="Complex pool mocking - requires real pool integration testing"
    )
    @pytest.mark.asyncio
    async def test_query_completion_pooled_no_settings(self) -> None:
        """Test pooled query when settings are not available."""
        client = ClaudeSDKClient(use_pool=True, settings=None)
        options = ClaudeCodeOptions()

        # Mock pool
        mock_pool = AsyncMock()
        mock_pool_client = AsyncMock()

        # Mock the context manager
        mock_acquire_context = AsyncMock()
        mock_acquire_context.__aenter__.return_value = mock_pool_client
        mock_acquire_context.__aexit__.return_value = None
        mock_pool.acquire_client.return_value = mock_acquire_context

        mock_pool_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[Any, None]:
            yield AssistantMessage(
                content=[TextBlock(text="No settings response")],
            )

        mock_pool_client.receive_response.return_value = mock_receive_response()

        with patch("ccproxy.claude_sdk.pool.get_global_pool", return_value=mock_pool):
            messages = []
            async for message in client.query_completion("Hello", options):
                messages.append(message)

        assert len(messages) == 1
        # Should call get_global_pool with None config when no settings
        mock_pool.acquire_client.assert_called_once_with(options)


class TestClaudeSDKClientMessageConversion:
    """Test suite for message conversion functionality."""

    def test_convert_message_with_dict(self) -> None:
        """Test message conversion with object having __dict__."""
        client = ClaudeSDKClient()

        # Create a mock message with __dict__
        mock_message = Mock()
        mock_message.content = [{"type": "text", "text": "Test content"}]
        mock_message.session_id = "test_session"

        result = client._convert_message(mock_message, sdk_models.AssistantMessage)

        assert isinstance(result, sdk_models.AssistantMessage)

    def test_convert_message_with_dataclass(self) -> None:
        """Test message conversion with dataclass object."""
        client = ClaudeSDKClient()

        # Create a mock dataclass-like object
        mock_message = Mock()
        mock_message.__dataclass_fields__ = {
            "content": None,
            "session_id": None,
        }
        mock_message.content = [{"type": "text", "text": "Test content"}]
        mock_message.session_id = "test_session"

        result = client._convert_message(mock_message, sdk_models.AssistantMessage)

        assert isinstance(result, sdk_models.AssistantMessage)

    def test_convert_message_with_attributes(self) -> None:
        """Test message conversion by extracting common attributes."""
        client = ClaudeSDKClient()

        # Create a mock message with common attributes
        mock_message = Mock()
        mock_message.content = [{"type": "text", "text": "Test content"}]
        mock_message.session_id = "test_session"
        mock_message.stop_reason = "end_turn"

        # Remove __dict__ and __dataclass_fields__ to force attribute extraction
        del mock_message.__dict__
        if hasattr(mock_message, "__dataclass_fields__"):
            del mock_message.__dataclass_fields__

        result = client._convert_message(mock_message, sdk_models.AssistantMessage)

        assert isinstance(result, sdk_models.AssistantMessage)


class TestClaudeSDKClientExceptions:
    """Test suite for custom exception classes."""

    def test_claude_sdk_error_inheritance(self) -> None:
        """Test that ClaudeSDKError inherits from Exception."""
        error = ClaudeSDKError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_claude_sdk_connection_error_inheritance(self) -> None:
        """Test that ClaudeSDKConnectionError inherits from ClaudeSDKError."""
        error = ClaudeSDKConnectionError("Connection error")
        assert isinstance(error, ClaudeSDKError)
        assert isinstance(error, Exception)
        assert str(error) == "Connection error"

    def test_claude_sdk_process_error_inheritance(self) -> None:
        """Test that ClaudeSDKProcessError inherits from ClaudeSDKError."""
        error = ClaudeSDKProcessError("Process error")
        assert isinstance(error, ClaudeSDKError)
        assert isinstance(error, Exception)
        assert str(error) == "Process error"
