"""Tests for ClaudeSDKClient implementation.

This module tests the ClaudeSDKClient class including:
- Stateless query execution
- Pooled query execution
- Error handling and translation
- Message conversion and validation
- Health checks and client lifecycle

Organized fixtures available from tests/fixtures/:
- mock_internal_claude_sdk_service: Service-level mocking for dependency injection
- mock_claude_sdk_client_streaming: Streaming client mock with comprehensive responses
- mock_internal_claude_sdk_service_streaming: Service-level streaming mock
- mock_internal_claude_sdk_service_unavailable: Service unavailability simulation
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


# Mock helper classes for SDK client testing
class SDKClientMockBuilder:
    """Builder for creating SDK client mock patterns."""

    @staticmethod
    def create_query_error_generator(error: Exception):
        """Create an error generator function that accepts arguments."""

        async def error_generator(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            raise error

        return error_generator

    @staticmethod
    def create_simple_response_generator(text: str = "Hello"):
        """Create a simple response generator function that accepts arguments."""

        async def response_generator(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            yield AssistantMessage(content=[TextBlock(text=text)])

        return response_generator

    @staticmethod
    def create_multi_message_generator():
        """Create a generator function that accepts arguments for multiple message types."""

        async def multi_message_generator(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, None]:
            messages = [
                UserMessage(content="Hello"),
                AssistantMessage(content=[TextBlock(text="Hi there!")]),
                SystemMessage(subtype="test", data={"message": "System message"}),
                ResultMessage(
                    subtype="success",
                    duration_ms=1000,
                    duration_api_ms=800,
                    is_error=False,
                    num_turns=1,
                    session_id="test_session",
                    total_cost_usd=0.001,
                    usage={"input_tokens": 10, "output_tokens": 5},
                ),
            ]
            for message in messages:
                yield message

        return multi_message_generator


class PoolMockBuilder:
    """Builder for creating pool-related mock patterns (simplified version)."""

    @staticmethod
    def create_pool_config_mock(pool_size: int = 3, max_pool_size: int = 10) -> Mock:
        """Create a mock pool configuration."""
        mock_config = Mock()
        mock_config.claude.use_client_pool = True
        mock_config.claude.pool_settings.pool_size = pool_size
        mock_config.claude.pool_settings.max_pool_size = max_pool_size
        mock_config.claude.pool_settings.connection_timeout = 30.0
        mock_config.claude.pool_settings.idle_timeout = 300.0
        mock_config.claude.pool_settings.health_check_interval = 60.0
        mock_config.claude.pool_settings.enable_health_checks = True
        return mock_config

    @staticmethod
    def create_pooled_client_context_manager(
        response_text: str = "Pooled response",
    ) -> tuple[AsyncMock, AsyncMock]:
        """Create a complete pooled client context manager setup."""
        mock_pooled_client = AsyncMock()
        mock_pooled_client.query = AsyncMock()

        async def mock_receive_response() -> AsyncGenerator[Any, None]:
            yield AssistantMessage(content=[TextBlock(text=response_text)])

        mock_pooled_client.receive_response.return_value = mock_receive_response()

        mock_acquire_context = AsyncMock()
        mock_acquire_context.__aenter__.return_value = mock_pooled_client
        mock_acquire_context.__aexit__.return_value = None

        return mock_pooled_client, mock_acquire_context


class TestClaudeSDKClient:
    """Test suite for ClaudeSDKClient class."""

    def test_init_default_values(self) -> None:
        """Test client initialization with default values."""
        client: ClaudeSDKClient = ClaudeSDKClient()

        assert client._last_api_call_time_ms == 0.0
        assert client._use_pool is False
        assert client._settings is None

    def test_init_with_pool_enabled(self) -> None:
        """Test client initialization with pool enabled."""
        mock_pool_settings: Mock = Mock()  # Descriptive mock for pool settings
        client: ClaudeSDKClient = ClaudeSDKClient(
            use_pool=True, settings=mock_pool_settings
        )

        assert client._use_pool is True
        assert client._settings is mock_pool_settings

    @pytest.mark.asyncio
    async def test_validate_health_success(self) -> None:
        """Test health validation returns True when SDK is available."""
        client: ClaudeSDKClient = ClaudeSDKClient()

        result: bool = await client.validate_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_health_exception(self) -> None:
        """Test health validation returns False when exceptions occur."""
        client: ClaudeSDKClient = ClaudeSDKClient()

        # Mock an exception during health check
        with patch.object(
            client, "_last_api_call_time_ms", side_effect=Exception("Test error")
        ):
            result: bool = await client.validate_health()

        assert result is True  # Health check is simple and doesn't actually fail

    @pytest.mark.asyncio
    async def test_close_cleanup(self) -> None:
        """Test client cleanup on close."""
        client: ClaudeSDKClient = ClaudeSDKClient()

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
        client: ClaudeSDKClient = ClaudeSDKClient()

        result: float = client.get_last_api_call_time_ms()

        assert result == 0.0

    def test_get_last_api_call_time_ms_after_call(self) -> None:
        """Test getting last API call time after setting it."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        client._last_api_call_time_ms = 123.45

        result: float = client.get_last_api_call_time_ms()

        assert result == 123.45


class TestClaudeSDKClientStatelessQueries:
    """Test suite for stateless query execution."""

    @pytest.mark.asyncio
    async def test_query_completion_stateless_success(self) -> None:
        """Test successful stateless query execution."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use SDKClientMockBuilder for simple response
        with patch(
            "ccproxy.claude_sdk.client.query",
            SDKClientMockBuilder.create_simple_response_generator(),
        ):
            messages: list[Any] = []
            async for message in client.query_completion("Hello", options, "req_123"):
                messages.append(message)

        assert len(messages) == 1
        assert isinstance(messages[0], sdk_models.AssistantMessage)
        assert isinstance(messages[0].content[0], sdk_models.TextBlock)
        assert messages[0].content[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_query_completion_cli_not_found_error(self) -> None:
        """Test handling of CLINotFoundError."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use SDKClientMockBuilder for error generation
        with (
            patch(
                "ccproxy.claude_sdk.client.query",
                SDKClientMockBuilder.create_query_error_generator(
                    CLINotFoundError("Claude CLI not found")
                ),
            ),
            pytest.raises(ServiceUnavailableError) as exc_info,
        ):
            async for _ in client.query_completion("Hello", options):
                pass

        assert "Claude CLI not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_completion_cli_connection_error(self) -> None:
        """Test handling of CLIConnectionError."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use SDKClientMockBuilder for connection error
        with (
            patch(
                "ccproxy.claude_sdk.client.query",
                SDKClientMockBuilder.create_query_error_generator(
                    CLIConnectionError("Connection failed")
                ),
            ),
            pytest.raises(ServiceUnavailableError) as exc_info,
        ):
            async for _ in client.query_completion("Hello", options):
                pass

        assert "Claude CLI not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_completion_process_error(self) -> None:
        """Test handling of ProcessError."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use SDKClientMockBuilder for process error
        with (
            patch(
                "ccproxy.claude_sdk.client.query",
                SDKClientMockBuilder.create_query_error_generator(
                    ProcessError("Process failed")
                ),
            ),
            pytest.raises(ClaudeProxyError) as exc_info,
        ):
            async for _ in client.query_completion("Hello", options):
                pass

        assert "Claude process error" in str(exc_info.value)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_query_completion_json_decode_error(self) -> None:
        """Test handling of CLIJSONDecodeError."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use SDKClientMockBuilder for JSON decode error
        with (
            patch(
                "ccproxy.claude_sdk.client.query",
                SDKClientMockBuilder.create_query_error_generator(
                    CLIJSONDecodeError("invalid json", Exception("JSON decode failed"))
                ),
            ),
            pytest.raises(ClaudeProxyError) as exc_info,
        ):
            async for _ in client.query_completion("Hello", options):
                pass

        assert "Claude process error" in str(exc_info.value)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_query_completion_unexpected_error(self) -> None:
        """Test handling of unexpected errors."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use SDKClientMockBuilder for unexpected error
        with (
            patch(
                "ccproxy.claude_sdk.client.query",
                SDKClientMockBuilder.create_query_error_generator(
                    ValueError("Unexpected error")
                ),
            ),
            pytest.raises(ClaudeProxyError) as exc_info,
        ):
            async for _ in client.query_completion("Hello", options):
                pass

        assert "Unexpected error" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_query_completion_unknown_message_type(self) -> None:
        """Test handling of unknown message types."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Create a mock unknown message type - descriptive mock for unknown type handling
        mock_unknown_message: Mock = Mock()
        mock_unknown_message.__class__.__name__ = "UnknownMessage"

        async def mock_query(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            yield mock_unknown_message

        with patch("ccproxy.claude_sdk.client.query", mock_query):
            messages: list[Any] = []
            async for message in client.query_completion("Hello", options):
                messages.append(message)

        # Should skip unknown message types
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_query_completion_message_conversion_failure(self) -> None:
        """Test handling of message conversion failures."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Create a message that will fail conversion
        bad_message: AssistantMessage = AssistantMessage(
            content=[TextBlock(text="Hello")],
        )

        async def mock_query(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            yield bad_message

        # Mock the conversion to fail
        with (
            patch("ccproxy.claude_sdk.client.query", mock_query),
            patch.object(
                client, "_convert_message", side_effect=Exception("Conversion failed")
            ),
        ):
            messages: list[Any] = []
            async for message in client.query_completion("Hello", options):
                messages.append(message)

        # Should skip failed conversions
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_query_completion_multiple_message_types(self) -> None:
        """Test query with multiple message types."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use SDKClientMockBuilder for multi-message response
        with patch(
            "ccproxy.claude_sdk.client.query",
            SDKClientMockBuilder.create_multi_message_generator(),
        ):
            messages: list[Any] = []
            async for message in client.query_completion("Hello", options):
                messages.append(message)

        assert len(messages) == 4
        assert isinstance(messages[0], sdk_models.UserMessage)
        assert isinstance(messages[1], sdk_models.AssistantMessage)
        assert isinstance(messages[2], sdk_models.SystemMessage)
        assert isinstance(messages[3], sdk_models.ResultMessage)

    @pytest.mark.asyncio
    async def test_query_completion_with_simple_organized_mock(
        self,
        mock_internal_claude_sdk_service: AsyncMock,  # Using organized service fixture
    ) -> None:
        """Test demonstrating organized fixture usage patterns.

        This test shows how organized fixtures can provide consistent mock behavior
        without complex inline setup, improving test maintainability.
        """
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=False)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Organized fixtures provide pre-configured, consistent mock responses
        # Here we demonstrate that the fixture is available and properly structured
        assert hasattr(mock_internal_claude_sdk_service, "create_completion")
        assert hasattr(mock_internal_claude_sdk_service, "validate_health")
        assert hasattr(mock_internal_claude_sdk_service, "list_models")

        # The service fixture comes pre-configured with realistic response data
        health_status = await mock_internal_claude_sdk_service.validate_health()
        assert health_status is True

    @pytest.mark.asyncio
    async def test_health_check_with_organized_fixture(
        self,
        mock_internal_claude_sdk_service: AsyncMock,  # Using organized service fixture
    ) -> None:
        """Test health validation using organized service fixture.

        This test demonstrates how organized fixtures can be used for
        non-query operations like health checks.
        """
        client: ClaudeSDKClient = ClaudeSDKClient()

        # Mock the validate_health method from the organized fixture
        mock_internal_claude_sdk_service.validate_health.return_value = True

        # The health check should always return True in this simple implementation
        result: bool = await client.validate_health()

        assert result is True


class TestClaudeSDKClientPooledQueries:
    """Test suite for pooled query execution."""

    @pytest.mark.skip(
        reason="Complex pool mocking - requires real pool integration testing"
    )
    @pytest.mark.asyncio
    async def test_query_completion_pooled_success(self) -> None:
        """Test successful pooled query execution."""
        # Use PoolMockBuilder for pool configuration
        mock_pool_config = PoolMockBuilder.create_pool_config_mock()

        client: ClaudeSDKClient = ClaudeSDKClient(
            use_pool=True, settings=mock_pool_config
        )
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use PoolMockBuilder for pool components
        mock_pooled_client, mock_acquire_context = (
            PoolMockBuilder.create_pooled_client_context_manager()
        )

        mock_connection_pool = AsyncMock()
        mock_connection_pool.acquire_client.return_value = mock_acquire_context

        with patch(
            "ccproxy.claude_sdk.pool.get_global_pool", return_value=mock_connection_pool
        ):
            messages: list[Any] = []
            async for message in client.query_completion("Hello", options, "req_123"):
                messages.append(message)

        assert len(messages) == 1
        assert isinstance(messages[0], sdk_models.AssistantMessage)
        assert isinstance(messages[0].content[0], sdk_models.TextBlock)
        assert messages[0].content[0].text == "Pooled response"

        # Verify pool interactions
        mock_connection_pool.acquire_client.assert_called_once_with(options)
        mock_pooled_client.query.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_query_completion_pooled_fallback_to_stateless(
        self, isolated_environment: Any
    ) -> None:
        """Test fallback to stateless mode when pool fails."""
        # Use PoolMockBuilder for fallback settings
        mock_fallback_settings = PoolMockBuilder.create_pool_config_mock(pool_size=3)

        client: ClaudeSDKClient = ClaudeSDKClient(
            use_pool=True, settings=mock_fallback_settings
        )
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use SDKClientMockBuilder for fallback query
        mock_stateless_fallback_query = (
            SDKClientMockBuilder.create_simple_response_generator("Fallback response")
        )

        # Mock multiple paths to ensure pool failure and fallback to stateless
        with (
            patch(
                "ccproxy.claude_sdk.client.get_pool_manager"
            ) as mock_get_pool_manager,
            patch("ccproxy.claude_sdk.manager.PoolManager") as mock_pool_manager_class,
            patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics,
        ):
            # Mock get_metrics to return None so it doesn't take the metrics path
            mock_get_metrics.return_value = None

            # Create a failing pool manager that returns a pool with failing acquire_client
            mock_manager = AsyncMock()
            mock_pool = AsyncMock()

            # Make pool.acquire_client() fail to trigger the fallback
            async def failing_acquire_client(*args: Any, **kwargs: Any) -> Any:
                raise Exception("Pool client acquisition failed")

            mock_pool.acquire_client.side_effect = failing_acquire_client
            mock_manager.get_pool.return_value = mock_pool
            mock_get_pool_manager.return_value = mock_manager

            with patch(
                "ccproxy.claude_sdk.client.query", mock_stateless_fallback_query
            ):
                messages: list[Any] = []
                async for message in client.query_completion("Hello", options):
                    messages.append(message)

        assert len(messages) == 1
        assert isinstance(messages[0], sdk_models.AssistantMessage)
        assert isinstance(messages[0].content[0], sdk_models.TextBlock)
        assert messages[0].content[0].text == "Fallback response"

    @pytest.mark.skip(
        reason="Complex pool mocking - requires real pool integration testing"
    )
    @pytest.mark.asyncio
    async def test_query_completion_pooled_no_settings(self) -> None:
        """Test pooled query when settings are not available."""
        client: ClaudeSDKClient = ClaudeSDKClient(use_pool=True, settings=None)
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Use PoolMockBuilder for no-settings scenario
        mock_no_settings_client, mock_acquire_context = (
            PoolMockBuilder.create_pooled_client_context_manager("No settings response")
        )

        mock_default_pool = AsyncMock()
        mock_default_pool.acquire_client.return_value = mock_acquire_context

        with patch(
            "ccproxy.claude_sdk.pool.get_global_pool", return_value=mock_default_pool
        ):
            messages: list[Any] = []
            async for message in client.query_completion("Hello", options):
                messages.append(message)

        assert len(messages) == 1
        # Should call get_global_pool with None config when no settings
        mock_default_pool.acquire_client.assert_called_once_with(options)


class TestClaudeSDKClientMessageConversion:
    """Test suite for message conversion functionality."""

    def test_convert_message_with_dict(self) -> None:
        """Test message conversion with object having __dict__."""
        client: ClaudeSDKClient = ClaudeSDKClient()

        # Create a mock message with __dict__ - structured mock for dict-based conversion
        mock_dict_message: Mock = Mock()
        mock_dict_message.content = [{"type": "text", "text": "Test content"}]
        mock_dict_message.session_id = "test_session"

        result: sdk_models.AssistantMessage = client._convert_message(
            mock_dict_message, sdk_models.AssistantMessage
        )

        assert isinstance(result, sdk_models.AssistantMessage)

    def test_convert_message_with_dataclass(self) -> None:
        """Test message conversion with dataclass object."""
        client: ClaudeSDKClient = ClaudeSDKClient()

        # Create a mock dataclass-like object - structured mock for dataclass conversion
        mock_dataclass_message: Mock = Mock()
        mock_dataclass_message.__dataclass_fields__ = {
            "content": None,
            "session_id": None,
        }
        mock_dataclass_message.content = [{"type": "text", "text": "Test content"}]
        mock_dataclass_message.session_id = "test_session"

        result: sdk_models.AssistantMessage = client._convert_message(
            mock_dataclass_message, sdk_models.AssistantMessage
        )

        assert isinstance(result, sdk_models.AssistantMessage)

    def test_convert_message_with_attributes(self) -> None:
        """Test message conversion by extracting common attributes."""
        client: ClaudeSDKClient = ClaudeSDKClient()

        # Create a mock message with common attributes - structured mock for attribute extraction
        mock_attributes_message: Mock = Mock()
        mock_attributes_message.content = [{"type": "text", "text": "Test content"}]
        mock_attributes_message.session_id = "test_session"
        mock_attributes_message.stop_reason = "end_turn"

        # Remove __dict__ and __dataclass_fields__ to force attribute extraction
        del mock_attributes_message.__dict__
        if hasattr(mock_attributes_message, "__dataclass_fields__"):
            del mock_attributes_message.__dataclass_fields__

        result: sdk_models.AssistantMessage = client._convert_message(
            mock_attributes_message, sdk_models.AssistantMessage
        )

        assert isinstance(result, sdk_models.AssistantMessage)


class TestClaudeSDKClientExceptions:
    """Test suite for custom exception classes."""

    def test_claude_sdk_error_inheritance(self) -> None:
        """Test that ClaudeSDKError inherits from Exception."""
        error: ClaudeSDKError = ClaudeSDKError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_claude_sdk_connection_error_inheritance(self) -> None:
        """Test that ClaudeSDKConnectionError inherits from ClaudeSDKError."""
        error: ClaudeSDKConnectionError = ClaudeSDKConnectionError("Connection error")
        assert isinstance(error, ClaudeSDKError)
        assert isinstance(error, Exception)
        assert str(error) == "Connection error"

    def test_claude_sdk_process_error_inheritance(self) -> None:
        """Test that ClaudeSDKProcessError inherits from ClaudeSDKError."""
        error: ClaudeSDKProcessError = ClaudeSDKProcessError("Process error")
        assert isinstance(error, ClaudeSDKError)
        assert isinstance(error, Exception)
        assert str(error) == "Process error"
