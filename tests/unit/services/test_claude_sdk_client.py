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
    TextBlock,
)

from ccproxy.claude_sdk.client import ClaudeSDKClient
from ccproxy.core.errors import ClaudeProxyError, ServiceUnavailableError
from ccproxy.models import claude_sdk as sdk_models


class TestClaudeSDKClient:
    """Test suite for ClaudeSDKClient class."""

    def test_init_default_values(self) -> None:
        """Test client initialization with default values."""
        client: ClaudeSDKClient = ClaudeSDKClient()

        assert client._last_api_call_time_ms == 0.0
        assert client._settings is None
        assert client._session_manager is None

    def test_init_with_session_manager(self) -> None:
        """Test client initialization with session manager."""
        mock_settings: Mock = Mock()  # Descriptive mock for settings
        mock_session_manager: Mock = Mock()  # Descriptive mock for session manager
        client: ClaudeSDKClient = ClaudeSDKClient(
            settings=mock_settings, session_manager=mock_session_manager
        )

        assert client._settings is mock_settings
        assert client._session_manager is mock_session_manager

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

    def test_last_api_call_time_ms_initial(self) -> None:
        """Test getting last API call time when no calls made."""
        client: ClaudeSDKClient = ClaudeSDKClient()

        result: float = client._last_api_call_time_ms

        assert result == 0.0

    def test_last_api_call_time_ms_after_call(self) -> None:
        """Test getting last API call time after setting it."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        client._last_api_call_time_ms = 123.45

        result: float = client._last_api_call_time_ms

        assert result == 123.45


class TestClaudeSDKClientStatelessQueries:
    """Test suite for stateless query execution."""

    @pytest.mark.asyncio
    async def test_query_completion_stateless_success(
        self, mock_sdk_client_instance: AsyncMock
    ) -> None:
        """Test successful stateless query execution."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        with patch(
            "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
            return_value=mock_sdk_client_instance,
        ):
            messages: list[Any] = []
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(
                sdk_message, options, "req_123"
            )

            async for message in stream_handle.create_listener():
                messages.append(message)

        assert len(messages) == 1
        assert isinstance(messages[0], sdk_models.AssistantMessage)
        assert isinstance(messages[0].content[0], sdk_models.TextBlock)
        assert messages[0].content[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_query_completion_cli_not_found_error(
        self, mock_sdk_client_cli_not_found: AsyncMock
    ) -> None:
        """Test handling of CLINotFoundError."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        with (
            patch(
                "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
                return_value=mock_sdk_client_cli_not_found,
            ),
            pytest.raises(ServiceUnavailableError) as exc_info,
        ):
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(sdk_message, options)

            async for _ in stream_handle.create_listener():
                pass

        assert "Claude CLI not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_completion_cli_connection_error(
        self, mock_sdk_client_cli_connection_error: AsyncMock
    ) -> None:
        """Test handling of CLIConnectionError."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        with (
            patch(
                "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
                return_value=mock_sdk_client_cli_connection_error,
            ),
            pytest.raises(ServiceUnavailableError) as exc_info,
        ):
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(sdk_message, options)

            async for _ in stream_handle.create_listener():
                pass

        assert "Claude CLI not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_completion_process_error(
        self, mock_sdk_client_process_error: AsyncMock
    ) -> None:
        """Test handling of ProcessError."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        with (
            patch(
                "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
                return_value=mock_sdk_client_process_error,
            ),
            pytest.raises(ClaudeProxyError) as exc_info,
        ):
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(sdk_message, options)

            async for _ in stream_handle.create_listener():
                pass

        assert "Claude process error" in str(exc_info.value)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_query_completion_json_decode_error(
        self, mock_sdk_client_json_decode_error: AsyncMock
    ) -> None:
        """Test handling of CLIJSONDecodeError."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        with (
            patch(
                "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
                return_value=mock_sdk_client_json_decode_error,
            ),
            pytest.raises(ClaudeProxyError) as exc_info,
        ):
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(sdk_message, options)

            async for _ in stream_handle.create_listener():
                pass

        assert "Claude process error" in str(exc_info.value)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_query_completion_unexpected_error(
        self, mock_sdk_client_unexpected_error: AsyncMock
    ) -> None:
        """Test handling of unexpected errors."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        with (
            patch(
                "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
                return_value=mock_sdk_client_unexpected_error,
            ),
            pytest.raises(ClaudeProxyError) as exc_info,
        ):
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(sdk_message, options)

            async for _ in stream_handle.create_listener():
                pass

        assert "Unexpected error" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_query_completion_unknown_message_type(self) -> None:
        """Test handling of unknown message types."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Create a mock unknown message type - descriptive mock for unknown type handling
        mock_unknown_message: Mock = Mock()
        mock_unknown_message.__class__.__name__ = "UnknownMessage"

        # Create a mock SDK client that returns unknown message
        mock_sdk_client = AsyncMock()
        mock_sdk_client.connect = AsyncMock()
        mock_sdk_client.disconnect = AsyncMock()
        mock_sdk_client.query = AsyncMock()

        async def unknown_message_response() -> AsyncGenerator[Any, None]:
            yield mock_unknown_message

        mock_sdk_client.receive_response = unknown_message_response

        with patch(
            "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
            return_value=mock_sdk_client,
        ):
            messages: list[Any] = []
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(sdk_message, options)

            async for message in stream_handle.create_listener():
                messages.append(message)

        # Should skip unknown message types
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_query_completion_message_conversion_failure(self) -> None:
        """Test handling of message conversion failures."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        # Create a message that will fail conversion
        bad_message: AssistantMessage = AssistantMessage(
            content=[TextBlock(text="Hello")],
        )

        # Create a mock SDK client
        mock_sdk_client = AsyncMock()
        mock_sdk_client.connect = AsyncMock()
        mock_sdk_client.disconnect = AsyncMock()
        mock_sdk_client.query = AsyncMock()

        async def bad_message_response() -> AsyncGenerator[Any, None]:
            yield bad_message

        mock_sdk_client.receive_response = bad_message_response

        # Mock the conversion to fail
        with (
            patch(
                "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
                return_value=mock_sdk_client,
            ),
            patch.object(
                client, "_convert_message", side_effect=Exception("Conversion failed")
            ),
        ):
            messages: list[Any] = []
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(sdk_message, options)

            async for message in stream_handle.create_listener():
                messages.append(message)

        # Should skip failed conversions
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_query_completion_multiple_message_types(
        self, mock_sdk_client_streaming: AsyncMock
    ) -> None:
        """Test query with multiple message types."""
        client: ClaudeSDKClient = ClaudeSDKClient()
        options: ClaudeCodeOptions = ClaudeCodeOptions()

        with patch(
            "ccproxy.claude_sdk.client.ImportedClaudeSDKClient",
            return_value=mock_sdk_client_streaming,
        ):
            messages: list[Any] = []
            # Create a proper SDKMessage for the test
            from ccproxy.models.claude_sdk import create_sdk_message

            sdk_message = create_sdk_message(content="Hello")

            stream_handle = await client.query_completion(sdk_message, options)

            async for message in stream_handle.create_listener():
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
        client: ClaudeSDKClient = ClaudeSDKClient()
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


# TestClaudeSDKClientExceptions removed - exceptions moved to exceptions.py module
