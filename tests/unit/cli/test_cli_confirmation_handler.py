"""Tests for CLI confirmation handler SSE client."""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import typer

from ccproxy.api.services.permission_service import PermissionRequest
from ccproxy.api.ui.terminal_permission_handler import TerminalPermissionHandler
from ccproxy.cli.commands.permission_handler import (
    SSEConfirmationHandler,
    connect,
)
from ccproxy.config.settings import Settings


@pytest.fixture
def mock_terminal_handler() -> Mock:
    """Create a mock terminal handler."""
    handler = Mock(spec=TerminalPermissionHandler)
    handler.handle_permission = AsyncMock(return_value=True)
    handler.cancel_confirmation = Mock()
    return handler


@pytest.fixture
def mock_httpx_client() -> Mock:
    """Create a mock httpx client."""
    client = Mock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.stream = Mock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
async def sse_handler(
    mock_terminal_handler: Mock,
) -> AsyncGenerator[SSEConfirmationHandler, None]:
    """Create an SSE confirmation handler."""
    handler = SSEConfirmationHandler(
        api_url="http://localhost:8080",
        terminal_handler=mock_terminal_handler,
        ui=True,
    )
    yield handler


class TestSSEConfirmationHandler:
    """Test cases for SSE confirmation handler."""

    async def test_context_manager(
        self,
        sse_handler: SSEConfirmationHandler,
    ) -> None:
        """Test context manager creates and closes client."""
        async with sse_handler as handler:
            assert handler.client is not None
            assert isinstance(handler.client, httpx.AsyncClient)

        # Client should be None after exit
        assert sse_handler.client is None

    async def test_handle_ping_event(
        self,
        sse_handler: SSEConfirmationHandler,
    ) -> None:
        """Test that ping events are ignored."""
        # Should not raise any errors
        await sse_handler.handle_event(
            "ping",
            cast(
                dict[str, Any],
                {"type": "ping", "request_id": "", "message": "keepalive"},
            ),
        )

    async def test_handle_permission_request_event(
        self,
        sse_handler: SSEConfirmationHandler,
        mock_terminal_handler: Mock,
    ) -> None:
        """Test handling new confirmation request event."""
        event_data = {
            "type": "permission_request",
            "request_id": "test-id-123",
            "tool_name": "bash",
            "input": {"command": "ls -la"},
            "created_at": "2024-01-01T12:00:00",
            "expires_at": "2024-01-01T12:00:30",
        }

        await sse_handler.handle_event(
            "permission_request", cast(dict[str, Any], event_data)
        )

        # Should have created a task
        assert "test-id-123" in sse_handler._ongoing_requests
        task = sse_handler._ongoing_requests["test-id-123"]
        assert isinstance(task, asyncio.Task)

        # Wait for task to complete
        await asyncio.sleep(0.1)

        # Terminal handler should have been called
        mock_terminal_handler.handle_permission.assert_called_once()
        call_args = mock_terminal_handler.handle_permission.call_args[0][0]
        assert isinstance(call_args, PermissionRequest)
        assert call_args.id == "test-id-123"  # ID should be preserved now
        assert call_args.tool_name == "bash"

    async def test_handle_permission_resolved_event(
        self,
        sse_handler: SSEConfirmationHandler,
        mock_terminal_handler: Mock,
    ) -> None:
        """Test handling confirmation resolved by another handler."""
        # First create a pending request
        request_event = {
            "type": "permission_request",
            "request_id": "test-id-123",
            "tool_name": "bash",
            "input": {"command": "ls"},
            "created_at": "2024-01-01T12:00:00",
            "expires_at": "2024-01-01T12:00:30",
        }

        # Make terminal handler wait so we can cancel it
        wait_event = asyncio.Event()

        async def slow_handler(request: PermissionRequest) -> bool:
            await wait_event.wait()
            return True

        mock_terminal_handler.handle_permission = slow_handler

        await sse_handler.handle_event(
            "permission_request", cast(dict[str, Any], request_event)
        )

        # Ensure task is created
        assert "test-id-123" in sse_handler._ongoing_requests

        # Now send resolved event
        resolved_event = {
            "type": "permission_resolved",
            "request_id": "test-id-123",
            "allowed": True,
        }

        await sse_handler.handle_event(
            "permission_resolved", cast(dict[str, Any], resolved_event)
        )

        # Should have cancelled the confirmation
        mock_terminal_handler.cancel_confirmation.assert_called_once_with(
            "test-id-123", "approved by another handler"
        )

        # Task should be removed
        assert "test-id-123" not in sse_handler._ongoing_requests

        # Allow task to finish
        wait_event.set()

    async def test_already_resolved_request(
        self,
        sse_handler: SSEConfirmationHandler,
    ) -> None:
        """Test handling request that was already resolved."""
        # Mark request as already resolved
        sse_handler._resolved_requests["test-id-123"] = (True, "approved by another")

        event_data = {
            "type": "permission_request",
            "request_id": "test-id-123",
            "tool_name": "bash",
            "input": {"command": "ls"},
            "created_at": "2024-01-01T12:00:00",
            "expires_at": "2024-01-01T12:00:30",
        }

        await sse_handler.handle_event(
            "permission_request", cast(dict[str, Any], event_data)
        )

        # Should not create a task
        assert "test-id-123" not in sse_handler._ongoing_requests

    async def test_send_response_success(
        self,
        sse_handler: SSEConfirmationHandler,
        mock_httpx_client: Mock,
    ) -> None:
        """Test successfully sending a response."""
        sse_handler.client = mock_httpx_client

        mock_response = Mock()
        mock_response.status_code = 200
        mock_httpx_client.post.return_value = mock_response

        await sse_handler.send_response("test-id", True)

        mock_httpx_client.post.assert_called_once_with(
            "http://localhost:8080/permissions/test-id/respond",
            json={"allowed": True},
        )

    async def test_send_response_already_resolved(
        self,
        sse_handler: SSEConfirmationHandler,
        mock_httpx_client: Mock,
    ) -> None:
        """Test sending response when already resolved."""
        sse_handler.client = mock_httpx_client

        mock_response = Mock()
        mock_response.status_code = 409
        mock_httpx_client.post.return_value = mock_response

        # Should not raise error
        await sse_handler.send_response("test-id", False)

    async def test_send_response_error(
        self,
        sse_handler: SSEConfirmationHandler,
        mock_httpx_client: Mock,
    ) -> None:
        """Test handling errors when sending response."""
        sse_handler.client = mock_httpx_client

        mock_httpx_client.post.side_effect = httpx.ConnectError("Connection failed")

        # Should not raise error (logged internally)
        await sse_handler.send_response("test-id", True)

    async def test_parse_sse_stream(self, sse_handler: SSEConfirmationHandler) -> None:
        """Test parsing SSE stream data."""
        # Create mock response with SSE data
        sse_data = """event: ping
data: {"type": "ping", "request_id": "", "message": "Connected"}

event: permission_request
data: {"type": "permission_request", "request_id": "123", "tool_name": "bash"}

data: {"type": "message", "request_id": "", "message": "No event type"}

event: test
data: {"type": "test", "request_id": "test-123", "allowed": true, "message": "test event"}

"""

        async def mock_aiter_text() -> AsyncGenerator[str, None]:
            for chunk in sse_data.split("\n"):
                yield chunk + "\n"

        mock_response = Mock()
        mock_response.aiter_text = mock_aiter_text

        # Parse events
        events = []
        async for event_type, data in sse_handler.parse_sse_stream(mock_response):
            events.append((event_type, data))

        # Verify parsed events
        assert len(events) == 4

        assert events[0][0] == "ping"
        assert events[0][1]["message"] == "Connected"

        assert events[1][0] == "permission_request"
        assert events[1][1]["request_id"] == "123"

        assert events[2][0] == "message"  # Default type
        assert events[2][1]["message"] == "No event type"

        assert events[3][0] == "test"
        assert events[3][1]["allowed"] is True
        assert events[3][1]["message"] == "test event"

    async def test_parse_sse_stream_invalid_json(
        self,
        sse_handler: SSEConfirmationHandler,
    ) -> None:
        """Test handling invalid JSON in SSE stream."""
        sse_data = """event: test
data: {invalid json}

"""

        async def mock_aiter_text() -> AsyncGenerator[str, None]:
            yield sse_data

        mock_response = Mock()
        mock_response.aiter_text = mock_aiter_text

        # Should handle error gracefully
        events = []
        async for event_type, data in sse_handler.parse_sse_stream(mock_response):
            events.append((event_type, data))

        # No events should be yielded for invalid JSON
        assert len(events) == 0

    @patch("httpx.AsyncClient")
    async def test_run_with_successful_connection(
        self,
        mock_client_class: Mock,
        sse_handler: SSEConfirmationHandler,
    ) -> None:
        """Test running SSE client with successful connection."""
        # Create mock client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        # Mock SSE events with finite stream
        async def mock_parse_sse() -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
            yield "ping", {"message": "Connected"}
            # Simulate stream ending after one event

        # Mock client.stream
        mock_client.stream.return_value = mock_response

        sse_handler.client = mock_client
        sse_handler.max_retries = 0  # Don't retry to avoid infinite loop

        # Use patch to properly mock the method
        with (
            patch.object(
                sse_handler,
                "parse_sse_stream",
                new=AsyncMock(side_effect=mock_parse_sse),
            ),
            contextlib.suppress(TimeoutError),
        ):
            # Run with timeout to prevent hanging
            await asyncio.wait_for(sse_handler.run(), timeout=1.0)

        # Verify stream was called
        mock_client.stream.assert_called_once_with(
            "GET", "http://localhost:8080/permissions/stream"
        )

    @patch("httpx.AsyncClient")
    async def test_run_with_connection_retry(
        self,
        mock_client_class: Mock,
        sse_handler: SSEConfirmationHandler,
    ) -> None:
        """Test connection retry on failure."""
        # Create mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # First attempt fails, second succeeds
        connect_error = httpx.ConnectError("Connection failed")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        # First call raises error, second returns response
        mock_client.stream.side_effect = [connect_error, mock_response]

        # Mock SSE parsing with finite stream
        async def mock_parse_sse() -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
            yield "ping", {"message": "Connected"}
            # Stream ends after one event

        sse_handler.client = mock_client
        sse_handler.max_retries = 1  # Allow one retry

        # Use patch to properly mock the method
        with (
            patch.object(
                sse_handler,
                "parse_sse_stream",
                new=AsyncMock(side_effect=mock_parse_sse),
            ),
            contextlib.suppress(TimeoutError),
        ):
            # Should retry and succeed with timeout to prevent hanging
            await asyncio.wait_for(sse_handler.run(), timeout=2.0)

        # Should have been called twice (first fails, second succeeds)
        assert mock_client.stream.call_count == 2

    async def test_handle_permission_with_cancellation(
        self,
        sse_handler: SSEConfirmationHandler,
        mock_terminal_handler: Mock,
    ) -> None:
        """Test handling confirmation that gets cancelled."""
        # Create a slow confirmation handler
        wait_event = asyncio.Event()

        async def slow_handler(request: PermissionRequest) -> bool:
            await wait_event.wait()
            return True

        mock_terminal_handler.handle_permission = slow_handler

        from datetime import datetime, timedelta

        now = datetime.utcnow()
        request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            expires_at=now + timedelta(seconds=30),
        )

        # Start handling in background
        task = asyncio.create_task(
            sse_handler._handle_permission_with_cancellation(request)
        )

        # Cancel after a short delay
        await asyncio.sleep(0.1)
        task.cancel()

        # Should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await task

        # Clean up
        wait_event.set()


class TestCLICommand:
    """Test the CLI command function."""

    @patch("ccproxy.cli.commands.permission_handler.get_settings")
    @patch("ccproxy.cli.commands.permission_handler.asyncio.run")
    def test_connect_command_default_url(
        self,
        mock_asyncio_run: Mock,
        mock_get_settings: Mock,
    ) -> None:
        """Test connect command with default URL from settings."""
        # Mock settings
        mock_settings = Mock(spec=Settings)
        mock_settings.server = Mock()
        mock_settings.server.host = "localhost"
        mock_settings.server.port = 8080
        mock_get_settings.return_value = mock_settings

        # Call command
        connect(api_url=None, no_ui=False)

        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()

    @patch("ccproxy.cli.commands.permission_handler.get_settings")
    @patch("ccproxy.cli.commands.permission_handler.asyncio.run")
    def test_connect_command_custom_url(
        self,
        mock_asyncio_run: Mock,
        mock_get_settings: Mock,
    ) -> None:
        """Test connect command with custom URL."""
        # Call command with custom URL
        connect(api_url="http://custom:9090", no_ui=True)

        # Settings should still be called (for other configs)
        mock_get_settings.assert_called_once()

        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()

    @patch("ccproxy.cli.commands.permission_handler.get_settings")
    @patch("ccproxy.cli.commands.permission_handler.asyncio.run")
    def test_connect_command_keyboard_interrupt(
        self,
        mock_asyncio_run: Mock,
        mock_get_settings: Mock,
    ) -> None:
        """Test handling KeyboardInterrupt."""
        # Mock settings
        mock_settings = Mock(spec=Settings)
        mock_settings.server = Mock()
        mock_settings.server.host = "localhost"
        mock_settings.server.port = 8080
        mock_get_settings.return_value = mock_settings

        # Make asyncio.run raise KeyboardInterrupt
        mock_asyncio_run.side_effect = KeyboardInterrupt()

        # Should not raise error
        connect(api_url=None, no_ui=False)

    @patch("ccproxy.cli.commands.permission_handler.get_settings")
    @patch("ccproxy.cli.commands.permission_handler.asyncio.run")
    @patch("ccproxy.cli.commands.permission_handler.logger")
    def test_connect_command_general_error(
        self,
        mock_logger: Mock,
        mock_asyncio_run: Mock,
        mock_get_settings: Mock,
    ) -> None:
        """Test handling general errors."""
        # Mock settings
        mock_settings = Mock(spec=Settings)
        mock_settings.server = Mock()
        mock_settings.server.host = "localhost"
        mock_settings.server.port = 8080
        mock_settings.security = Mock()
        mock_settings.security.auth_token = None
        mock_get_settings.return_value = mock_settings

        # Make asyncio.run raise an error
        mock_asyncio_run.side_effect = Exception("Test error")

        # Should raise typer.Exit
        with pytest.raises(typer.Exit) as exc_info:
            connect(api_url=None, no_ui=False)

        assert exc_info.value.exit_code == 1
