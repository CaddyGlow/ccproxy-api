"""Tests for terminal confirmation handler."""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from rich.console import Console
from rich.table import Table

from ccproxy.api.ui.terminal_confirmation_handler import TerminalConfirmationHandler
from ccproxy.models.confirmations import ConfirmationRequest


@pytest.fixture
def mock_console() -> Mock:
    """Create a mock console."""
    console = Mock(spec=Console)
    return console


@pytest.fixture
def terminal_handler(mock_console: Mock) -> TerminalConfirmationHandler:
    """Create a terminal handler with mocked console."""
    handler = TerminalConfirmationHandler()
    handler._console = mock_console
    return handler


@pytest.fixture
def sample_request() -> ConfirmationRequest:
    """Create a sample confirmation request."""
    return ConfirmationRequest(
        tool_name="bash",
        input={"command": "ls -la", "cwd": "/home/user"},
        expires_at=datetime.utcnow() + timedelta(seconds=30),
    )


class TestTerminalConfirmationHandler:
    """Test cases for terminal confirmation handler."""

    def test_format_value(
        self,
        terminal_handler: TerminalConfirmationHandler,
    ) -> None:
        """Test value formatting."""
        # Short value
        assert terminal_handler._format_value("short") == "short"

        # Long value truncation
        long_value = "x" * 100
        formatted = terminal_handler._format_value(long_value)
        assert len(formatted) == 60
        assert formatted.endswith("...")

    def test_display_request(
        self,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
        mock_console: Mock,
    ) -> None:
        """Test displaying a confirmation request."""
        terminal_handler._display_request(sample_request)

        # Verify console was used
        assert mock_console.print.called

        # Check that a Panel was printed with the correct title
        calls = mock_console.print.call_args_list
        panel_call = None
        for call in calls:
            if (
                len(call[0]) > 0
                and hasattr(call[0][0], "__class__")
                and call[0][0].__class__.__name__ == "Panel"
            ):
                panel_call = call[0][0]
                break

        assert panel_call is not None, "No Panel was printed"
        assert hasattr(panel_call, "title"), "Panel has no title"
        assert "Permission Request" in panel_call.title

    def test_display_result(
        self,
        terminal_handler: TerminalConfirmationHandler,
        mock_console: Mock,
    ) -> None:
        """Test displaying confirmation results."""
        # Test allowed
        terminal_handler._display_result(True)
        print_call = str(mock_console.print.call_args)
        assert "allowed" in print_call.lower()
        assert "green" in print_call

        # Reset mock
        mock_console.reset_mock()

        # Test denied
        terminal_handler._display_result(False)
        print_call = str(mock_console.print.call_args)
        assert "denied" in print_call.lower()
        assert "red" in print_call

    @patch(
        "ccproxy.api.ui.terminal_confirmation_handler.TerminalConfirmationHandler._get_async_input"
    )
    async def test_handle_confirmation_allowed(
        self,
        mock_get_input: AsyncMock,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
        mock_console: Mock,
    ) -> None:
        """Test handling confirmation that gets allowed."""
        # Setup mock to return 'y' for allowed
        mock_get_input.return_value = "y"

        # Handle confirmation
        result = await terminal_handler.handle_confirmation(sample_request)

        # Verify result
        assert result is True

        # Verify console interactions
        assert mock_console.print.call_count >= 2  # Request display + result

    @patch(
        "ccproxy.api.ui.terminal_confirmation_handler.TerminalConfirmationHandler._get_async_input"
    )
    async def test_handle_confirmation_denied(
        self,
        mock_get_input: AsyncMock,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
        mock_console: Mock,
    ) -> None:
        """Test handling confirmation that gets denied."""
        # Setup mock to return 'n' for denied
        mock_get_input.return_value = "n"

        # Handle confirmation
        result = await terminal_handler.handle_confirmation(sample_request)

        # Verify result
        assert result is False

        # Verify result message
        last_print_call = mock_console.print.call_args_list[-1]
        assert "denied" in str(last_print_call).lower()

    @patch(
        "ccproxy.api.ui.terminal_confirmation_handler.TerminalConfirmationHandler._get_async_input"
    )
    async def test_handle_confirmation_with_executor(
        self,
        mock_get_input: AsyncMock,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
    ) -> None:
        """Test that confirmation uses async input handling."""
        # Setup mock to return 'y' for allowed
        mock_get_input.return_value = "y"

        # Handle confirmation
        result = await terminal_handler.handle_confirmation(sample_request)

        # Verify async input was called
        mock_get_input.assert_called_once()
        assert result is True

    async def test_handle_confirmation_timeout(
        self,
        terminal_handler: TerminalConfirmationHandler,
    ) -> None:
        """Test handling confirmation timeout."""
        # Create a request that expires immediately
        request = ConfirmationRequest(
            tool_name="bash",
            input={"command": "test"},
            expires_at=datetime.utcnow() - timedelta(seconds=1),
        )

        # Should return False on timeout
        result = await terminal_handler.handle_confirmation(request)
        assert result is False

    def test_cancel_confirmation(
        self,
        terminal_handler: TerminalConfirmationHandler,
    ) -> None:
        """Test cancelling a confirmation."""
        request_id = "test-id-12345"

        # Create a cancellation event
        event = asyncio.Event()
        terminal_handler._cancellation_events[request_id] = event

        # Cancel confirmation
        terminal_handler.cancel_confirmation(request_id, "test reason")

        # Verify event was set
        assert event.is_set()

        # Cancel non-existent - should not raise
        terminal_handler.cancel_confirmation("non-existent", "test")

    async def test_handle_confirmation_with_cancellation(
        self,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
    ) -> None:
        """Test handling confirmation that gets cancelled."""

        # Create a task that will cancel the confirmation
        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            terminal_handler.cancel_confirmation(sample_request.id, "test cancel")

        cancel_task = asyncio.create_task(cancel_after_delay())

        # Patch the executor to delay so cancellation can happen
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_future: asyncio.Future[bool] = asyncio.Future()

            async def delayed_set_result():
                await asyncio.sleep(0.2)
                mock_future.set_result(True)

            asyncio.create_task(delayed_set_result())
            mock_loop.run_in_executor.return_value = mock_future
            mock_get_loop.return_value = mock_loop

            # Handle confirmation - should be cancelled
            with pytest.raises(asyncio.CancelledError):
                await terminal_handler.handle_confirmation(sample_request)

        await cancel_task

    @patch(
        "ccproxy.api.ui.terminal_confirmation_handler.TerminalConfirmationHandler._get_async_input"
    )
    async def test_handle_confirmation_error_handling(
        self,
        mock_get_input: AsyncMock,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
        mock_console: Mock,
    ) -> None:
        """Test error handling during confirmation."""
        # Setup mock to raise exception
        mock_get_input.side_effect = Exception("Test error")

        # Handle confirmation - should return False on error
        result = await terminal_handler.handle_confirmation(sample_request)
        assert result is False

        # Verify error was displayed
        print_calls = str(mock_console.print.call_args_list)
        assert "Error handling confirmation" in print_calls
