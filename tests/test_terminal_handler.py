"""Tests for terminal confirmation handler."""

import asyncio
from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from rich.console import Console

from ccproxy.api.ui.terminal_confirmation_handler import TerminalConfirmationHandler
from ccproxy.models.confirmations import ConfirmationRequest


@pytest.fixture
def mock_console() -> Mock:
    """Create a mock console following TESTING.md spec."""
    return Mock(spec=Console)


@pytest.fixture
def terminal_handler(mock_console: Mock) -> TerminalConfirmationHandler:
    """Create a terminal handler with mocked console following TESTING.md spec."""
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
    """Test cases for terminal confirmation handler following TESTING.md spec."""

    def test_format_value_truncates_long_strings(
        self,
        terminal_handler: TerminalConfirmationHandler,
    ) -> None:
        """Test value formatting truncates long strings correctly."""
        # Short value
        assert terminal_handler._format_value("short") == "short"

        # Long value truncation
        long_value = "x" * 100
        formatted = terminal_handler._format_value(long_value)
        assert len(formatted) == 60
        assert formatted.endswith("...")

    def test_display_request_shows_permission_panel(
        self,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
        mock_console: Mock,
    ) -> None:
        """Test displaying confirmation request shows permission panel."""
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

    def test_display_result_shows_allowed_denied_status(
        self,
        terminal_handler: TerminalConfirmationHandler,
        mock_console: Mock,
    ) -> None:
        """Test displaying confirmation result shows allowed/denied status."""
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
        "ccproxy.api.ui.terminal_confirmation_handler.TerminalConfirmationHandler._get_single_key"
    )
    async def test_handle_confirmation_allowed(
        self,
        mock_get_key: Mock,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
        mock_console: Mock,
    ) -> None:
        """Test handling confirmation that gets allowed."""
        # Setup mock to return 'y' for allowed
        mock_get_key.return_value = "y"

        # Mock the Live context manager
        with patch("ccproxy.api.ui.terminal_confirmation_handler.Live") as mock_live:
            mock_live_instance = Mock()
            mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
            mock_live_instance.__exit__ = Mock(return_value=None)
            mock_live.return_value = mock_live_instance

            # Handle confirmation
            result = await terminal_handler.handle_confirmation(sample_request)

        # Verify result
        assert result is True

    @patch(
        "ccproxy.api.ui.terminal_confirmation_handler.TerminalConfirmationHandler._get_single_key"
    )
    async def test_handle_confirmation_denied(
        self,
        mock_get_key: Mock,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
        mock_console: Mock,
    ) -> None:
        """Test handling confirmation that gets denied."""
        # Setup mock to return 'n' for denied
        mock_get_key.return_value = "n"

        # Mock the Live context manager
        with patch("ccproxy.api.ui.terminal_confirmation_handler.Live") as mock_live:
            mock_live_instance = Mock()
            mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
            mock_live_instance.__exit__ = Mock(return_value=None)
            mock_live.return_value = mock_live_instance

            # Handle confirmation
            result = await terminal_handler.handle_confirmation(sample_request)

        # Verify result
        assert result is False

    @patch(
        "ccproxy.api.ui.terminal_confirmation_handler.TerminalConfirmationHandler._get_single_key"
    )
    async def test_handle_confirmation_with_executor(
        self,
        mock_get_key: Mock,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
    ) -> None:
        """Test that confirmation uses key input handling."""
        # Setup mock to return 'y' for allowed
        mock_get_key.return_value = "y"

        # Mock the Live context manager
        with patch("ccproxy.api.ui.terminal_confirmation_handler.Live") as mock_live:
            mock_live_instance = Mock()
            mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
            mock_live_instance.__exit__ = Mock(return_value=None)
            mock_live.return_value = mock_live_instance

            # Handle confirmation
            result = await terminal_handler.handle_confirmation(sample_request)

        # Verify key input was called
        assert mock_get_key.called
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

        # Mock the Live context manager
        with patch("ccproxy.api.ui.terminal_confirmation_handler.Live") as mock_live:
            mock_live_instance = Mock()
            mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
            mock_live_instance.__exit__ = Mock(return_value=None)
            mock_live.return_value = mock_live_instance

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

        # Mock the Live context manager and _get_single_key
        with patch("ccproxy.api.ui.terminal_confirmation_handler.Live") as mock_live:
            mock_live_instance = Mock()
            mock_live_instance.__enter__ = Mock(return_value=mock_live_instance)
            mock_live_instance.__exit__ = Mock(return_value=None)
            mock_live.return_value = mock_live_instance

            with (
                patch.object(terminal_handler, "_get_single_key", return_value=None),
                pytest.raises(asyncio.CancelledError),
            ):
                # Handle confirmation - should be cancelled
                await terminal_handler.handle_confirmation(sample_request)

        await cancel_task

    async def test_handle_confirmation_error_handling(
        self,
        terminal_handler: TerminalConfirmationHandler,
        sample_request: ConfirmationRequest,
        mock_console: Mock,
    ) -> None:
        """Test error handling during confirmation."""
        # Mock Live to raise an exception
        with patch("ccproxy.api.ui.terminal_confirmation_handler.Live") as mock_live:
            mock_live.side_effect = Exception("Test error")

            # Handle confirmation - should return False on error
            result = await terminal_handler.handle_confirmation(sample_request)
            assert result is False

            # Verify error was displayed
            print_calls = str(mock_console.print.call_args_list)
            assert "Error handling confirmation" in print_calls
