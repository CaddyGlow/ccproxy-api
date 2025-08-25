"""Tests for terminal permission handler."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from plugins.permissions.handlers.terminal import TerminalPermissionHandler
from plugins.permissions.models import PermissionRequest


@pytest.fixture
def terminal_handler() -> TerminalPermissionHandler:
    """Create a terminal handler for testing."""
    return TerminalPermissionHandler()


@pytest.fixture
def sample_request() -> PermissionRequest:
    """Create a sample permission request."""
    return PermissionRequest(
        tool_name="bash",
        input={"command": "ls -la", "cwd": "/home/user"},
        expires_at=datetime.utcnow() + timedelta(seconds=30),
    )


class TestTerminalPermissionHandler:
    """Test cases for terminal permission handler."""

    async def test_handle_permission_timeout(
        self,
        terminal_handler: TerminalPermissionHandler,
    ) -> None:
        """Test handling permission timeout."""
        # Create a request that expires immediately
        request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            expires_at=datetime.utcnow() - timedelta(seconds=1),
        )

        # Should return False on timeout
        result = await terminal_handler.handle_permission(request)
        assert result is False

    @patch("ccproxy.api.ui.terminal_permission_handler.ConfirmationApp")
    async def test_handle_permission_allowed(
        self,
        mock_app_class: Mock,
        terminal_handler: TerminalPermissionHandler,
        sample_request: PermissionRequest,
    ) -> None:
        """Test handling permission that gets allowed."""
        # Mock the app instance and its run_async method
        mock_app = Mock()
        mock_app.run_async = AsyncMock(return_value=True)
        mock_app_class.return_value = mock_app

        # Handle permission
        result = await terminal_handler.handle_permission(sample_request)

        # Verify result
        assert result is True
        assert mock_app_class.called
        assert mock_app.run_async.called

    @patch("ccproxy.api.ui.terminal_permission_handler.ConfirmationApp")
    async def test_handle_permission_denied(
        self,
        mock_app_class: Mock,
        terminal_handler: TerminalPermissionHandler,
        sample_request: PermissionRequest,
    ) -> None:
        """Test handling permission that gets denied."""
        # Mock the app instance and its run_async method
        mock_app = Mock()
        mock_app.run_async = AsyncMock(return_value=False)
        mock_app_class.return_value = mock_app

        # Handle permission
        result = await terminal_handler.handle_permission(sample_request)

        # Verify result
        assert result is False
        assert mock_app_class.called
        assert mock_app.run_async.called

    @patch("ccproxy.api.ui.terminal_permission_handler.ConfirmationApp")
    async def test_handle_permission_keyboard_interrupt(
        self,
        mock_app_class: Mock,
        terminal_handler: TerminalPermissionHandler,
        sample_request: PermissionRequest,
    ) -> None:
        """Test handling permission with keyboard interrupt."""
        # Mock the app instance to raise KeyboardInterrupt
        mock_app = Mock()
        mock_app.run_async = AsyncMock(side_effect=KeyboardInterrupt("User cancelled"))
        mock_app_class.return_value = mock_app

        # Handle permission - should raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            await terminal_handler.handle_permission(sample_request)

    @patch("ccproxy.api.ui.terminal_permission_handler.ConfirmationApp")
    async def test_handle_permission_error_handling(
        self,
        mock_app_class: Mock,
        terminal_handler: TerminalPermissionHandler,
        sample_request: PermissionRequest,
    ) -> None:
        """Test error handling during permission."""
        # Mock the app instance to raise an exception
        mock_app = Mock()
        mock_app.run_async = AsyncMock(side_effect=Exception("Test error"))
        mock_app_class.return_value = mock_app

        # Handle permission - should return False on error
        result = await terminal_handler.handle_permission(sample_request)
        assert result is False

    def test_cancel_confirmation(
        self,
        terminal_handler: TerminalPermissionHandler,
    ) -> None:
        """Test cancelling a confirmation."""
        request_id = "test-id-12345"

        # Cancel confirmation - should not raise
        terminal_handler.cancel_confirmation(request_id, "test reason")

        # Verify the request is marked as cancelled
        assert request_id in terminal_handler._cancelled_requests

    async def test_handle_permission_with_cancellation(
        self,
        terminal_handler: TerminalPermissionHandler,
        sample_request: PermissionRequest,
    ) -> None:
        """Test handling permission that gets cancelled."""
        # Cancel the request before handling
        terminal_handler.cancel_confirmation(sample_request.id, "test cancel")

        with patch(
            "ccproxy.api.ui.terminal_permission_handler.ConfirmationApp"
        ) as mock_app_class:
            mock_app = Mock()
            mock_app.run_async = AsyncMock(return_value=True)
            mock_app_class.return_value = mock_app

            # Handle permission - should return False due to cancellation
            result = await terminal_handler.handle_permission(sample_request)
            assert result is False

    async def test_shutdown(
        self,
        terminal_handler: TerminalPermissionHandler,
    ) -> None:
        """Test shutting down the handler."""
        # Create a mock processing task
        terminal_handler._processing_task = Mock()
        terminal_handler._processing_task.done.return_value = False
        terminal_handler._processing_task.cancel = Mock()

        # Create a future that completes when awaited
        future: asyncio.Future[None] = asyncio.Future()
        future.set_exception(asyncio.CancelledError())
        terminal_handler._processing_task = asyncio.create_task(asyncio.sleep(0))
        terminal_handler._processing_task.cancel()

        # Shutdown should not raise
        await terminal_handler.shutdown()

        # Processing task should be None after shutdown
        assert terminal_handler._processing_task is None

    async def test_queue_processing(
        self,
        terminal_handler: TerminalPermissionHandler,
        sample_request: PermissionRequest,
    ) -> None:
        """Test that requests are queued and processed."""
        with patch(
            "ccproxy.api.ui.terminal_permission_handler.ConfirmationApp"
        ) as mock_app_class:
            mock_app = Mock()
            mock_app.run_async = AsyncMock(return_value=True)
            mock_app_class.return_value = mock_app

            # Start processing task
            terminal_handler._ensure_processing_task_running()
            assert terminal_handler._processing_task is not None
            assert not terminal_handler._processing_task.done()

            # Handle permission - should use queue
            result = await terminal_handler.handle_permission(sample_request)
            assert result is True

            # Cleanup
            await terminal_handler.shutdown()
