"""Tests for startup utility functions.

This module tests the startup helper functions extracted from the lifespan function:
- Authentication validation
- Claude CLI checking
- Service initialization (detection, SDK, scheduler, storage, permissions)
- Graceful degradation and error handling
- Component lifecycle management

All tests use mocks to avoid external dependencies and test in isolation.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI

from ccproxy.config.settings import Settings
from ccproxy.scheduler.errors import SchedulerError
from ccproxy.utils.startup_helpers import (
    check_claude_cli_startup,
    setup_scheduler_shutdown,
    setup_scheduler_startup,
    setup_session_manager_shutdown,
)


# TestValidateAuthenticationStartup removed - authentication now handled by plugins
# class TestValidateAuthenticationStartup:
#     """Test authentication validation during startup."""
#
#     @pytest.fixture
#     def mock_app(self) -> FastAPI:
#         """Create a mock FastAPI app."""
#         return FastAPI()
#
#     @pytest.fixture
#     def mock_settings(self) -> Mock:
#         """Create mock settings."""
#         settings = Mock(spec=Settings)
#         # Configure nested attributes properly
#         settings.auth = Mock()
#         settings.auth.storage = Mock()
#         settings.auth.storage.storage_paths = ["/path1", "/path2"]
#         return settings
#
#     @pytest.fixture
#     def mock_credentials_manager(self) -> Mock:
#         """Create mock credentials manager."""
#         return AsyncMock()

# All test methods commented out - authentication now handled by plugins
#
#     async def test_valid_authentication_with_oauth_token(
#         self, mock_app: FastAPI, mock_settings: Mock
#     ) -> None:
#         pass
#
#     async def test_valid_authentication_without_oauth_token(
#         self, mock_app: FastAPI, mock_settings: Mock
#     ) -> None:
#         pass
#
#     async def test_expired_authentication(
#         self, mock_app: FastAPI, mock_settings: Mock
#     ) -> None:
#         pass
#
#     async def test_invalid_authentication(
#         self, mock_app: FastAPI, mock_settings: Mock
#     ) -> None:
#         pass
#
#     async def test_credentials_not_found(
#         self, mock_app: FastAPI, mock_settings: Mock
#     ) -> None:
#         pass
#
#     async def test_authentication_validation_error(
#         self, mock_app: FastAPI, mock_settings: Mock
#     ) -> None:
#         pass


class TestCheckClaudeCLIStartup:
    """Test Claude CLI checking during startup."""

    @pytest.fixture
    def mock_app(self) -> FastAPI:
        """Create a mock FastAPI app."""
        return FastAPI()

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        return Mock(spec=Settings)

    async def test_claude_cli_available(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test successful Claude CLI detection."""
        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await check_claude_cli_startup(mock_app, mock_settings)

            # The function now just passes (handled by plugin)
            # Verify no logs were called
            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()

    async def test_claude_cli_unavailable(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test handling when Claude CLI is unavailable."""
        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await check_claude_cli_startup(mock_app, mock_settings)

            # The function now just passes (handled by plugin)
            # Verify no logs were called
            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()

    async def test_claude_cli_check_error(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test handling of errors during Claude CLI check."""
        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await check_claude_cli_startup(mock_app, mock_settings)

            # The function now just passes (handled by plugin)
            # Verify no logs were called
            mock_logger.info.assert_not_called()
            mock_logger.warning.assert_not_called()
            mock_logger.error.assert_not_called()


# Removed old log storage lifecycle tests (migrated to duckdb_storage plugin)


class TestSchedulerLifecycle:
    """Test scheduler startup and shutdown."""

    @pytest.fixture
    def mock_app(self) -> FastAPI:
        """Create a mock FastAPI app."""
        app = FastAPI()
        app.state = Mock()
        app.state.service_container = Mock()
        return app

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        return Mock(spec=Settings)

    async def test_scheduler_startup_success(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test successful scheduler startup."""
        # Ensure no session_manager exists to avoid task addition
        if hasattr(mock_app.state, "session_manager"):
            delattr(mock_app.state, "session_manager")

        with patch("ccproxy.utils.startup_helpers.start_scheduler") as mock_start:
            mock_scheduler = AsyncMock()
            mock_start.return_value = mock_scheduler

            with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
                await setup_scheduler_startup(mock_app, mock_settings)

                # Verify scheduler was started with container and stored
                mock_start.assert_called_once_with(
                    mock_settings, mock_app.state.service_container
                )
                assert mock_app.state.scheduler == mock_scheduler

                # Verify debug log was called
                mock_logger.debug.assert_called_with("scheduler_initialized")

    async def test_scheduler_startup_with_session_manager(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test scheduler startup with session manager for task addition."""
        mock_session_manager = AsyncMock()
        mock_app.state.session_manager = mock_session_manager

        with patch("ccproxy.utils.startup_helpers.start_scheduler") as mock_start:
            mock_scheduler = AsyncMock()
            mock_start.return_value = mock_scheduler

            with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
                await setup_scheduler_startup(mock_app, mock_settings)

                # Verify task was added to scheduler
                mock_scheduler.add_task.assert_called_once()
                task_args = mock_scheduler.add_task.call_args[1]
                assert task_args["task_name"] == "session_pool_stats"
                assert task_args["task_type"] == "pool_stats"
                assert task_args["interval_seconds"] == 60
                assert task_args["pool_manager"] == mock_session_manager

    async def test_scheduler_startup_error(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test error handling during scheduler startup."""
        with patch("ccproxy.utils.startup_helpers.start_scheduler") as mock_start:
            mock_start.side_effect = SchedulerError("Scheduler start failed")

            with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
                await setup_scheduler_startup(mock_app, mock_settings)

                # Verify error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args[1]
                assert (
                    "scheduler_initialization_failed" in mock_logger.error.call_args[0]
                )
                assert call_args["error"] == "Scheduler start failed"

    async def test_scheduler_shutdown_success(self, mock_app: FastAPI) -> None:
        """Test successful scheduler shutdown."""
        mock_scheduler = AsyncMock()
        mock_app.state.scheduler = mock_scheduler

        with (
            patch("ccproxy.utils.startup_helpers.stop_scheduler") as mock_stop,
            patch("ccproxy.utils.startup_helpers.logger") as mock_logger,
        ):
            await setup_scheduler_shutdown(mock_app)

            # Verify scheduler was stopped
            mock_stop.assert_called_once_with(mock_scheduler)

            # Verify debug log was called
            mock_logger.debug.assert_called_once_with("scheduler_stopped_lifespan")

    async def test_scheduler_shutdown_error(self, mock_app: FastAPI) -> None:
        """Test error handling during scheduler shutdown."""
        mock_scheduler = AsyncMock()
        mock_app.state.scheduler = mock_scheduler

        with patch("ccproxy.utils.startup_helpers.stop_scheduler") as mock_stop:
            mock_stop.side_effect = SchedulerError("Stop failed")

            with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
                await setup_scheduler_shutdown(mock_app)

                # Verify error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args[1]
                assert "scheduler_stop_failed" in mock_logger.error.call_args[0]
                assert call_args["error"] == "Stop failed"


class TestSessionManagerShutdown:
    """Test session manager shutdown."""

    @pytest.fixture
    def mock_app(self) -> FastAPI:
        """Create a mock FastAPI app."""
        app = FastAPI()
        app.state = Mock()
        return app

    async def test_session_manager_shutdown_success(self, mock_app: FastAPI) -> None:
        """Test successful session manager shutdown."""
        mock_session_manager = AsyncMock()
        mock_app.state.session_manager = mock_session_manager

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await setup_session_manager_shutdown(mock_app)

            # Verify session manager was shut down
            mock_session_manager.shutdown.assert_called_once()

            # Verify debug log was called
            mock_logger.debug.assert_called_once_with(
                "claude_sdk_session_manager_shutdown"
            )

    async def test_session_manager_shutdown_no_manager(self, mock_app: FastAPI) -> None:
        """Test shutdown when no session manager exists."""
        # Ensure no session_manager attribute exists
        if hasattr(mock_app.state, "session_manager"):
            delattr(mock_app.state, "session_manager")

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await setup_session_manager_shutdown(mock_app)

            # Verify no logs were called
            mock_logger.debug.assert_not_called()
            mock_logger.error.assert_not_called()

    async def test_session_manager_shutdown_error(self, mock_app: FastAPI) -> None:
        """Test error handling during session manager shutdown."""
        mock_session_manager = AsyncMock()
        mock_session_manager.shutdown.side_effect = Exception("Shutdown failed")
        mock_app.state.session_manager = mock_session_manager

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await setup_session_manager_shutdown(mock_app)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[1]
            assert (
                "claude_sdk_session_manager_shutdown_unexpected_error"
                in mock_logger.error.call_args[0]
            )
            assert call_args["error"] == "Shutdown failed"
            assert call_args["exc_info"] is not None


class TestFlushStreamingBatchesShutdown:
    """Test streaming batches flushing during shutdown."""

    @pytest.fixture
    def mock_app(self) -> FastAPI:
        """Create a mock FastAPI app."""
        return FastAPI()


class TestClaudeDetectionStartup:
    """Test Claude detection service initialization."""

    @pytest.fixture
    def mock_app(self) -> FastAPI:
        """Create a mock FastAPI app."""
        app = FastAPI()
        app.state = Mock()
        return app

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        return Mock(spec=Settings)
