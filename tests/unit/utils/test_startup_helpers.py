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
    flush_streaming_batches_shutdown,
    initialize_log_storage_shutdown,
    initialize_log_storage_startup,
    initialize_permission_service_startup,
    setup_permission_service_shutdown,
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


class TestLogStorageLifecycle:
    """Test log storage initialization and shutdown."""

    @pytest.fixture
    def mock_app(self) -> FastAPI:
        """Create a mock FastAPI app."""
        app = FastAPI()
        app.state = Mock()
        return app

    @pytest.fixture
    def mock_settings(self) -> Mock:
        """Create mock settings."""
        settings = Mock(spec=Settings)
        # Configure nested attributes properly
        settings.observability = Mock()
        settings.observability.needs_storage_backend = True
        settings.observability.log_storage_backend = "duckdb"
        settings.observability.duckdb_path = "/tmp/test.db"
        settings.observability.logs_collection_enabled = True
        return settings

    async def test_log_storage_startup_success(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test successful log storage initialization."""
        with patch("ccproxy.utils.startup_helpers.SimpleDuckDBStorage") as MockStorage:
            mock_storage = AsyncMock()
            MockStorage.return_value = mock_storage

            with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
                await initialize_log_storage_startup(mock_app, mock_settings)

                # Verify storage was created and initialized
                MockStorage.assert_called_once_with(database_path="/tmp/test.db")
                mock_storage.initialize.assert_called_once()

                # Verify storage was stored in app state
                assert mock_app.state.log_storage == mock_storage

                # Verify debug log was called
                mock_logger.debug.assert_called_once()
                call_args = mock_logger.debug.call_args[1]
                assert "log_storage_initialized" in mock_logger.debug.call_args[0]
                assert call_args["backend"] == "duckdb"

    async def test_log_storage_startup_not_needed(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test when log storage is not needed."""
        mock_settings.observability.needs_storage_backend = False

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await initialize_log_storage_startup(mock_app, mock_settings)

            # Verify no logs were called (function returns early)
            mock_logger.debug.assert_not_called()
            mock_logger.error.assert_not_called()

    async def test_log_storage_startup_error(
        self, mock_app: FastAPI, mock_settings: Mock
    ) -> None:
        """Test error handling during log storage initialization."""
        with patch("ccproxy.utils.startup_helpers.SimpleDuckDBStorage") as MockStorage:
            mock_storage = AsyncMock()
            mock_storage.initialize.side_effect = Exception("Storage init failed")
            MockStorage.return_value = mock_storage

            with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
                await initialize_log_storage_startup(mock_app, mock_settings)

                # Verify error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args[1]
                assert (
                    "log_storage_initialization_unexpected_error"
                    in mock_logger.error.call_args[0]
                )
                assert call_args["error"] == "Storage init failed"
                assert call_args["exc_info"] is not None

    async def test_log_storage_shutdown_success(self, mock_app: FastAPI) -> None:
        """Test successful log storage shutdown."""
        mock_storage = AsyncMock()
        mock_app.state.log_storage = mock_storage

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await initialize_log_storage_shutdown(mock_app)

            # Verify storage was closed
            mock_storage.close.assert_called_once()

            # Verify debug log was called
            mock_logger.debug.assert_called_once_with("log_storage_closed")

    async def test_log_storage_shutdown_no_storage(self, mock_app: FastAPI) -> None:
        """Test shutdown when no log storage exists."""
        # Ensure no log_storage attribute exists
        if hasattr(mock_app.state, "log_storage"):
            delattr(mock_app.state, "log_storage")

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await initialize_log_storage_shutdown(mock_app)

            # Verify no logs were called
            mock_logger.debug.assert_not_called()
            mock_logger.error.assert_not_called()

    async def test_log_storage_shutdown_error(self, mock_app: FastAPI) -> None:
        """Test error handling during log storage shutdown."""
        mock_storage = AsyncMock()
        mock_storage.close.side_effect = Exception("Close failed")
        mock_app.state.log_storage = mock_storage

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await initialize_log_storage_shutdown(mock_app)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[1]
            assert (
                "log_storage_close_unexpected_error" in mock_logger.error.call_args[0]
            )
            assert call_args["error"] == "Close failed"
            assert call_args["exc_info"] is not None


class TestSchedulerLifecycle:
    """Test scheduler startup and shutdown."""

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

                # Verify scheduler was started and stored
                mock_start.assert_called_once_with(mock_settings)
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

    async def test_flush_streaming_batches_success(self, mock_app: FastAPI) -> None:
        """Test successful streaming batches flushing."""
        with patch(
            "ccproxy.utils.simple_request_logger.flush_all_streaming_batches"
        ) as mock_flush:
            mock_flush.return_value = None  # Async function returns None

            with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
                await flush_streaming_batches_shutdown(mock_app)

                # Verify flush function was called
                mock_flush.assert_called_once()

                # Verify debug log was called
                mock_logger.debug.assert_called_once_with("streaming_batches_flushed")

    async def test_flush_streaming_batches_error(self, mock_app: FastAPI) -> None:
        """Test error handling during streaming batches flushing."""
        with patch(
            "ccproxy.utils.simple_request_logger.flush_all_streaming_batches"
        ) as mock_flush:
            mock_flush.side_effect = Exception("Flush failed")

            with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
                await flush_streaming_batches_shutdown(mock_app)

                # Verify error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args[1]
                assert (
                    "streaming_batches_flush_unexpected_error"
                    in mock_logger.error.call_args[0]
                )
                assert call_args["error"] == "Flush failed"
                assert call_args["exc_info"] is not None


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

    # Removed deprecated Claude detection startup tests - function no longer exists


class TestPermissionServiceLifecycle:
    """Test permission service initialization and shutdown."""

    @pytest.fixture
    def mock_app(self) -> FastAPI:
        """Create a mock FastAPI app."""
        app = FastAPI()
        app.state = Mock()
        return app

    @pytest.fixture
    def mock_settings_enabled(self) -> Mock:
        """Create mock settings with permissions enabled."""
        settings = Mock(spec=Settings)
        # Configure nested attributes properly
        settings.claude = Mock()
        settings.claude.builtin_permissions = True
        settings.server = Mock()
        settings.server.use_terminal_permission_handler = False
        return settings

    @pytest.fixture
    def mock_settings_disabled(self) -> Mock:
        """Create mock settings with permissions disabled."""
        settings = Mock(spec=Settings)
        # Configure nested attributes properly
        settings.claude = Mock()
        settings.claude.builtin_permissions = False
        return settings

    async def test_permission_service_startup_success(
        self, mock_app: FastAPI, mock_settings_enabled: Mock
    ) -> None:
        """Test successful permission service initialization."""
        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await initialize_permission_service_startup(mock_app, mock_settings_enabled)

            # The function now just passes (handled by plugin)
            # Verify no logs were called
            mock_logger.debug.assert_not_called()
            mock_logger.error.assert_not_called()

    async def test_permission_service_startup_disabled(
        self, mock_app: FastAPI, mock_settings_disabled: Mock
    ) -> None:
        """Test when permission service is disabled."""
        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await initialize_permission_service_startup(
                mock_app, mock_settings_disabled
            )

            # The function now just passes (handled by plugin)
            # Verify no logs were called
            mock_logger.debug.assert_not_called()
            mock_logger.error.assert_not_called()

    async def test_permission_service_shutdown_success(
        self, mock_app: FastAPI, mock_settings_enabled: Mock
    ) -> None:
        """Test successful permission service shutdown."""
        mock_permission_service = AsyncMock()
        mock_app.state.permission_service = mock_permission_service

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await setup_permission_service_shutdown(mock_app, mock_settings_enabled)

            # The function now just passes (handled by plugin)
            # Verify no logs were called
            mock_logger.debug.assert_not_called()
            mock_logger.error.assert_not_called()

    async def test_permission_service_shutdown_disabled(
        self, mock_app: FastAPI, mock_settings_disabled: Mock
    ) -> None:
        """Test shutdown when permission service is disabled."""
        mock_app.state.permission_service = AsyncMock()  # Present but disabled

        with patch("ccproxy.utils.startup_helpers.logger") as mock_logger:
            await setup_permission_service_shutdown(mock_app, mock_settings_disabled)

            # The function now just passes (handled by plugin)
            # Verify no logs were called
            mock_logger.debug.assert_not_called()
            mock_logger.error.assert_not_called()
