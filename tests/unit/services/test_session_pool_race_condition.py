"""Tests for SessionPool race condition fixes.

This module tests race condition scenarios in the SessionPool class,
specifically the fix for the active_stream_handle race condition
that occurs when multiple simultaneous requests access the same session.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from claude_code_sdk import ClaudeCodeOptions

from plugins.claude_sdk.config import SessionPoolSettings
from plugins.claude_sdk.session_client import SessionClient, SessionStatus
from plugins.claude_sdk.session_pool import SessionPool


class TestSessionPoolRaceCondition:
    """Test suite for SessionPool race condition scenarios."""

    @pytest.fixture
    def session_pool_config(self) -> SessionPoolSettings:
        """Create a SessionPoolSettings for testing."""
        return SessionPoolSettings(
            enabled=True,
            max_sessions=10,
            session_ttl=3600,
            cleanup_interval=300,
            connection_recovery=True,
        )

    @pytest.fixture
    def session_pool(self, session_pool_config: SessionPoolSettings) -> SessionPool:
        """Create a SessionPool instance for testing."""
        return SessionPool(config=session_pool_config)

    @pytest.fixture
    def mock_options(self) -> ClaudeCodeOptions:
        """Create mock ClaudeCodeOptions for testing."""
        return ClaudeCodeOptions()

    @pytest.fixture
    def mock_session_client(self) -> SessionClient:
        """Create a mock SessionClient for testing."""
        session_client = Mock(spec=SessionClient)
        session_client.client_id = 12345
        session_client.status = SessionStatus.ACTIVE
        session_client.has_active_stream = True
        session_client.active_stream_handle = None  # Initially None
        session_client.metrics = Mock()
        session_client.metrics.idle_seconds = 0.1
        session_client.metrics.age_seconds = 10.0
        session_client.metrics.message_count = 1
        session_client.is_expired.return_value = False
        session_client.is_healthy = AsyncMock(return_value=True)
        session_client.ensure_connected = AsyncMock(return_value=True)
        session_client.lock = asyncio.Lock()
        return session_client

    @pytest.fixture
    def mock_stream_handle(self) -> Mock:
        """Create a mock stream handle for testing."""
        handle = Mock()
        handle.handle_id = "test-handle-123"
        handle.idle_seconds = 0.1
        handle.has_first_chunk = True
        handle.is_completed = False
        handle.is_first_chunk_timeout.return_value = False
        handle.is_ongoing_timeout.return_value = False
        return handle

    async def test_active_stream_handle_null_check_prevents_race_condition(
        self,
        session_pool: SessionPool,
        mock_options: ClaudeCodeOptions,
        mock_session_client: SessionClient,
        mock_stream_handle: Mock,
    ) -> None:
        """Test that null check prevents race condition when handle becomes None."""
        session_id = "test-session-race"

        # Set up session with active stream handle that will return values simulating a race condition
        # The race condition occurs when handle is checked as not None, but becomes None before
        # timeout method calls are made
        mock_session_client.active_stream_handle = mock_stream_handle
        mock_stream_handle.is_first_chunk_timeout.return_value = False
        mock_stream_handle.is_ongoing_timeout.return_value = False

        # Mock the session pool to have our test session
        with patch.object(session_pool, "sessions", {session_id: mock_session_client}):
            # This test should execute without any AttributeError or other exceptions
            # The race condition protection should handle cases where handle becomes None
            result = await session_pool.get_session_client(session_id, mock_options)

            # Verify we got a session client back
            assert result is not None
            assert isinstance(result, SessionClient)

            # Verify the timeout methods were called (meaning the code path was exercised)
            mock_stream_handle.is_first_chunk_timeout.assert_called_once()
            mock_stream_handle.is_ongoing_timeout.assert_called_once()

    async def test_handle_cleared_by_concurrent_request_no_timeout_checks(
        self,
        session_pool: SessionPool,
        mock_options: ClaudeCodeOptions,
        mock_session_client: SessionClient,
    ) -> None:
        """Test behavior when handle is cleared by concurrent request."""
        session_id = "test-session-concurrent"

        # Set up session that indicates it has an active stream but handle is None
        # This simulates the state after another request cleared the handle
        mock_session_client.has_active_stream = True
        mock_session_client.active_stream_handle = None

        # Mock the session pool to have our test session
        with patch.object(session_pool, "sessions", {session_id: mock_session_client}):
            # This should handle the None handle gracefully
            result = await session_pool.get_session_client(session_id, mock_options)

            # Verify we got a session client back
            assert result is not None
            assert result == mock_session_client
            # The has_active_stream flag should be cleared when handle is None
            assert not mock_session_client.has_active_stream

    async def test_concurrent_requests_same_session_id(
        self,
        session_pool: SessionPool,
        mock_options: ClaudeCodeOptions,
        mock_session_client: SessionClient,
        mock_stream_handle: Mock,
    ) -> None:
        """Test multiple concurrent requests to the same session_id."""
        session_id = "test-session-concurrent-multiple"

        # Set up session with active stream handle
        mock_session_client.active_stream_handle = mock_stream_handle

        # Mock the session pool to have our test session
        with patch.object(session_pool, "sessions", {session_id: mock_session_client}):

            async def concurrent_request() -> SessionClient:
                return await session_pool.get_session_client(session_id, mock_options)

            # Simulate scenario where one request clears handle while others are processing
            async def clear_handle_after_delay() -> None:
                await asyncio.sleep(0.01)  # Small delay
                mock_session_client.active_stream_handle = None
                mock_session_client.has_active_stream = False

            # Start multiple concurrent requests and a handle-clearing task
            tasks = [
                asyncio.create_task(concurrent_request()),
                asyncio.create_task(concurrent_request()),
                asyncio.create_task(concurrent_request()),
                asyncio.create_task(clear_handle_after_delay()),
            ]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # First 3 results should be session clients, last should be None (clear task)
            session_results = results[:3]
            for result in session_results:
                assert not isinstance(result, Exception), f"Got exception: {result}"
                assert result is not None
                assert result == mock_session_client

    async def test_handle_timeout_methods_called_safely(
        self,
        session_pool: SessionPool,
        mock_options: ClaudeCodeOptions,
        mock_session_client: SessionClient,
        mock_stream_handle: Mock,
    ) -> None:
        """Test that timeout methods are called safely when handle exists."""
        session_id = "test-session-timeout-safe"

        # Set up session with active stream handle that has timeout
        mock_stream_handle.is_first_chunk_timeout.return_value = False
        mock_stream_handle.is_ongoing_timeout.return_value = (
            True  # Simulate ongoing timeout
        )
        mock_session_client.active_stream_handle = mock_stream_handle

        # Mock interrupt method
        mock_stream_handle.interrupt = AsyncMock(return_value=True)

        # Mock the session pool to have our test session
        with patch.object(session_pool, "sessions", {session_id: mock_session_client}):
            result = await session_pool.get_session_client(session_id, mock_options)

            # Verify timeout methods were called
            mock_stream_handle.is_first_chunk_timeout.assert_called_once()
            mock_stream_handle.is_ongoing_timeout.assert_called_once()

            # Verify interrupt was called due to ongoing timeout
            mock_stream_handle.interrupt.assert_called_once()

            # Verify handle was cleared after interrupt
            assert mock_session_client.active_stream_handle is None
            assert not mock_session_client.has_active_stream

            assert result is not None
            assert result == mock_session_client
