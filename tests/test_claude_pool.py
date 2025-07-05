"""Unit tests for Claude instance connection pool."""

import asyncio
import contextlib
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_code_proxy.exceptions import ServiceUnavailableError
from claude_code_proxy.services.claude_pool import ClaudeInstancePool, PooledConnection


@pytest.mark.unit
class TestPooledConnection:
    """Test PooledConnection functionality."""

    def test_pooled_connection_initialization(self):
        """Test PooledConnection initializes correctly."""
        conn = PooledConnection()

        assert conn.id is not None
        assert conn.created_at is not None
        assert conn.last_used_at is not None
        assert conn.use_count == 0
        assert conn.is_healthy is True
        assert conn.client is None

    def test_mark_used(self):
        """Test marking connection as used."""
        conn = PooledConnection()
        initial_time = conn.last_used_at
        initial_count = conn.use_count

        conn.mark_used()

        assert conn.use_count == initial_count + 1
        assert conn.last_used_at > initial_time

    def test_is_idle_expired(self):
        """Test idle expiration check."""
        conn = PooledConnection()

        # Should not be expired immediately
        assert not conn.is_idle_expired(300)

        # Mock time to simulate idle period
        from datetime import datetime, timedelta, timezone

        conn.last_used_at = datetime.now(UTC) - timedelta(seconds=400)

        # Should be expired after 300 seconds
        assert conn.is_idle_expired(300)
        assert not conn.is_idle_expired(500)


@pytest.mark.unit
class TestClaudeInstancePool:
    """Test ClaudeInstancePool functionality."""

    @pytest.fixture
    async def pool(self):
        """Create a test pool instance."""
        pool = ClaudeInstancePool(
            min_size=2,
            max_size=5,
            idle_timeout=300,
            health_check_interval=60,
            acquire_timeout=2.0,
        )
        yield pool
        # Ensure cleanup
        if pool._initialized:
            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool):
        """Test pool initializes with correct parameters."""
        assert pool.min_size == 2
        assert pool.max_size == 5
        assert pool.idle_timeout == 300
        assert pool.health_check_interval == 60
        assert pool.acquire_timeout == 2.0
        assert not pool._initialized
        assert not pool._shutdown

    @pytest.mark.asyncio
    async def test_pool_initialize_creates_min_connections(self, pool):
        """Test pool creates minimum connections on initialization."""
        with patch(
            "claude_code_proxy.services.claude_client.ClaudeClient"
        ) as mock_client:
            mock_client.return_value = MagicMock()

            await pool.initialize()

            assert pool._initialized
            assert pool._total_created == pool.min_size
            assert pool._available.qsize() == pool.min_size
            assert pool._stats["connections_created"] == pool.min_size

    @pytest.mark.asyncio
    async def test_acquire_from_available_pool(self):
        """Test acquiring connection from available pool."""
        # Create fresh pool
        pool = ClaudeInstancePool(min_size=0, max_size=5)

        # Pre-populate pool
        conn = PooledConnection()
        conn.client = MagicMock()
        await pool._available.put(conn)
        pool._total_created = 1

        # Acquire connection
        acquired = await pool.acquire()

        assert acquired.id == conn.id
        assert acquired.use_count == 1
        assert pool._in_use[conn.id] == conn
        assert pool._available.qsize() == 0

        # Cleanup
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_creates_new_when_empty(self):
        """Test acquiring creates new connection when pool is empty."""
        # Create pool with no min_size
        pool = ClaudeInstancePool(min_size=0, max_size=5)

        with patch(
            "claude_code_proxy.services.claude_client.ClaudeClient"
        ) as mock_client:
            mock_client.return_value = MagicMock()

            # Pool is empty, should create new
            acquired = await pool.acquire()

            assert acquired is not None
            assert acquired.client is not None
            assert pool._total_created == 1
            assert pool._stats["connections_created"] == 1

            # Cleanup
            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test acquire times out when no connections available."""
        # Create a pool with no min_size to avoid pre-created connections
        pool = ClaudeInstancePool(min_size=0, max_size=5, acquire_timeout=0.1)

        # Set max_size to current total to prevent new creation
        pool._total_created = pool.max_size
        pool._initialized = True  # Mark as initialized

        with pytest.raises(ServiceUnavailableError) as exc_info:
            await pool.acquire()

        assert "Could not acquire Claude connection" in str(exc_info.value)
        assert pool._stats["acquire_timeouts"] == 1

        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_release_healthy_connection(self, pool):
        """Test releasing healthy connection back to pool."""
        conn = PooledConnection()
        conn.client = MagicMock()
        pool._in_use[conn.id] = conn

        await pool.release(conn)

        assert conn.id not in pool._in_use
        assert pool._available.qsize() == 1

    @pytest.mark.asyncio
    async def test_release_unhealthy_connection(self, pool):
        """Test releasing unhealthy connection destroys it."""
        conn = PooledConnection()
        conn.client = MagicMock()
        conn.is_healthy = False
        pool._in_use[conn.id] = conn
        pool._total_created = 1

        await pool.release(conn)

        assert conn.id not in pool._in_use
        assert pool._available.qsize() == 0
        assert pool._total_created == 0
        assert pool._stats["connections_destroyed"] == 1

    @pytest.mark.asyncio
    async def test_validate_connection(self, pool):
        """Test connection validation."""
        # Healthy connection
        conn = PooledConnection()
        conn.client = MagicMock()
        assert await pool._validate_connection(conn)

        # Unhealthy connection
        conn.is_healthy = False
        assert not await pool._validate_connection(conn)

        # No client
        conn.is_healthy = True
        conn.client = None
        assert not await pool._validate_connection(conn)

    @pytest.mark.asyncio
    async def test_pool_shutdown(self):
        """Test pool shutdown cleans up all resources."""
        # Create a fresh pool
        pool = ClaudeInstancePool(min_size=0, max_size=5)

        # Create some connections
        conn1 = PooledConnection()
        conn1.client = MagicMock()
        conn2 = PooledConnection()
        conn2.client = MagicMock()

        await pool._available.put(conn1)
        pool._in_use[conn2.id] = conn2
        pool._total_created = 2

        # Start background tasks
        pool._health_check_task = asyncio.create_task(asyncio.sleep(10))
        pool._cleanup_task = asyncio.create_task(asyncio.sleep(10))

        await pool.shutdown()

        assert pool._shutdown
        assert pool._available.qsize() == 0
        assert len(pool._in_use) == 0
        assert pool._total_created == 0
        assert pool._health_check_task.cancelled()
        assert pool._cleanup_task.cancelled()

    @pytest.mark.asyncio
    async def test_get_stats(self, pool):
        """Test getting pool statistics."""
        pool._total_created = 3
        pool._stats["connections_created"] = 5
        pool._stats["connections_destroyed"] = 2

        stats = pool.get_stats()

        assert stats["total_connections"] == 3
        assert stats["connections_created"] == 5
        assert stats["connections_destroyed"] == 2
        assert "available_connections" in stats
        assert "in_use_connections" in stats


@pytest.mark.unit
class TestHealthCheckLoop:
    """Test health check background task."""

    @pytest.mark.asyncio
    async def test_health_check_removes_unhealthy(self):
        """Test health check removes unhealthy connections."""
        pool = ClaudeInstancePool(health_check_interval=1)  # Short interval for test

        # Add healthy and unhealthy connections
        healthy_conn = PooledConnection()
        healthy_conn.client = MagicMock()

        unhealthy_conn = PooledConnection()
        unhealthy_conn.client = MagicMock()
        unhealthy_conn.is_healthy = False

        await pool._available.put(healthy_conn)
        await pool._available.put(unhealthy_conn)
        pool._total_created = 2

        # Run one iteration of health check
        pool._shutdown = False
        health_task = asyncio.create_task(pool._health_check_loop())
        # Wait for health check to complete one cycle
        await asyncio.sleep(1.5)  # Slightly longer than the 1s interval
        pool._shutdown = True
        health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await health_task

        # Check results - the unhealthy connection should be removed
        assert pool._available.qsize() == 1
        assert pool._stats["health_check_failures"] == 1
        assert pool._total_created == 1  # One destroyed


@pytest.mark.unit
class TestCleanupLoop:
    """Test cleanup background task."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_idle_connections(self):
        """Test cleanup removes idle connections above minimum."""
        pool = ClaudeInstancePool(min_size=1, idle_timeout=30)

        # Add connections
        conn1 = PooledConnection()
        conn1.client = MagicMock()

        conn2 = PooledConnection()
        conn2.client = MagicMock()
        # Make conn2 idle
        from datetime import datetime, timedelta, timezone

        conn2.last_used_at = datetime.now(UTC) - timedelta(seconds=1)

        await pool._available.put(conn1)
        await pool._available.put(conn2)
        pool._total_created = 2

        # Run cleanup
        pool._shutdown = False
        cleanup_task = asyncio.create_task(pool._cleanup_loop())
        await asyncio.sleep(0.1)  # Let it start
        pool._shutdown = True
        cleanup_task.cancel()

        # Should keep minimum connections
        assert pool._total_created >= pool.min_size
