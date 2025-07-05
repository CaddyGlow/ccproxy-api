"""Claude instance connection pool for improved performance."""

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from typing import Any

from claude_code_proxy.exceptions import ServiceUnavailableError
from claude_code_proxy.utils.helper import patched_typing


with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions


logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """Represents a pooled Claude connection instance."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    use_count: int = 0
    is_healthy: bool = True
    client: Any | None = None  # Will hold the actual Claude client instance

    def mark_used(self) -> None:
        """Mark this connection as recently used."""
        self.last_used_at = datetime.now(UTC)
        self.use_count += 1

    def is_idle_expired(self, idle_timeout_seconds: int) -> bool:
        """Check if this connection has been idle too long."""
        idle_time = (datetime.now(UTC) - self.last_used_at).total_seconds()
        return idle_time > idle_timeout_seconds


class ClaudeInstancePool:
    """
    Manages a pool of Claude client instances for improved performance.

    This pool maintains pre-initialized Claude connections that can be
    reused across requests, eliminating the overhead of creating new
    subprocess connections for each API call.
    """

    def __init__(
        self,
        min_size: int = 2,
        max_size: int = 10,
        idle_timeout: int = 300,
        health_check_interval: int = 60,
        acquire_timeout: float = 5.0,
    ) -> None:
        """
        Initialize the Claude instance pool.

        Args:
            min_size: Minimum number of instances to maintain
            max_size: Maximum number of instances allowed
            idle_timeout: Seconds before idle connections are closed
            health_check_interval: Seconds between health checks
            acquire_timeout: Maximum seconds to wait for an available instance
        """
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.health_check_interval = health_check_interval
        self.acquire_timeout = acquire_timeout

        # Pool state
        self._available: asyncio.Queue[PooledConnection] = asyncio.Queue()
        self._in_use: dict[str, PooledConnection] = {}
        self._total_created = 0
        self._initialized = False
        self._shutdown = False
        self._lock = asyncio.Lock()

        # Background tasks
        self._health_check_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

        # Statistics
        self._stats = {
            "connections_created": 0,
            "connections_destroyed": 0,
            "connections_reused": 0,
            "acquire_timeouts": 0,
            "health_check_failures": 0,
        }

    async def initialize(self) -> None:
        """Initialize the pool and start background tasks."""
        async with self._lock:
            if self._initialized:
                return

            logger.info(
                f"[POOL] Initializing Claude instance pool (min={self.min_size}, max={self.max_size})"
            )

            # Pre-create minimum instances
            for _ in range(self.min_size):
                try:
                    conn = await self._create_connection()
                    await self._available.put(conn)
                except Exception as e:
                    logger.error(f"[POOL] Failed to create initial connection: {e}")

            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._initialized = True
            logger.info(
                f"[POOL] Claude instance pool initialized successfully with {self._total_created} connections"
            )

    async def acquire(self) -> PooledConnection:
        """
        Acquire a connection from the pool.

        Returns:
            A healthy pooled connection ready for use

        Raises:
            ServiceUnavailableError: If no connection can be acquired within timeout
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        timeout_time = start_time + self.acquire_timeout

        while time.time() < timeout_time:
            # Try to get an available connection
            try:
                remaining_timeout = timeout_time - time.time()
                if remaining_timeout <= 0:
                    break

                conn = await asyncio.wait_for(
                    self._available.get(), timeout=min(remaining_timeout, 0.5)
                )

                # Validate connection health
                if await self._validate_connection(conn):
                    conn.mark_used()
                    self._in_use[conn.id] = conn
                    self._stats["connections_reused"] += 1
                    logger.info(
                        f"[POOL] Reusing existing connection {conn.id} "
                        f"(use_count: {conn.use_count}, pool_size: {self._total_created})"
                    )
                    return conn
                else:
                    # Connection is unhealthy, destroy it
                    await self._destroy_connection(conn)
                    continue

            except TimeoutError:
                # No available connection, try to create a new one if under limit
                if self._total_created < self.max_size:
                    try:
                        conn = await self._create_connection()
                        conn.mark_used()
                        self._in_use[conn.id] = conn
                        logger.info(
                            f"[POOL] Created new connection {conn.id} "
                            f"(total: {self._total_created}/{self.max_size})"
                        )
                        return conn
                    except Exception as e:
                        logger.error(f"[POOL] Failed to create new connection: {e}")
                        continue

        # Acquisition timeout
        self._stats["acquire_timeouts"] += 1
        raise ServiceUnavailableError(
            f"Could not acquire Claude connection within {self.acquire_timeout}s timeout"
        )

    async def release(self, connection: PooledConnection) -> None:
        """
        Release a connection back to the pool.

        Args:
            connection: The connection to release
        """
        if connection.id not in self._in_use:
            logger.warning(
                f"[POOL] Attempting to release unknown connection {connection.id}"
            )
            return

        # Remove from in-use tracking
        del self._in_use[connection.id]

        # Check if connection is still healthy
        if connection.is_healthy and not self._shutdown:
            # Return to available pool
            await self._available.put(connection)
            logger.info(
                f"[POOL] Released connection {connection.id} back to pool "
                f"(available: {self._available.qsize() + 1}, in_use: {len(self._in_use) - 1})"
            )
        else:
            # Destroy unhealthy connection
            await self._destroy_connection(connection)

    async def shutdown(self) -> None:
        """Shutdown the pool and clean up all resources."""
        logger.info(
            f"[POOL] Shutting down Claude instance pool "
            f"(destroying {self._total_created} connections)"
        )
        self._shutdown = True

        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Destroy all connections
        all_connections = list(self._in_use.values())

        # Drain available queue
        while not self._available.empty():
            try:
                conn = self._available.get_nowait()
                all_connections.append(conn)
            except asyncio.QueueEmpty:
                break

        # Destroy all connections
        for conn in all_connections:
            await self._destroy_connection(conn)

        # Clear the in-use dictionary
        self._in_use.clear()

        logger.info("[POOL] Claude instance pool shutdown complete")

    async def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        # Import here to avoid circular dependencies
        from claude_code_proxy.services.claude_client import ClaudeClient

        conn = PooledConnection()
        conn.client = ClaudeClient()

        self._total_created += 1
        self._stats["connections_created"] += 1

        logger.info(f"[POOL] Created new pooled connection {conn.id}")
        return conn

    async def _destroy_connection(self, connection: PooledConnection) -> None:
        """Destroy a connection and clean up resources."""
        try:
            if connection.client and hasattr(connection.client, "close"):
                await connection.client.close()
        except Exception as e:
            logger.error(f"[POOL] Error closing connection {connection.id}: {e}")

        self._total_created -= 1
        self._stats["connections_destroyed"] += 1
        logger.info(
            f"[POOL] Destroyed connection {connection.id} "
            f"(total remaining: {self._total_created})"
        )

    async def _validate_connection(self, connection: PooledConnection) -> bool:
        """
        Validate that a connection is still healthy.

        Args:
            connection: The connection to validate

        Returns:
            True if connection is healthy, False otherwise
        """
        if not connection.is_healthy:
            return False

        # TODO: Implement actual health check
        # For now, just check if the client exists
        return connection.client is not None

    async def _health_check_loop(self) -> None:
        """Background task to periodically check connection health."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check all available connections
                connections_to_check = []
                while not self._available.empty():
                    try:
                        conn = self._available.get_nowait()
                        connections_to_check.append(conn)
                    except asyncio.QueueEmpty:
                        break

                # Validate and return healthy connections
                for conn in connections_to_check:
                    if await self._validate_connection(conn):
                        await self._available.put(conn)
                    else:
                        self._stats["health_check_failures"] += 1
                        await self._destroy_connection(conn)

            except Exception as e:
                logger.error(f"[POOL] Error in health check loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up idle connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Clean up idle connections above minimum
                if self._total_created > self.min_size:
                    connections_to_check = []
                    while (
                        not self._available.empty()
                        and self._total_created > self.min_size
                    ):
                        try:
                            conn = self._available.get_nowait()
                            connections_to_check.append(conn)
                        except asyncio.QueueEmpty:
                            break

                    # Check idle timeout and destroy or return connections
                    for conn in connections_to_check:
                        if (
                            conn.is_idle_expired(self.idle_timeout)
                            and self._total_created > self.min_size
                        ):
                            await self._destroy_connection(conn)
                            logger.info(
                                f"[POOL] Cleaned up idle connection {conn.id} "
                                f"(was idle for >={self.idle_timeout} seconds)"
                            )
                        else:
                            await self._available.put(conn)

            except Exception as e:
                logger.error(f"[POOL] Error in cleanup loop: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            **self._stats,
            "total_connections": self._total_created,
            "available_connections": self._available.qsize(),
            "in_use_connections": len(self._in_use),
        }
