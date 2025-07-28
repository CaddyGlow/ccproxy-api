"""ClaudeSDKClient connection pool for improved performance."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from ccproxy.core.async_utils import patched_typing

if TYPE_CHECKING:
    from ccproxy.observability.metrics import PrometheusMetrics


with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions
    from claude_code_sdk import ClaudeSDKClient as SDKClient


logger = structlog.get_logger(__name__)


@dataclass
class PoolConfig:
    """Configuration for the Claude SDK client pool."""

    pool_size: int = 3
    max_pool_size: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    health_check_interval: float = 60.0
    enable_health_checks: bool = True


class PoolStats(BaseModel):
    """Statistics for the client pool."""

    total_clients: int
    available_clients: int
    active_clients: int
    connections_created: int
    connections_closed: int
    acquire_count: int
    release_count: int
    health_check_failures: int


class PooledClient:
    """Wrapper for pooled Claude SDK client with metadata."""

    def __init__(self, client: SDKClient, options: ClaudeCodeOptions):
        self.client = client
        self.options = options
        self.created_at = time.time()
        self.last_used = time.time()
        self.is_connected = False
        self.is_healthy = True
        self.use_count = 0

    async def connect(self) -> None:
        """Connect the client if not already connected."""
        if not self.is_connected:
            await self.client.connect()
            self.is_connected = True
            logger.debug("pooled_client_connected", client_id=id(self.client))

    async def disconnect(self) -> None:
        """Disconnect the client."""
        if self.is_connected:
            try:
                await self.client.disconnect()
            except Exception as e:
                logger.warning("pooled_client_disconnect_error", error=str(e))
            finally:
                self.is_connected = False
                logger.debug("pooled_client_disconnected", client_id=id(self.client))

    def mark_used(self) -> None:
        """Mark the client as recently used."""
        self.last_used = time.time()
        self.use_count += 1

    def is_idle(self, idle_timeout: float) -> bool:
        """Check if the client has been idle too long."""
        return time.time() - self.last_used > idle_timeout

    async def health_check(self) -> bool:
        """Perform a health check on the client."""
        try:
            # Simple health check - ensure client is still connected and responsive
            if not self.is_connected:
                return False

            # Could add more sophisticated health checks here
            self.is_healthy = True
            return True
        except Exception as e:
            logger.warning("pooled_client_health_check_failed", error=str(e))
            self.is_healthy = False
            return False


class ClaudeSDKClientPool:
    """
    Connection pool for Claude SDK clients to improve performance by reusing connections.

    This pool manages a set of pre-connected ClaudeSDKClient instances to eliminate
    the overhead of creating new subprocesses/connections for each request.
    """

    def __init__(
        self,
        config: PoolConfig | None = None,
        default_options: ClaudeCodeOptions | None = None,
        metrics: "PrometheusMetrics | None" = None,
    ):
        """
        Initialize the client pool.

        Args:
            config: Pool configuration
            default_options: Default Claude Code options for clients
            metrics: Optional PrometheusMetrics instance for monitoring
        """
        self.config = config or PoolConfig()
        self.default_options = default_options or ClaudeCodeOptions()
        self._metrics = metrics

        self._available_clients: asyncio.Queue[PooledClient] = asyncio.Queue()
        self._all_clients: set[PooledClient] = set()
        self._active_clients: set[PooledClient] = set()
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = PoolStats(
            total_clients=0,
            available_clients=0,
            active_clients=0,
            connections_created=0,
            connections_closed=0,
            acquire_count=0,
            release_count=0,
            health_check_failures=0,
        )

        # Background tasks
        self._health_check_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._shutdown = False

    async def start(self) -> None:
        """Start the pool and create initial connections."""
        logger.info("claude_sdk_pool_starting", pool_size=self.config.pool_size)

        # Create initial pool of clients and add them to available queue
        for _ in range(self.config.pool_size):
            pooled_client = await self._create_client()
            await self._available_clients.put(pooled_client)

        # Update initial metrics
        if self._metrics:
            stats = self.get_stats()
            self._metrics.update_pool_gauges(
                stats.total_clients, stats.available_clients, stats.active_clients
            )

        # Start background tasks
        if self.config.enable_health_checks:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            "claude_sdk_pool_started",
            total_clients=len(self._all_clients),
            available_clients=self._available_clients.qsize(),
        )

    async def stop(self) -> None:
        """Stop the pool and cleanup all connections."""
        self._shutdown = True
        logger.info("claude_sdk_pool_stopping")

        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_check_task

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Disconnect all clients
        async with self._lock:
            all_clients = list(self._all_clients)
            for pooled_client in all_clients:
                await pooled_client.disconnect()

            self._all_clients.clear()
            self._active_clients.clear()

            # Clear the queue
            while not self._available_clients.empty():
                self._available_clients.get_nowait()

        logger.info("claude_sdk_pool_stopped")

    @asynccontextmanager
    async def acquire_client(
        self, options: ClaudeCodeOptions | None = None
    ) -> AsyncIterator[SDKClient]:
        """
        Acquire a client from the pool.

        Args:
            options: Claude Code options (uses default if None)

        Yields:
            Connected Claude SDK client
        """
        start_time = time.time()
        pooled_client = await self._get_client(options or self.default_options)
        acquisition_time = time.time() - start_time
        
        self._stats.acquire_count += 1
        
        # Record metrics
        if self._metrics:
            self._metrics.inc_pool_acquisitions()
            self._metrics.record_pool_acquisition_time(acquisition_time)

        try:
            await pooled_client.connect()
            pooled_client.mark_used()
            yield pooled_client.client
        finally:
            await self._return_client(pooled_client)
            self._stats.release_count += 1
            
            # Record metrics
            if self._metrics:
                self._metrics.inc_pool_releases()
                stats = self.get_stats()
                self._metrics.update_pool_gauges(
                    stats.total_clients, stats.available_clients, stats.active_clients
                )

    async def _get_client(self, options: ClaudeCodeOptions) -> PooledClient:
        """Get a client from the pool or create a new one."""
        try:
            # Try to get an available client with a short timeout
            pooled_client = await asyncio.wait_for(
                self._available_clients.get(), timeout=self.config.connection_timeout
            )

            async with self._lock:
                self._active_clients.add(pooled_client)

            logger.debug(
                "claude_sdk_pool_client_acquired_from_pool",
                client_id=id(pooled_client.client),
            )
            return pooled_client

        except TimeoutError:
            # No available clients, create a new one if under max limit
            async with self._lock:
                if len(self._all_clients) < self.config.max_pool_size:
                    pooled_client = await self._create_client()
                    self._active_clients.add(pooled_client)
                    logger.debug(
                        "claude_sdk_pool_client_created_on_demand",
                        client_id=id(pooled_client.client),
                    )
                    return pooled_client
                else:
                    # Wait for a client to become available
                    logger.warning(
                        "claude_sdk_pool_max_size_reached",
                        max_size=self.config.max_pool_size,
                    )
                    raise RuntimeError(
                        f"Pool max size ({self.config.max_pool_size}) reached"
                    ) from None

    async def _return_client(self, pooled_client: PooledClient) -> None:
        """Return a client to the pool."""
        async with self._lock:
            if pooled_client in self._active_clients:
                self._active_clients.remove(pooled_client)

            if pooled_client.is_healthy and pooled_client in self._all_clients:
                await self._available_clients.put(pooled_client)
                logger.debug(
                    "claude_sdk_pool_client_returned",
                    client_id=id(pooled_client.client),
                )
            else:
                # Remove unhealthy or unknown clients
                await self._remove_client(pooled_client)

    async def _create_client(self) -> PooledClient:
        """Create a new pooled client."""
        client = SDKClient(options=self.default_options)
        pooled_client = PooledClient(client, self.default_options)

        async with self._lock:
            self._all_clients.add(pooled_client)
            self._stats.connections_created += 1

        # Record metrics
        if self._metrics:
            self._metrics.inc_pool_connections_created()
            stats = self.get_stats()
            self._metrics.update_pool_gauges(
                stats.total_clients, stats.available_clients, stats.active_clients
            )

        logger.debug(
            "claude_sdk_pool_client_created",
            client_id=id(client),
            total_clients=len(self._all_clients),
        )
        return pooled_client

    async def _remove_client(self, pooled_client: PooledClient) -> None:
        """Remove a client from the pool."""
        await pooled_client.disconnect()

        if pooled_client in self._all_clients:
            self._all_clients.remove(pooled_client)
            self._stats.connections_closed += 1

        # Record metrics
        if self._metrics:
            self._metrics.inc_pool_connections_closed()
            stats = self.get_stats()
            self._metrics.update_pool_gauges(
                stats.total_clients, stats.available_clients, stats.active_clients
            )

        logger.debug(
            "claude_sdk_pool_client_removed",
            client_id=id(pooled_client.client),
            total_clients=len(self._all_clients),
        )

    async def _health_check_loop(self) -> None:
        """Background task to perform health checks on idle clients."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("claude_sdk_pool_health_check_error", error=str(e))

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup idle clients."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_clients()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("claude_sdk_pool_cleanup_error", error=str(e))

    async def _perform_health_checks(self) -> None:
        """Perform health checks on available clients."""
        # Get a snapshot of available clients
        available_clients = []
        while not self._available_clients.empty():
            try:
                client = self._available_clients.get_nowait()
                available_clients.append(client)
            except asyncio.QueueEmpty:
                break

        healthy_clients = []
        for pooled_client in available_clients:
            if await pooled_client.health_check():
                healthy_clients.append(pooled_client)
            else:
                self._stats.health_check_failures += 1
                # Record metrics for health check failure
                if self._metrics:
                    self._metrics.inc_pool_health_check_failures()
                await self._remove_client(pooled_client)

        # Put healthy clients back
        for client in healthy_clients:
            await self._available_clients.put(client)

        logger.debug(
            "claude_sdk_pool_health_check_completed",
            checked=len(available_clients),
            healthy=len(healthy_clients),
            removed=len(available_clients) - len(healthy_clients),
        )

    async def _cleanup_idle_clients(self) -> None:
        """Remove clients that have been idle too long."""
        if len(self._all_clients) <= self.config.pool_size:
            return  # Don't shrink below minimum pool size

        # Get available clients and check for idle ones
        available_clients = []
        while not self._available_clients.empty():
            try:
                client = self._available_clients.get_nowait()
                available_clients.append(client)
            except asyncio.QueueEmpty:
                break

        active_clients = []
        removed_count = 0

        for pooled_client in available_clients:
            if (
                pooled_client.is_idle(self.config.idle_timeout)
                and len(self._all_clients) - removed_count > self.config.pool_size
            ):
                await self._remove_client(pooled_client)
                removed_count += 1
            else:
                active_clients.append(pooled_client)

        # Put non-idle clients back
        for client in active_clients:
            await self._available_clients.put(client)

        if removed_count > 0:
            logger.debug(
                "claude_sdk_pool_idle_cleanup_completed", removed_count=removed_count
            )

    def get_stats(self) -> PoolStats:
        """Get current pool statistics."""
        self._stats.total_clients = len(self._all_clients)
        self._stats.available_clients = self._available_clients.qsize()
        self._stats.active_clients = len(self._active_clients)
        return self._stats


# Global pool instance
_global_pool: ClaudeSDKClientPool | None = None


async def get_global_pool(config: PoolConfig | None = None, metrics: "PrometheusMetrics | None" = None) -> ClaudeSDKClientPool:
    """Get or create the global client pool."""
    global _global_pool

    if _global_pool is None:
        # Get metrics if not provided
        if metrics is None:
            try:
                from ccproxy.observability.metrics import get_metrics
                metrics = get_metrics()
            except ImportError:
                metrics = None
        
        _global_pool = ClaudeSDKClientPool(config=config, metrics=metrics)
        await _global_pool.start()

    return _global_pool


async def shutdown_global_pool() -> None:
    """Shutdown the global client pool."""
    global _global_pool

    if _global_pool:
        await _global_pool.stop()
        _global_pool = None
