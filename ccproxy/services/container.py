"""Dependency injection container for all services.

This module provides a clean, testable dependency injection container that
manages service lifecycles and dependencies without singleton anti-patterns.
"""

from typing import Any, cast

import httpx
import structlog

from ccproxy.config.settings import Settings
from ccproxy.core.http import BaseProxyClient
from ccproxy.services.cache import ResponseCache
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.factories import ConcreteServiceFactory
from ccproxy.services.http.connection_pool import ConnectionPoolManager
from ccproxy.services.http_pool import HTTPPoolManager
from ccproxy.services.interfaces import (
    IRequestTracer,
    NullMetricsCollector,
    NullRequestTracer,
)
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.services.streaming import StreamingHandler
from ccproxy.services.tracing import RequestTracer
from ccproxy.utils.binary_resolver import BinaryResolver


logger = structlog.get_logger(__name__)


class ServiceContainer:
    """Dependency injection container for all services.

    This container manages service lifecycles and dependencies using proper
    dependency injection patterns. It removes singleton anti-patterns and
    provides a clean, testable interface for service management.

    Key improvements:
    - No singleton pattern (_instance class variable removed)
    - Uses factory pattern for service creation
    - Implements service interfaces for better testing
    - Manages resource lifecycle properly
    - Supports dependency injection for better testability
    """

    def __init__(self, settings: Settings, service_factory: Any | None = None) -> None:
        """Initialize the service container.

        Args:
            settings: Application settings
            service_factory: Optional service factory (for testing/customization)
        """
        self.settings = settings
        self._factory = service_factory or ConcreteServiceFactory()

        # Service instances (created on-demand, not singletons)
        self._services: dict[str, object] = {}
        self._http_client: httpx.AsyncClient | None = None
        self._pool_manager: HTTPPoolManager | None = None
        self._request_tracer: RequestTracer | None = None  # Set by plugin
        self._proxy_client: BaseProxyClient | None = None  # Managed proxy client

        logger.debug(
            "service_container_initialized",
            has_factory=service_factory is not None,
            category="lifecycle",
        )

    def set_proxy_client(self, proxy_client: BaseProxyClient) -> None:
        """Set the proxy client for the container to manage.

        Args:
            proxy_client: HTTP proxy client to manage
        """
        self._proxy_client = proxy_client
        logger.debug("proxy_client_set", category="lifecycle")

    def get_request_tracer(self) -> IRequestTracer:
        """Get request tracer service instance.

        Returns the plugin-injected tracer or NullRequestTracer as fallback.

        Returns:
            Request tracer service instance
        """
        if self._request_tracer is None:
            # No plugin has registered a tracer, use null implementation
            self._request_tracer = NullRequestTracer()
            logger.debug("using_null_request_tracer", category="lifecycle")
        return self._request_tracer

    def set_request_tracer(self, tracer: RequestTracer) -> None:
        """Set the request tracer (called by plugin).

        Args:
            tracer: The request tracer implementation
        """
        self._request_tracer = tracer
        logger.info(
            "request_tracer_set",
            tracer_type=type(tracer).__name__,
            category="lifecycle",
        )

    def get_mock_handler(self) -> MockResponseHandler:
        """Get mock handler service instance.

        Uses caching to ensure single instance per container lifetime,
        but allows multiple containers for testing.

        Returns:
            Mock handler service instance
        """
        service_key = "mock_handler"
        if service_key not in self._services:
            self._services[service_key] = self._factory.create_mock_handler(
                self.settings
            )
        return self._services[service_key]  # type: ignore

    def get_streaming_handler(
        self,
        metrics: Any | None = None,
    ) -> StreamingHandler:
        """Get streaming handler service instance.

        Uses caching to ensure single instance per container lifetime,
        but allows multiple containers for testing.

        Args:
            metrics: Optional metrics service

        Returns:
            Streaming handler service instance
        """
        service_key = "streaming_handler"
        if service_key not in self._services:
            request_tracer = self.get_request_tracer()
            self._services[service_key] = self._factory.create_streaming_handler(
                self.settings,
                metrics=metrics,
                request_tracer=request_tracer,
            )
        return cast(StreamingHandler, self._services[service_key])

    def get_binary_resolver(self) -> BinaryResolver:
        """Get binary resolver service instance.

        Uses caching to ensure single instance per container lifetime,
        but allows multiple containers for testing.

        Returns:
            Binary resolver service instance
        """
        service_key = "binary_resolver"
        if service_key not in self._services:
            self._services[service_key] = BinaryResolver.from_settings(self.settings)
        return self._services[service_key]  # type: ignore

    def get_cli_detection_service(self) -> CLIDetectionService:
        """Get CLI detection service instance.

        Uses caching to ensure single instance per container lifetime,
        but allows multiple containers for testing.

        Returns:
            CLI detection service instance
        """
        service_key = "cli_detection_service"
        if service_key not in self._services:
            binary_resolver = self.get_binary_resolver()
            self._services[service_key] = CLIDetectionService(
                self.settings, binary_resolver
            )
        return self._services[service_key]  # type: ignore

    def get_proxy_config(self) -> ProxyConfiguration:
        """Get proxy configuration service instance.

        Uses caching to ensure single instance per container lifetime,
        but allows multiple containers for testing.

        Returns:
            Proxy configuration service instance
        """
        service_key = "proxy_config"
        if service_key not in self._services:
            self._services[service_key] = self._factory.create_proxy_config()
        return self._services[service_key]  # type: ignore

    def get_http_client(self) -> httpx.AsyncClient:
        """Get shared HTTP client instance.

        This provides the centralized HTTP client with optimized configuration
        for the proxy use case. Only one HTTP client per container to ensure
        proper resource management.

        Returns:
            Shared httpx.AsyncClient instance
        """
        if not self._http_client:
            # Use pool manager for centralized management
            pool_manager = self.get_pool_manager()
            # Use synchronous version during initialization
            self._http_client = pool_manager.get_shared_client_sync()
        return self._http_client

    def get_pool_manager(self) -> HTTPPoolManager:
        """Get HTTP connection pool manager instance.

        This provides centralized management of HTTP connection pools,
        ensuring efficient resource usage across all components.

        Returns:
            HTTPPoolManager instance
        """
        if not self._pool_manager:
            self._pool_manager = HTTPPoolManager(self.settings)
            logger.debug("http_pool_manager_created", category="lifecycle")
        return self._pool_manager

    def get_response_cache(self) -> ResponseCache:
        """Get response cache service instance.

        Returns:
            ResponseCache instance for caching API responses
        """
        service_key = "response_cache"
        if service_key not in self._services:
            # Configure cache based on settings
            cache_settings = getattr(self.settings, "cache", None)
            if cache_settings:
                ttl = getattr(cache_settings, "default_ttl", 300.0)
                max_size = getattr(cache_settings, "max_size", 1000)
                self._services[service_key] = ResponseCache(
                    default_ttl=ttl, max_size=max_size
                )
            else:
                # Default configuration
                self._services[service_key] = ResponseCache()
            logger.debug("response_cache_created", category="lifecycle")
        return self._services[service_key]  # type: ignore

    def get_connection_pool_manager(self) -> ConnectionPoolManager:
        """Get connection pool manager service instance.

        Returns:
            ConnectionPoolManager instance for managing HTTP connection pools
        """
        service_key = "connection_pool_manager"
        if service_key not in self._services:
            # Configure based on settings
            pool_settings = getattr(self.settings, "http", None)
            if pool_settings:
                timeout = getattr(pool_settings, "timeout", 120.0)
                pool_size = getattr(pool_settings, "pool_size", 20)
                self._services[service_key] = ConnectionPoolManager(
                    default_timeout=timeout, pool_size=pool_size
                )
            else:
                # Default configuration
                self._services[service_key] = ConnectionPoolManager()
            logger.debug("connection_pool_manager_created", category="lifecycle")
        return self._services[service_key]  # type: ignore

    def get_adapter_dependencies(self, metrics: Any | None = None) -> dict[str, Any]:
        """Get all services an adapter might need.

        This method provides individual services for explicit dependency injection
        in adapters, replacing the ProxyService service locator pattern.

        Args:
            metrics: Optional metrics service

        Returns:
            Dictionary of services that can be injected into adapters
        """
        return {
            "http_client": self.get_http_client(),
            "request_tracer": self.get_request_tracer(),
            "metrics": metrics or NullMetricsCollector(),
            "streaming_handler": self.get_streaming_handler(metrics),
            "logger": structlog.get_logger(),
            "config": self.get_proxy_config(),
            "cli_detection_service": self.get_cli_detection_service(),
        }

    async def close(self) -> None:
        """Close all managed resources during shutdown.

        This method properly cleans up all resources managed by the container,
        ensuring graceful shutdown and preventing resource leaks.
        """
        # Close proxy client if it exists
        if self._proxy_client:
            if hasattr(self._proxy_client, "close"):
                try:
                    await self._proxy_client.close()
                    logger.debug("proxy_client_closed", category="lifecycle")
                except Exception as e:
                    logger.error(
                        "proxy_client_close_failed",
                        error=str(e),
                        exc_info=e,
                        category="lifecycle",
                    )
            self._proxy_client = None

        # Close pool manager (which closes all HTTP clients including the shared client)
        if self._pool_manager:
            await self._pool_manager.close_all()
            self._pool_manager = None
            # Clear the HTTP client reference since it's closed by pool manager
            self._http_client = None
            logger.debug("http_pool_manager_closed", category="lifecycle")

        # Close HTTP client if it was created separately (should not happen normally)
        elif self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            logger.debug("http_client_closed_directly", category="lifecycle")

        # Clear service cache
        self._services.clear()
        logger.debug("service_container_resources_closed", category="lifecycle")
