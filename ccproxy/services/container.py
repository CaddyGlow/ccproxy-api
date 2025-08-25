"""Dependency injection container for all services.

This module provides a clean, testable dependency injection container that
manages service lifecycles and dependencies without singleton anti-patterns.
"""

from typing import TYPE_CHECKING, Any, cast

import httpx
import structlog

from ccproxy.config.settings import Settings
from ccproxy.core.http import BaseProxyClient
from ccproxy.observability.metrics import PrometheusMetrics
from ccproxy.services.cache import ResponseCache
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.factories import ConcreteServiceFactory
from ccproxy.services.http.connection_pool import ConnectionPoolManager
from ccproxy.services.http_pool import HTTPPoolManager
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.services.streaming import StreamingHandler
from ccproxy.services.tracing import CoreRequestTracer
from ccproxy.utils.binary_resolver import BinaryResolver
from plugins.pricing.service import PricingService


if TYPE_CHECKING:
    from ccproxy.services.proxy_service import ProxyService


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

        logger.debug(
            "service_container_initialized",
            has_factory=service_factory is not None,
            category="lifecycle",
        )

    def create_proxy_service(
        self,
        proxy_client: BaseProxyClient,
        metrics: PrometheusMetrics | None = None,
    ) -> "ProxyService":
        """Factory method to create fully configured ProxyService.

        Creates ProxyService with all required dependencies.
        V2 plugins are managed separately via the FastAPI app lifecycle.

        Args:
            proxy_client: HTTP proxy client
            metrics: Optional metrics service

        Returns:
            Ready-to-use ProxyService instance with clean dependencies
        """
        # Import here to avoid circular dependency
        from ccproxy.services.proxy_service import ProxyService

        # Create ProxyService without old plugin system
        # Track components being initialized
        components = []

        request_tracer = self.get_request_tracer()
        components.append("request_tracer")

        mock_handler = self.get_mock_handler()
        components.append("mock_handler")

        streaming_handler = self.get_streaming_handler(metrics)
        components.append("streaming_handler")

        config = self.get_proxy_config()
        components.append("proxy_config")

        http_client = self.get_http_client()
        components.append("http_client")

        response_cache = self.get_response_cache()
        components.append("response_cache")

        connection_pool_manager = self.get_connection_pool_manager()
        components.append("connection_pool")

        proxy_service = ProxyService(
            proxy_client=proxy_client,
            settings=self.settings,
            request_tracer=request_tracer,
            mock_handler=mock_handler,
            streaming_handler=streaming_handler,
            config=config,
            http_client=http_client,
            metrics=metrics,
            response_cache=response_cache,
            connection_pool_manager=connection_pool_manager,
        )

        logger.info(
            "services_initialized",
            components=components,
            category="lifecycle",
        )
        return proxy_service

    def get_request_tracer(self) -> CoreRequestTracer:
        """Get request tracer service instance.

        Uses caching to ensure single instance per container lifetime,
        but allows multiple containers for testing.

        Returns:
            Request tracer service instance
        """
        service_key = "request_tracer"
        if service_key not in self._services:
            self._services[service_key] = self._factory.create_request_tracer(
                self.settings
            )
        return self._services[service_key]  # type: ignore

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
        metrics: PrometheusMetrics | None = None,
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
            pricing_service = self.get_pricing_service()
            self._services[service_key] = self._factory.create_streaming_handler(
                self.settings,
                metrics=metrics,
                request_tracer=request_tracer,
                pricing_service=pricing_service,
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

    async def close(self) -> None:
        """Close all managed resources during shutdown.

        This method properly cleans up all resources managed by the container,
        ensuring graceful shutdown and preventing resource leaks.
        """
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

    def get_pricing_service(self) -> PricingService:
        """Get pricing service instance.

        Uses caching to ensure single instance per container lifetime,
        but allows multiple containers for testing.

        Returns:
            Pricing service instance
        """
        service_key = "pricing_service"
        if service_key not in self._services:
            from plugins.pricing.config import PricingConfig
            from plugins.pricing.service import PricingService

            # Create pricing service with default config
            config = PricingConfig()
            self._services[service_key] = PricingService(config)
        return self._services[service_key]  # type: ignore
