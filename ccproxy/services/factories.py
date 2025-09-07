"""Concrete service factory implementations.

This module provides concrete implementations of service factories that
create and configure service instances according to their interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import httpx
import structlog

from ccproxy.adapters.openai.adapter import OpenAIAdapter
from ccproxy.config.settings import Settings
from ccproxy.http.client import HTTPClientFactory
from ccproxy.hooks import HookManager
from ccproxy.hooks.thread_manager import BackgroundHookThreadManager
from ccproxy.services.adapters.format_detector import FormatDetectionService
from ccproxy.services.adapters.format_registry import FormatAdapterRegistry
from ccproxy.services.cache import ResponseCache
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.services.config import ProxyConfiguration
from ccproxy.http.connection_pool import ConnectionPoolManager
from ccproxy.services.http_pool import HTTPPoolManager
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.services.streaming import StreamingHandler
from ccproxy.testing import RealisticMockResponseGenerator
from ccproxy.utils.binary_resolver import BinaryResolver


if TYPE_CHECKING:
    from ccproxy.services.container import ServiceContainer

logger = structlog.get_logger(__name__)


class ConcreteServiceFactory:
    """Concrete implementation of service factory."""

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize the service factory."""
        self._container = container

    def register_services(self) -> None:
        """Register all services with the container."""
        self._container.register_service(
            MockResponseHandler, factory=self.create_mock_handler
        )
        self._container.register_service(
            StreamingHandler, factory=self.create_streaming_handler
        )
        self._container.register_service(
            ProxyConfiguration, factory=self.create_proxy_config
        )
        self._container.register_service(
            httpx.AsyncClient, factory=self.create_http_client
        )
        self._container.register_service(
            CLIDetectionService, factory=self.create_cli_detection_service
        )
        self._container.register_service(
            HTTPPoolManager, factory=self.create_http_pool_manager
        )
        self._container.register_service(
            ResponseCache, factory=self.create_response_cache
        )
        self._container.register_service(
            ConnectionPoolManager, factory=self.create_connection_pool_manager
        )
        self._container.register_service(
            BinaryResolver, factory=self.create_binary_resolver
        )

        # NEW: Register format services
        self._container.register_service(
            FormatAdapterRegistry, factory=self.create_format_registry
        )
        self._container.register_service(
            FormatDetectionService, factory=self.create_format_detector
        )

        # Register background thread manager for hooks
        self._container.register_service(
            BackgroundHookThreadManager,
            factory=self.create_background_hook_thread_manager,
        )

    def create_mock_handler(self) -> MockResponseHandler:
        """Create mock handler instance."""
        mock_generator = RealisticMockResponseGenerator()
        openai_adapter = OpenAIAdapter()

        handler = MockResponseHandler(
            mock_generator=mock_generator,
            openai_adapter=openai_adapter,
            error_rate=0.05,
            latency_range=(0.5, 2.0),
        )
        return handler

    def create_streaming_handler(self) -> StreamingHandler:
        """Create streaming handler instance.

        Requires HookManager to be registered before resolution to avoid
        post-hoc patching of the handler.
        """
        hook_manager = self._container.get_service(HookManager)
        handler = StreamingHandler(hook_manager=hook_manager)
        return handler

    def create_proxy_config(self) -> ProxyConfiguration:
        """Create proxy configuration instance."""
        config = ProxyConfiguration()
        return config

    def create_http_client(self) -> httpx.AsyncClient:
        """Create HTTP client instance."""
        settings = self._container.get_service(Settings)
        hook_manager = self._container.get_service(HookManager)
        client = HTTPClientFactory.create_client(
            settings=settings, hook_manager=hook_manager
        )
        logger.debug("http_client_created", category="lifecycle")
        return client

    def create_cli_detection_service(self) -> CLIDetectionService:
        """Create CLI detection service instance."""
        settings = self._container.get_service(Settings)
        return CLIDetectionService(settings)

    def create_http_pool_manager(self) -> HTTPPoolManager:
        """Create HTTP pool manager instance."""
        settings = self._container.get_service(Settings)
        hook_manager = self._container.get_service(HookManager)
        logger.debug(
            "http_pool_manager_created",
            has_hook_manager=hook_manager is not None,
            hook_manager_type=type(hook_manager).__name__ if hook_manager else "None",
            category="lifecycle",
        )
        return HTTPPoolManager(settings, hook_manager)

    def create_response_cache(self) -> ResponseCache:
        """Create response cache instance."""
        return ResponseCache()

    def create_connection_pool_manager(self) -> ConnectionPoolManager:
        """Create connection pool manager instance."""
        # Use defaults from constants; tune here if settings add knobs later
        return ConnectionPoolManager()

    def create_binary_resolver(self) -> BinaryResolver:
        """Create a BinaryResolver from settings."""
        settings = self._container.get_service(Settings)
        return BinaryResolver.from_settings(settings)

    def create_format_registry(self) -> FormatAdapterRegistry:
        """Create format adapter registry with core adapters pre-registered.

        Pre-registers common format conversions to prevent plugin conflicts.
        Plugins can still register their own plugin-specific adapters.
        """
        settings = self._container.get_service(Settings)

        # Always use priority mode (latest behavior)
        conflict_mode: Literal["fail_fast", "priority"] = "priority"
        registry = FormatAdapterRegistry(conflict_mode=conflict_mode)

        # Pre-register core format adapters
        self._register_core_format_adapters(registry)

        logger.debug(
            "format_registry_created",
            conflict_mode=conflict_mode,
            category="format",
        )

        return registry

    def _register_core_format_adapters(self, registry: FormatAdapterRegistry) -> None:
        """Pre-register core format adapters with high priority."""
        from ccproxy.adapters.openai import AnthropicResponseAPIAdapter
        from ccproxy.adapters.openai.adapter import OpenAIAdapter

        # Core adapters that are always available
        core_adapters = {
            ("anthropic", "response_api"): AnthropicResponseAPIAdapter(),
            ("response_api", "anthropic"): AnthropicResponseAPIAdapter(),
            ("openai", "anthropic"): OpenAIAdapter(),
        }

        for format_pair, adapter in core_adapters.items():
            registry.register(format_pair[0], format_pair[1], adapter, "core")

        logger.debug(
            "core_format_adapters_registered",
            adapters=list(core_adapters.keys()),
            category="format",
        )

    def create_format_detector(self) -> FormatDetectionService:
        """Create format detection service."""
        return FormatDetectionService()

    def create_background_hook_thread_manager(self) -> BackgroundHookThreadManager:
        """Create background hook thread manager instance."""
        manager = BackgroundHookThreadManager()
        logger.debug("background_hook_thread_manager_created", category="lifecycle")
        return manager
