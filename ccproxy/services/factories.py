"""Concrete service factory implementations.

This module provides concrete implementations of service factories that
create and configure service instances according to their interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import httpx
import structlog

from ccproxy.config.settings import Settings
from ccproxy.core.plugins.hooks import HookManager
from ccproxy.core.plugins.hooks.registry import HookRegistry
from ccproxy.core.plugins.hooks.thread_manager import BackgroundHookThreadManager
from ccproxy.http.client import HTTPClientFactory
from ccproxy.http.pool import HTTPPoolManager
from ccproxy.llms.formatters.formatter_adapter import FormatterRegistryAdapter
from ccproxy.llms.formatters.formatter_registry import (
    FormatterRegistry,
    iter_registered_formatters,
    load_builtin_formatter_modules,
)
from ccproxy.scheduler.registry import TaskRegistry
from ccproxy.services.adapters.format_registry import FormatAdapterRegistry
from ccproxy.services.cache import ResponseCache
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.streaming import StreamingHandler
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
            BinaryResolver, factory=self.create_binary_resolver
        )

        self._container.register_service(
            FormatAdapterRegistry, factory=self.create_format_registry
        )
        self._container.register_service(
            FormatterRegistry, factory=self.create_formatter_registry
        )

        # Registries
        self._container.register_service(
            HookRegistry, factory=self.create_hook_registry
        )
        # Delay import of OAuthRegistry to avoid circular import via auth package
        from ccproxy.auth.oauth import registry as oauth_registry_module

        self._container.register_service(
            oauth_registry_module.OAuthRegistry, factory=self.create_oauth_registry
        )
        self._container.register_service(
            TaskRegistry, factory=self.create_task_registry
        )

        # Register background thread manager for hooks
        self._container.register_service(
            BackgroundHookThreadManager,
            factory=self.create_background_hook_thread_manager,
        )

    def create_mock_handler(self) -> MockResponseHandler:
        """Create mock handler instance."""
        mock_generator = RealisticMockResponseGenerator()
        settings = self._container.get_service(Settings)
        # Create formatter adapter for anthropic->openai conversion (for mock responses)
        formatter_registry = FormatterRegistry()
        load_builtin_formatter_modules()  # Load global formatters
        # Populate registry from global static registrations
        for registration in iter_registered_formatters():
            formatter_registry.register(
                source_format=registration.source_format,
                target_format=registration.target_format,
                operation=registration.operation,
                formatter=registration.formatter,
                module_name=getattr(registration.formatter, "__module__", None),
            )
        openai_adapter = FormatterRegistryAdapter(
            formatter_registry=formatter_registry,
            source_format="anthropic.messages",
            target_format="openai.chat_completions",
        )
        # Configure streaming settings if needed
        openai_thinking_xml = getattr(
            getattr(settings, "llm", object()), "openai_thinking_xml", True
        )
        if hasattr(openai_adapter, "configure_streaming"):
            openai_adapter.configure_streaming(openai_thinking_xml=openai_thinking_xml)

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

    # ConnectionPoolManager is no longer used; HTTPPoolManager only

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
        self._register_core_format_adapters(registry, settings)

        logger.debug(
            "format_registry_created",
            conflict_mode=conflict_mode,
            category="format",
        )

        return registry

    def create_formatter_registry(self) -> FormatterRegistry:
        """Create formatter registry populated from helper modules."""

        load_builtin_formatter_modules()

        registry = FormatterRegistry()
        for registration in iter_registered_formatters():
            try:
                registry.register(
                    registration.source_format,
                    registration.target_format,
                    registration.operation,
                    registration.formatter,
                    registration.module_name,
                )
            except ValueError as exc:
                logger.warning(
                    "formatter_registration_conflict",
                    source_format=registration.source_format,
                    target_format=registration.target_format,
                    operation=registration.operation,
                    module=registration.module_name,
                    error=str(exc),
                    category="formatter",
                )

        logger.debug(
            "formatter_registry_created",
            total=len(registry.list()),
            category="formatter",
        )

        return registry

    def create_hook_registry(self) -> HookRegistry:
        """Create a HookRegistry instance."""
        return HookRegistry()

    def create_oauth_registry(self) -> Any:
        """Create an OAuthRegistry instance (imported lazily to avoid cycles)."""
        from ccproxy.auth.oauth.registry import OAuthRegistry

        return OAuthRegistry()

    def create_task_registry(self) -> TaskRegistry:
        """Create a TaskRegistry instance."""
        return TaskRegistry()

    def _register_core_format_adapters(
        self, registry: FormatAdapterRegistry, settings: Settings | None = None
    ) -> None:
        pass

    def create_background_hook_thread_manager(self) -> BackgroundHookThreadManager:
        """Create background hook thread manager instance."""
        manager = BackgroundHookThreadManager()
        logger.debug("background_hook_thread_manager_created", category="lifecycle")
        return manager
