"""Concrete service factory implementations.

This module provides concrete implementations of service factories that
create and configure service instances according to their interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import httpx
import structlog

from ccproxy.auth.oauth import registry as oauth_registry_module
from ccproxy.config.settings import Settings
from ccproxy.core.plugins.hooks import HookManager
from ccproxy.core.plugins.hooks.registry import HookRegistry
from ccproxy.core.plugins.hooks.thread_manager import BackgroundHookThreadManager
from ccproxy.http.client import HTTPClientFactory
from ccproxy.http.pool import HTTPPoolManager
from ccproxy.scheduler.registry import TaskRegistry
from ccproxy.services.adapters.format_adapter import DictFormatAdapter
from ccproxy.services.adapters.format_registry import FormatRegistry
from ccproxy.services.adapters.simple_converters import (
    convert_anthropic_to_openai_response,
)
from ccproxy.services.auth_registry import AuthManagerRegistry
from ccproxy.services.cache import ResponseCache
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.streaming import StreamingHandler
from ccproxy.testing import RealisticMockResponseGenerator
from ccproxy.utils.binary_resolver import BinaryResolver


if TYPE_CHECKING:
    from ccproxy.core.async_task_manager import AsyncTaskManager
    from ccproxy.services.container import ServiceContainer

logger = structlog.get_logger(__name__)


class _CoreAdapterSpec(TypedDict):
    """Type definition for core adapter specification dictionary."""

    from_format: str
    to_format: str
    request: Any  # Format converter function
    response: Any  # Format converter function
    stream: Any  # Format converter function
    error: Any  # Error converter function
    name: str


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
            FormatRegistry, factory=self.create_format_registry
        )

        # Registries
        self._container.register_service(
            HookRegistry, factory=self.create_hook_registry
        )

        self._container.register_service(
            oauth_registry_module.OAuthRegistry, factory=self.create_oauth_registry
        )
        self._container.register_service(
            TaskRegistry, factory=self.create_task_registry
        )
        self._container.register_service(
            AuthManagerRegistry, factory=self.create_auth_manager_registry
        )

        # Register background thread manager for hooks
        self._container.register_service(
            BackgroundHookThreadManager,
            factory=self.create_background_hook_thread_manager,
        )
        from ccproxy.core.async_task_manager import AsyncTaskManager

        self._container.register_service(
            AsyncTaskManager,
            factory=self.create_async_task_manager,
        )

    def create_mock_handler(self) -> MockResponseHandler:
        """Create mock handler instance."""
        mock_generator = RealisticMockResponseGenerator()
        settings = self._container.get_service(Settings)
        # Create simple format adapter for anthropic->openai conversion (for mock responses)
        openai_adapter = DictFormatAdapter(
            response=convert_anthropic_to_openai_response,
            name="mock_anthropic_to_openai",
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

    def create_format_registry(self) -> FormatRegistry:
        """Create format adapter registry with core adapters pre-registered.

        Pre-registers common format conversions to prevent plugin conflicts.
        Plugins can still register their own plugin-specific adapters.
        """
        settings = self._container.get_service(Settings)

        # Always use priority mode (latest behavior)
        registry = FormatRegistry()

        # Pre-register core format adapters
        self._register_core_format_adapters(registry, settings)
        registry.flush_all_logs()

        logger.debug(
            "format_registry_created",
            category="format",
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

    def create_auth_manager_registry(self) -> AuthManagerRegistry:
        """Create an AuthManagerRegistry instance.

        Note: Auth managers are registered by their respective plugins during initialization.
        """
        return AuthManagerRegistry()

    def _register_core_format_adapters(
        self, registry: FormatRegistry, settings: Settings | None = None
    ) -> None:
        """Register essential format adapters provided by core.

        Registers commonly-needed format conversions to prevent plugin duplication
        and ensure required adapters are available for plugin dependencies.
        """
        from ccproxy.core.constants import (
            FORMAT_ANTHROPIC_MESSAGES,
            FORMAT_OPENAI_CHAT,
            FORMAT_OPENAI_RESPONSES,
        )
        from ccproxy.services.adapters.simple_converters import (
            convert_anthropic_to_openai_error,
            convert_anthropic_to_openai_request,
            convert_anthropic_to_openai_response,
            convert_anthropic_to_openai_responses_error,
            convert_anthropic_to_openai_responses_request,
            convert_anthropic_to_openai_responses_response,
            convert_anthropic_to_openai_responses_stream,
            convert_anthropic_to_openai_stream,
            convert_openai_chat_to_openai_responses_error,
            convert_openai_chat_to_openai_responses_request,
            convert_openai_chat_to_openai_responses_response,
            convert_openai_chat_to_openai_responses_stream,
            convert_openai_responses_to_anthropic_error,
            convert_openai_responses_to_anthropic_request,
            convert_openai_responses_to_anthropic_response,
            convert_openai_responses_to_anthropic_stream,
            convert_openai_responses_to_openai_chat_error,
            convert_openai_responses_to_openai_chat_request,
            convert_openai_responses_to_openai_chat_response,
            convert_openai_responses_to_openai_chat_stream,
            convert_openai_to_anthropic_error,
            convert_openai_to_anthropic_request,
            convert_openai_to_anthropic_response,
            convert_openai_to_anthropic_stream,
        )

        # Define core format adapter specifications
        core_adapter_specs: list[_CoreAdapterSpec] = [
            # Most commonly required: Anthropic ↔ OpenAI Responses
            {
                "from_format": FORMAT_ANTHROPIC_MESSAGES,
                "to_format": FORMAT_OPENAI_RESPONSES,
                "request": convert_anthropic_to_openai_responses_request,
                "response": convert_anthropic_to_openai_responses_response,
                "stream": convert_anthropic_to_openai_responses_stream,
                "error": convert_anthropic_to_openai_responses_error,
                "name": "core_anthropic_to_openai_responses",
            },
            {
                "from_format": FORMAT_OPENAI_RESPONSES,
                "to_format": FORMAT_ANTHROPIC_MESSAGES,
                "request": convert_openai_responses_to_anthropic_request,
                "response": convert_openai_responses_to_anthropic_response,
                "stream": convert_openai_responses_to_anthropic_stream,
                "error": convert_openai_responses_to_anthropic_error,
                "name": "core_openai_responses_to_anthropic",
            },
            # OpenAI Chat ↔ Responses (needed by Codex plugin)
            {
                "from_format": FORMAT_OPENAI_CHAT,
                "to_format": FORMAT_OPENAI_RESPONSES,
                "request": convert_openai_chat_to_openai_responses_request,
                "response": convert_openai_chat_to_openai_responses_response,
                "stream": convert_openai_chat_to_openai_responses_stream,
                "error": convert_openai_chat_to_openai_responses_error,
                "name": "core_openai_chat_to_responses",
            },
            # Reverse: OpenAI Responses -> OpenAI Chat
            {
                "from_format": FORMAT_OPENAI_RESPONSES,
                "to_format": FORMAT_OPENAI_CHAT,
                "request": convert_openai_responses_to_openai_chat_request,
                "response": convert_openai_responses_to_openai_chat_response,
                "stream": convert_openai_responses_to_openai_chat_stream,
                "error": convert_openai_responses_to_openai_chat_error,
                "name": "core_openai_responses_to_chat",
            },
            # Anthropic ↔ OpenAI Chat (commonly needed for proxying)
            {
                "from_format": FORMAT_ANTHROPIC_MESSAGES,
                "to_format": FORMAT_OPENAI_CHAT,
                "request": convert_anthropic_to_openai_request,
                "response": convert_anthropic_to_openai_response,
                "stream": convert_anthropic_to_openai_stream,
                "error": convert_anthropic_to_openai_error,
                "name": "core_anthropic_to_openai_chat",
            },
            # Reverse: OpenAI Chat -> Anthropic
            {
                "from_format": FORMAT_OPENAI_CHAT,
                "to_format": FORMAT_ANTHROPIC_MESSAGES,
                "request": convert_openai_to_anthropic_request,
                "response": convert_openai_to_anthropic_response,
                "stream": convert_openai_to_anthropic_stream,
                "error": convert_openai_to_anthropic_error,
                "name": "core_openai_chat_to_anthropic",
            },
        ]

        # Register each core adapter
        for spec in core_adapter_specs:
            adapter = DictFormatAdapter(
                request=spec["request"],
                response=spec["response"],
                stream=spec["stream"],
                error=spec["error"],
                name=spec["name"],
            )
            registry.register(
                from_format=spec["from_format"],
                to_format=spec["to_format"],
                adapter=adapter,
                plugin_name="core",
            )

        logger.debug(
            "core_format_adapters_registered",
            count=len(core_adapter_specs),
            adapters=[
                f"{spec['from_format']}->{spec['to_format']}"
                for spec in core_adapter_specs
            ],
            category="format",
        )

    def create_background_hook_thread_manager(self) -> BackgroundHookThreadManager:
        """Create background hook thread manager instance."""
        manager = BackgroundHookThreadManager()
        logger.debug("background_hook_thread_manager_created", category="lifecycle")
        return manager

    def create_async_task_manager(self) -> AsyncTaskManager:
        """Create async task manager instance."""
        from ccproxy.core.async_task_manager import AsyncTaskManager

        return AsyncTaskManager()
