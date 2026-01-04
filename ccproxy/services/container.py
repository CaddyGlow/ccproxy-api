"""Dependency injection container for all services.

This module provides a clean, testable dependency injection container that
manages service lifecycles and dependencies without singleton anti-patterns.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

import httpx
import structlog

from ccproxy.config.settings import Settings
from ccproxy.core.plugins.hooks.registry import HookRegistry
from ccproxy.core.plugins.hooks.thread_manager import BackgroundHookThreadManager
from ccproxy.http.pool import HTTPPoolManager
from ccproxy.scheduler.registry import TaskRegistry
from ccproxy.services.adapters.format_registry import FormatRegistry
from ccproxy.services.auth_registry import AuthManagerRegistry
from ccproxy.services.cache import ResponseCache
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.factories import ConcreteServiceFactory
from ccproxy.services.interfaces import (
    IRequestTracer,
    NullMetricsCollector,
    NullRequestTracer,
)
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.streaming import StreamingHandler
from ccproxy.utils.binary_resolver import BinaryResolver


if TYPE_CHECKING:
    from ccproxy.core.async_task_manager import AsyncTaskManager


logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ServiceContainer:
    """Dependency injection container for all services."""

    _current_container: ClassVar[ContextVar[ServiceContainer | None]] = ContextVar(
        "ccproxy_service_container",
        default=None,
    )
    _default_container: ClassVar[ServiceContainer | None] = None

    def __init__(self, settings: Settings) -> None:
        """Initialize the service container."""
        self.settings = settings
        self._services: dict[object, Any] = {}
        self._factories: dict[object, Callable[[], Any]] = {}

        self.register_service(Settings, self.settings)
        self.register_service(ServiceContainer, self)

        factory = ConcreteServiceFactory(self)
        factory.register_services()

        # Ensure a request tracer is always available for early consumers
        # Plugins may override this with a real tracer at runtime
        # Register a default tracer using the protocol as key
        self.register_service(IRequestTracer, instance=NullRequestTracer())

        # Make this container available for modules that resolve services globally
        self.activate()

    def activate(self, *, set_default: bool = True) -> None:
        """Mark this container as the current active container."""
        self.__class__._current_container.set(self)
        if set_default:
            self.__class__._default_container = self

    @classmethod
    def get_current(cls, *, strict: bool = True) -> ServiceContainer | None:
        """Return the currently active container.

        Args:
            strict: When True, raise an error if no container is active.

        Returns:
            Active service container or None.
        """

        container = cls._current_container.get()
        if container is None:
            container = cls._default_container
        if container is None and strict:
            raise RuntimeError("ServiceContainer is not available")
        return container

    def register_service(
        self,
        service_type: object,
        instance: Any | None = None,
        factory: Callable[[], Any] | None = None,
    ) -> None:
        """Register a service instance or factory."""
        if instance is not None:
            self._services[service_type] = instance
        elif factory is not None:
            self._factories[service_type] = factory
        else:
            raise ValueError("Either instance or factory must be provided")

    def get_service(self, service_type: type[T]) -> T:
        """Get a service instance by key (type or protocol)."""
        if service_type not in self._services:
            if service_type in self._factories:
                self._services[service_type] = self._factories[service_type]()
            else:
                # Best-effort name for error messages
                type_name = getattr(service_type, "__name__", str(service_type))
                raise ValueError(f"Service {type_name} not registered")
        return cast(T, self._services[service_type])

    def get_request_tracer(self) -> IRequestTracer:
        """Get request tracer service instance."""
        service = self._services.get(IRequestTracer)
        if service is None:
            raise ValueError("Service IRequestTracer not registered")
        return cast(IRequestTracer, service)

    def set_request_tracer(self, tracer: IRequestTracer) -> None:
        """Set the request tracer (called by plugin)."""
        self.register_service(IRequestTracer, instance=tracer)

    def get_mock_handler(self) -> MockResponseHandler:
        """Get mock handler service instance."""
        return self.get_service(MockResponseHandler)

    def get_streaming_handler(self) -> StreamingHandler:
        """Get streaming handler service instance."""
        return self.get_service(StreamingHandler)

    def get_binary_resolver(self) -> BinaryResolver:
        """Get binary resolver service instance."""
        return self.get_service(BinaryResolver)

    def get_cli_detection_service(self) -> CLIDetectionService:
        """Get CLI detection service instance."""
        return self.get_service(CLIDetectionService)

    def get_proxy_config(self) -> ProxyConfiguration:
        """Get proxy configuration service instance."""
        return self.get_service(ProxyConfiguration)

    def get_http_client(self) -> httpx.AsyncClient:
        """Get container-managed HTTP client instance."""
        return self.get_service(httpx.AsyncClient)

    def get_pool_manager(self) -> HTTPPoolManager:
        """Get HTTP connection pool manager instance."""
        return self.get_service(HTTPPoolManager)

    def get_response_cache(self) -> ResponseCache:
        """Get response cache service instance."""
        return self.get_service(ResponseCache)

    # Use HTTPPoolManager for pooling

    def get_format_registry(self) -> FormatRegistry:
        """Get format adapter registry service instance."""
        return self.get_service(FormatRegistry)

    # FormatterRegistry removed; use FormatRegistry exclusively.

    def get_oauth_registry(self) -> Any:
        """Get OAuth provider registry instance."""
        # Import lazily to avoid circular imports through auth package
        from ccproxy.auth.oauth.registry import OAuthRegistry

        return self.get_service(OAuthRegistry)

    def get_hook_registry(self) -> HookRegistry:
        """Get hook registry instance."""
        return self.get_service(HookRegistry)

    def get_task_registry(self) -> TaskRegistry:
        """Get scheduled task registry instance."""
        return self.get_service(TaskRegistry)

    def get_auth_manager_registry(self) -> AuthManagerRegistry:
        """Get auth manager registry instance."""
        return self.get_service(AuthManagerRegistry)

    def get_background_hook_thread_manager(self) -> BackgroundHookThreadManager:
        """Get background hook thread manager instance."""
        return self.get_service(BackgroundHookThreadManager)

    def get_async_task_manager(self) -> AsyncTaskManager:
        """Get async task manager instance."""
        from ccproxy.core.async_task_manager import AsyncTaskManager

        return self.get_service(AsyncTaskManager)

    def get_adapter_dependencies(self, metrics: Any | None = None) -> dict[str, Any]:
        """Get all services an adapter might need."""
        return {
            "http_client": self.get_http_client(),
            "request_tracer": self.get_request_tracer(),
            "metrics": metrics or NullMetricsCollector(),
            "streaming_handler": self.get_streaming_handler(),
            "logger": structlog.get_logger(),
            "config": self.get_proxy_config(),
            "cli_detection_service": self.get_cli_detection_service(),
            "format_registry": self.get_format_registry(),
        }

    async def close(self) -> None:
        """Close all managed resources during shutdown."""
        for service in list(self._services.values()):
            # Avoid recursive self-close
            if service is self:
                continue

            try:
                # Prefer aclose() if available (e.g., httpx.AsyncClient)
                if hasattr(service, "aclose") and callable(service.aclose):
                    maybe_coro = service.aclose()
                    if inspect.isawaitable(maybe_coro):
                        await maybe_coro
                elif hasattr(service, "close") and callable(service.close):
                    maybe_coro = service.close()
                    if inspect.isawaitable(maybe_coro):
                        await maybe_coro
                elif hasattr(service, "stop") and callable(service.stop):
                    stop_result = service.stop()
                    if inspect.isawaitable(stop_result):
                        await stop_result
                # else: nothing to close
            except Exception as e:
                logger.error(
                    "service_close_failed",
                    service=type(service).__name__,
                    error=str(e),
                    exc_info=e,
                    category="lifecycle",
                )
        self._services.clear()
        logger.debug("service_container_resources_closed", category="lifecycle")

    async def shutdown(self) -> None:
        """Shutdown all services in the container."""
        await self.close()
        if self.__class__._default_container is self:
            self.__class__._default_container = None
            self.__class__._current_container.set(None)
