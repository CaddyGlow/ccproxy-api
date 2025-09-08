"""Base provider plugin factory to eliminate boilerplate code.

This module provides a BaseProviderPluginFactory that implements common patterns
shared across all provider plugin factories, reducing code duplication by 60-70%.
"""

from typing import TYPE_CHECKING, Any, cast

import httpx
from fastapi import APIRouter

from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.services.interfaces import (
    IMetricsCollector,
    IRequestTracer,
    NullMetricsCollector,
    NullRequestTracer,
    NullStreamingHandler,
    StreamingMetrics,
)

from .declaration import (
    FormatAdapterSpec,
    FormatPair,
    PluginContext,
    PluginManifest,
    RouteSpec,
    TaskSpec,
)
from .factory import ProviderPluginFactory


if TYPE_CHECKING:
    from ccproxy.config.settings import Settings
    from ccproxy.http.pool import HTTPPoolManager


class BaseProviderPluginFactory(ProviderPluginFactory):
    """Base factory for provider plugins that eliminates common boilerplate.

    This class uses class attributes for plugin configuration and implements
    common methods that all provider factories share. Subclasses only need
    to define class attributes and override methods that need custom behavior.

    Required class attributes to be defined by subclasses:
    - plugin_name: str
    - plugin_description: str
    - runtime_class: type[ProviderPluginRuntime]
    - adapter_class: type[BaseAdapter]
    - config_class: type[BaseSettings]

    Optional class attributes with defaults:
    - plugin_version: str = "1.0.0"
    - detection_service_class: type | None = None
    - credentials_manager_class: type | None = None
    - router: APIRouter | None = None
    - route_prefix: str = "/api"
    - dependencies: list[str] = []
    - optional_requires: list[str] = []
    - tasks: list[TaskSpec] = []
    """

    # Required class attributes (must be overridden by subclasses)
    plugin_name: str
    plugin_description: str
    runtime_class: Any  # Should be type[ProviderPluginRuntime] subclass
    adapter_class: Any  # Should be type[BaseAdapter] subclass
    config_class: Any  # Should be type[BaseSettings] subclass

    # Optional class attributes with defaults
    plugin_version: str = "1.0.0"
    detection_service_class: type | None = None
    credentials_manager_class: type | None = None
    router: APIRouter | None = None
    route_prefix: str = "/api"
    dependencies: list[str] = []
    optional_requires: list[str] = []
    tasks: list[TaskSpec] = []

    # Format adapter declarations (populated by subclasses)
    format_adapters: list[FormatAdapterSpec] = []
    requires_format_adapters: list[FormatPair] = []

    def __init__(self) -> None:
        """Initialize factory with manifest built from class attributes."""
        # Validate required class attributes
        self._validate_class_attributes()

        # Validate runtime class is a proper subclass
        # Import locally to avoid circular import during module import
        from ccproxy.core.plugins import ProviderPluginRuntime

        if not issubclass(self.runtime_class, ProviderPluginRuntime):
            raise TypeError(
                f"runtime_class {self.runtime_class.__name__} must be a subclass of ProviderPluginRuntime"
            )

        # Build routes from router if provided
        routes = []
        if self.router is not None:
            # Normalize tag naming: use kebab-case (underscores -> hyphens)
            normalized_tag = self.plugin_name.replace("_", "-")
            routes.append(
                RouteSpec(
                    router=self.router,
                    prefix=self.route_prefix,
                    tags=[normalized_tag],
                )
            )

        # Create manifest from class attributes
        manifest = PluginManifest(
            name=self.plugin_name,
            version=self.plugin_version,
            description=self.plugin_description,
            is_provider=True,
            config_class=self.config_class,
            dependencies=self.dependencies.copy(),
            optional_requires=self.optional_requires.copy(),
            routes=routes,
            tasks=self.tasks.copy(),
            format_adapters=self.format_adapters.copy(),
            requires_format_adapters=self.requires_format_adapters.copy(),
        )

        # Format adapter specification validation is deferred to runtime
        # when settings are available via dependency injection

        # Store the manifest and runtime class directly
        # We don't call parent __init__ because ProviderPluginFactory
        # would override our runtime_class with ProviderPluginRuntime
        self.manifest = manifest
        self.runtime_class = self.__class__.runtime_class

    def validate_format_adapters_with_settings(self, settings: "Settings") -> None:
        """Validate format adapter specifications (feature flags removed)."""
        self._validate_format_adapter_specs()

    def _validate_class_attributes(self) -> None:
        """Validate that required class attributes are defined."""
        required_attrs = [
            "plugin_name",
            "plugin_description",
            "runtime_class",
            "adapter_class",
            "config_class",
        ]

        for attr in required_attrs:
            if (
                not hasattr(self.__class__, attr)
                or getattr(self.__class__, attr) is None
            ):
                raise ValueError(
                    f"Class attribute '{attr}' must be defined in {self.__class__.__name__}"
                )

    def _validate_format_adapter_specs(self) -> None:
        """Validate format adapter specifications."""
        for spec in self.format_adapters:
            if not callable(spec.adapter_factory):
                raise ValueError(
                    f"Invalid adapter factory for {spec.from_format} -> {spec.to_format}: "
                    f"must be callable"
                ) from None

    def create_runtime(self) -> Any:
        """Create runtime instance using the configured runtime class."""
        return cast(Any, self.runtime_class(self.manifest))

    async def create_adapter(self, context: PluginContext) -> BaseAdapter:
        """Create adapter instance with explicit dependencies.

        This method extracts services from context and creates the adapter
        with explicit dependency injection. Subclasses can override this
        method if they need custom adapter creation logic.

        Args:
            context: Plugin context

        Returns:
            Adapter instance
        """
        # Extract services from context (one-time extraction)
        http_pool_manager: HTTPPoolManager | None = cast(
            "HTTPPoolManager | None", context.get("http_pool_manager")
        )
        request_tracer: IRequestTracer | None = context.get("request_tracer")
        metrics: IMetricsCollector | None = context.get("metrics")
        streaming_handler: StreamingMetrics | None = context.get("streaming_handler")
        hook_manager = context.get("hook_manager")

        # Get auth and detection services that may have been created by factory
        auth_manager = context.get("credentials_manager")
        detection_service = context.get("detection_service")

        # Get config if available
        config = context.get("config")

        # Get all adapter dependencies from service container
        service_container = context.get("service_container")
        if not service_container:
            raise RuntimeError("Service container is required for adapter services")

        # Get standardized adapter dependencies
        adapter_dependencies = service_container.get_adapter_dependencies(metrics)

        # Check if this is an HTTP-based adapter
        if issubclass(self.adapter_class, BaseHTTPAdapter):
            # HTTP adapters require http_pool_manager
            if not http_pool_manager:
                raise RuntimeError(
                    f"HTTP pool manager required for {self.adapter_class.__name__} but not available in context"
                )

            # Create HTTP adapter with explicit dependencies including format services
            return cast(
                BaseAdapter,
                self.adapter_class(
                    auth_manager=auth_manager,
                    detection_service=detection_service,
                    http_pool_manager=http_pool_manager,
                    request_tracer=request_tracer or NullRequestTracer(),
                    metrics=metrics or NullMetricsCollector(),
                    streaming_handler=streaming_handler or NullStreamingHandler(),
                    hook_manager=hook_manager,
                    format_registry=adapter_dependencies["format_registry"],
                    format_detector=adapter_dependencies["format_detector"],
                    context=context,
                ),
            )
        else:
            # Non-HTTP adapters (like ClaudeSDK) have different dependencies
            # Build kwargs based on adapter class constructor signature
            import inspect

            adapter_kwargs: dict[str, Any] = {}

            # Get the adapter's __init__ signature
            sig = inspect.signature(self.adapter_class.__init__)
            params = sig.parameters

            # For non-HTTP adapters, create http_client from pool manager if needed
            client_for_non_http: httpx.AsyncClient | None = None
            if http_pool_manager and "http_client" in params:
                client_for_non_http = await http_pool_manager.get_client()

            # Map available services to expected parameters
            param_mapping = {
                "config": config,
                "http_client": client_for_non_http,
                "http_pool_manager": http_pool_manager,
                "auth_manager": auth_manager,
                "detection_service": detection_service,
                "session_manager": context.get("session_manager"),
                "request_tracer": request_tracer,
                "metrics": metrics,
                "streaming_handler": streaming_handler,
                "hook_manager": hook_manager,
                "format_registry": adapter_dependencies["format_registry"],
                "format_detector": adapter_dependencies["format_detector"],
                "context": context,
            }

            # Add parameters that the adapter expects
            for param_name, param in params.items():
                if param_name in ("self", "kwargs"):
                    continue
                if (
                    param_name in param_mapping
                    and param_mapping[param_name] is not None
                ):
                    adapter_kwargs[param_name] = param_mapping[param_name]
                elif (
                    param.default is inspect.Parameter.empty
                    and param_name not in adapter_kwargs
                    and param_name == "config"
                    and config is None
                    and self.manifest.config_class
                ):
                    # Try to get config from manifest
                    config = self.manifest.config_class()
                    adapter_kwargs["config"] = config

            return cast(BaseAdapter, self.adapter_class(**adapter_kwargs))

    def create_detection_service(self, context: PluginContext) -> Any:
        """Create detection service instance if class is configured.

        Args:
            context: Plugin context

        Returns:
            Detection service instance or None if no class configured
        """
        if self.detection_service_class is None:
            return None

        settings = context.get("settings")
        if settings is None:
            from ccproxy.config.settings import Settings

            settings = Settings()

        cli_service = context.get("cli_detection_service")
        return self.detection_service_class(settings, cli_service)

    def create_credentials_manager(self, context: PluginContext) -> Any:
        """Create credentials manager instance if class is configured.

        Args:
            context: Plugin context

        Returns:
            Credentials manager instance or None if no class configured
        """
        if self.credentials_manager_class is None:
            return None

        return self.credentials_manager_class()

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context with provider-specific components.

        This method provides a hook for subclasses to customize context creation.
        The default implementation just returns the base context.

        Args:
            core_services: Core services container

        Returns:
            Plugin context
        """
        return super().create_context(core_services)
