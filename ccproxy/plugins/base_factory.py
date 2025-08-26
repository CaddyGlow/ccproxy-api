"""Base provider plugin factory to eliminate boilerplate code.

This module provides a BaseProviderPluginFactory that implements common patterns
shared across all provider plugin factories, reducing code duplication by 60-70%.
"""

from typing import Any

from fastapi import APIRouter

from ccproxy.services.adapters.base import BaseAdapter

from .declaration import PluginContext, PluginManifest, RouteSpec, TaskSpec
from .factory import ProviderPluginFactory
from .runtime import ProviderPluginRuntime


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

    def __init__(self) -> None:
        """Initialize factory with manifest built from class attributes."""
        # Validate required class attributes
        self._validate_class_attributes()

        # Build routes from router if provided
        routes = []
        if self.router is not None:
            routes.append(
                RouteSpec(
                    router=self.router,
                    prefix=self.route_prefix,
                    tags=[f"plugin-{self.plugin_name}"],
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
        )

        # Initialize parent with manifest
        super().__init__(manifest)

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

    def create_runtime(self) -> ProviderPluginRuntime:
        """Create runtime instance using the configured runtime class."""
        return self.runtime_class(self.manifest)

    def create_adapter(self, context: PluginContext) -> BaseAdapter:
        """Create adapter instance with common parameter extraction.

        This method extracts common parameters from context and creates
        the adapter. Subclasses can override this method if they need
        custom adapter creation logic.

        Args:
            context: Plugin context

        Returns:
            Adapter instance
        """
        # Extract common parameters
        proxy_service = context.get("proxy_service")
        http_client = context.get("http_client")
        logger_instance = context.get("logger")

        # Get optional components that may have been created by factory
        detection_service = context.get("detection_service")
        credentials_manager = context.get("credentials_manager")

        # Build adapter kwargs with common parameters
        adapter_kwargs = {
            "proxy_service": proxy_service,
            "http_client": http_client,
            "logger": logger_instance,
            "context": context,
        }

        # Add auth_manager if credentials_manager exists
        if credentials_manager is not None:
            adapter_kwargs["auth_manager"] = credentials_manager

        # Add detection_service if it exists
        if detection_service is not None:
            adapter_kwargs["detection_service"] = detection_service

        return self.adapter_class(**adapter_kwargs)

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
