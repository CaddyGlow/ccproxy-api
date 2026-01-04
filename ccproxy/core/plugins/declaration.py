"""Plugin declaration system for static plugin specification.

This module provides the declaration layer of the plugin system, allowing plugins
to specify their requirements and capabilities at declaration time (app creation)
rather than runtime (lifespan).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import httpx
import structlog
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from ccproxy.services.adapters.format_adapter import FormatAdapterProtocol


if TYPE_CHECKING:
    from ccproxy.auth.oauth.registry import OAuthRegistry
    from ccproxy.config.settings import Settings
    from ccproxy.core.plugins import PluginRegistry
    from ccproxy.core.plugins.hooks.base import Hook
    from ccproxy.core.plugins.hooks.manager import HookManager
    from ccproxy.core.plugins.hooks.registry import HookRegistry
    from ccproxy.core.plugins.protocol import OAuthClientProtocol
    from ccproxy.scheduler.core import Scheduler
    from ccproxy.scheduler.tasks import BaseScheduledTask
    from ccproxy.services.adapters.base import BaseAdapter
    from ccproxy.services.cli_detection import CLIDetectionService
    from ccproxy.services.interfaces import (
        IMetricsCollector,
        IRequestTracer,
        StreamingMetrics,
    )

T = TypeVar("T")

# Type aliases for format adapter system
FormatPair = tuple[str, str]


@dataclass
class FormatAdapterSpec:
    """Specification for format adapter registration."""

    from_format: str
    to_format: str
    adapter_factory: Callable[
        [], FormatAdapterProtocol | Awaitable[FormatAdapterProtocol]
    ]
    priority: int = 100  # Lower = higher priority for conflict resolution
    description: str = ""

    def __post_init__(self) -> None:
        """Validate specification."""
        if not self.from_format or not self.to_format:
            raise ValueError("Format names cannot be empty") from None
        if self.from_format == self.to_format:
            raise ValueError("from_format and to_format cannot be the same") from None

    @property
    def format_pair(self) -> FormatPair:
        """Get the format pair tuple."""
        return (self.from_format, self.to_format)


class MiddlewareLayer(IntEnum):
    """Middleware layers for ordering."""

    SECURITY = 100  # Authentication, rate limiting
    OBSERVABILITY = 200  # Logging, metrics
    TRANSFORMATION = 300  # Compression, encoding
    ROUTING = 400  # Path rewriting, proxy
    APPLICATION = 500  # Business logic


@dataclass
class MiddlewareSpec:
    """Specification for plugin middleware."""

    middleware_class: type[BaseHTTPMiddleware]
    priority: int = MiddlewareLayer.APPLICATION
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "MiddlewareSpec") -> bool:
        """Sort by priority (lower values first)."""
        return self.priority < other.priority


@dataclass
class RouterSpec:
    """Specification for individual routers in a plugin."""

    router: APIRouter | Callable[[], APIRouter]
    prefix: str
    tags: list[str] = field(default_factory=list)
    dependencies: list[Any] = field(default_factory=list)


@dataclass
class RouteSpec:
    """Specification for plugin routes."""

    router: APIRouter
    prefix: str
    tags: list[str] = field(default_factory=list)
    dependencies: list[Any] = field(default_factory=list)


@dataclass
class TaskSpec:
    """Specification for scheduled tasks."""

    task_name: str
    task_type: str
    task_class: type["BaseScheduledTask"]  # BaseScheduledTask type from scheduler.tasks
    interval_seconds: float
    enabled: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookSpec:
    """Specification for plugin hooks."""

    hook_class: type["Hook"]  # Hook type from hooks.base
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthCommandSpec:
    """Specification for auth commands."""

    command_name: str
    description: str
    handler: Callable[..., Any]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class CliCommandSpec:
    """Specification for plugin CLI commands."""

    command_name: str
    command_function: Callable[..., Any]
    help_text: str = ""
    parent_command: str | None = None  # For subcommands like "auth login-myservice"

    def __post_init__(self) -> None:
        """Validate CLI command specification."""
        if not self.command_name:
            raise ValueError("command_name cannot be empty") from None
        if not callable(self.command_function):
            raise ValueError("command_function must be callable") from None


@dataclass
class CliArgumentSpec:
    """Specification for adding arguments to existing commands."""

    target_command: str  # e.g., "serve", "auth"
    argument_name: str
    argument_type: type = str
    help_text: str = ""
    default: Any = None
    required: bool = False
    typer_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate CLI argument specification."""
        if not self.target_command:
            raise ValueError("target_command cannot be empty") from None
        if not self.argument_name:
            raise ValueError("argument_name cannot be empty") from None


@dataclass
class PluginManifest:
    """Complete static declaration of a plugin's capabilities.

    This manifest is created at module import time and contains all
    static information needed to integrate the plugin into the application.
    """

    # Basic metadata
    name: str
    version: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)

    # Plugin type
    is_provider: bool = False  # True for provider plugins, False for system plugins

    # Service declarations
    provides: list[str] = field(default_factory=list)  # Services this plugin provides
    requires: list[str] = field(default_factory=list)  # Required service dependencies
    optional_requires: list[str] = field(
        default_factory=list
    )  # Optional service dependencies

    # Static specifications
    middleware: list[MiddlewareSpec] = field(default_factory=list)
    routes: list[RouteSpec] = field(default_factory=list)
    tasks: list[TaskSpec] = field(default_factory=list)
    hooks: list[HookSpec] = field(default_factory=list)
    auth_commands: list[AuthCommandSpec] = field(default_factory=list)

    # Configuration
    config_class: type[BaseModel] | None = None
    tool_accumulator_class: type | None = None

    # OAuth support (for provider plugins)
    oauth_client_factory: Callable[[], "OAuthClientProtocol"] | None = (
        None  # Returns OAuthClientProtocol
    )
    oauth_provider_factory: Callable[[], Any] | None = (
        None  # Returns OAuthProviderProtocol
    )
    token_manager_factory: Callable[[], Any] | None = (
        None  # Returns TokenManager for the provider
    )
    oauth_config_class: type[BaseModel] | None = None  # OAuth configuration model
    oauth_routes: list[RouteSpec] = field(
        default_factory=list
    )  # Plugin-specific OAuth routes

    # Format adapter declarations
    format_adapters: list[FormatAdapterSpec] = field(default_factory=list)
    requires_format_adapters: list[FormatPair] = field(default_factory=list)

    # CLI extensions
    cli_commands: list[CliCommandSpec] = field(default_factory=list)
    cli_arguments: list[CliArgumentSpec] = field(default_factory=list)

    def validate_dependencies(self, available_plugins: set[str]) -> list[str]:
        """Validate that all dependencies are available.

        Args:
            available_plugins: Set of available plugin names

        Returns:
            List of missing dependencies
        """
        return [dep for dep in self.dependencies if dep not in available_plugins]

    def validate_service_dependencies(self, available_services: set[str]) -> list[str]:
        """Validate that required services are available.

        Args:
            available_services: Set of available service names

        Returns:
            List of missing required services
        """
        missing = []
        for required in self.requires:
            if required not in available_services:
                missing.append(required)
        return missing

    def get_sorted_middleware(self) -> list[MiddlewareSpec]:
        """Get middleware sorted by priority."""
        return sorted(self.middleware)

    def validate_format_adapter_requirements(
        self, available_adapters: set[FormatPair]
    ) -> list[FormatPair]:
        """Validate that required format adapters are available."""
        return [
            req
            for req in self.requires_format_adapters
            if req not in available_adapters
        ]


class PluginContext:
    """Context provided to plugin runtime during initialization."""

    def __init__(self) -> None:
        """Initialize plugin context."""
        # Application settings
        self.settings: Settings | None = None
        self.http_client: httpx.AsyncClient | None = None
        self.logger: structlog.BoundLogger | None = None
        self.scheduler: Scheduler | None = None
        self.config: BaseModel | None = None
        self.cli_detection_service: CLIDetectionService | None = None
        self.plugin_registry: PluginRegistry | None = None

        # Core app and hook system
        self.app: FastAPI | None = None
        self.hook_registry: HookRegistry | None = None
        self.hook_manager: HookManager | None = None

        # Observability and streaming
        self.request_tracer: IRequestTracer | None = None
        self.streaming_handler: StreamingMetrics | None = None
        self.metrics: IMetricsCollector | None = None

        # Provider-specific
        self.adapter: BaseAdapter | None = None
        self.detection_service: Any = None
        self.credentials_manager: Any = None
        self.oauth_registry: OAuthRegistry | None = None
        self.http_pool_manager: Any = None
        self.service_container: Any = None
        self.auth_provider: Any = None
        self.token_manager: Any = None
        self.storage: Any = None

        self.format_registry: Any = None
        self.model_mapper: Any = None

        # Testing/utilities
        self.proxy_service: Any = None

        # Internal service mapping for type-safe access
        self._service_map: dict[type[Any], str] = {}
        self._initialize_service_map()

    def _initialize_service_map(self) -> None:
        """Initialize the service type mapping."""
        if TYPE_CHECKING:
            pass

        # Map service types to their attribute names
        self._service_map = {
            # Core services - using Any to avoid circular imports at runtime
            **(
                {}
                if TYPE_CHECKING
                else {
                    type(None): "settings",  # Placeholder, will be populated at runtime
                }
            ),
            httpx.AsyncClient: "http_client",
            structlog.BoundLogger: "logger",
            BaseModel: "config",
        }

    def get_service(self, service_type: type[T]) -> T:
        """Get a service instance by type with proper type safety.

        Args:
            service_type: The type of service to retrieve

        Returns:
            The service instance

        Raises:
            ValueError: If the service is not available
        """
        # Create service mappings dynamically to access current values
        service_mappings: dict[type[Any], Any] = {}

        # Common concrete types
        if self.settings is not None:
            service_mappings[type(self.settings)] = self.settings
        if self.http_client is not None:
            service_mappings[httpx.AsyncClient] = self.http_client
        if self.logger is not None:
            service_mappings[structlog.BoundLogger] = self.logger
        if self.config is not None:
            service_mappings[type(self.config)] = self.config
            service_mappings[BaseModel] = self.config

        # Check if service type directly matches a known service
        if service_type in service_mappings:
            return service_mappings[service_type]  # type: ignore[no-any-return]

        # Check all attributes for an instance of the requested type
        for attr_name in dir(self):
            if not attr_name.startswith("_"):  # Skip private attributes
                attr_value = getattr(self, attr_name)
                if attr_value is not None and isinstance(attr_value, service_type):
                    return attr_value  # type: ignore[no-any-return]

        # Service not found
        type_name = getattr(service_type, "__name__", str(service_type))
        raise ValueError(f"Service {type_name} not available in plugin context")

    def get(self, key_or_type: type[T] | str, default: Any = None) -> T | Any:
        """Get service by type (new) or by string key (backward compatibility).

        Args:
            key_or_type: Service type for type-safe access or string key for compatibility
            default: Default value for string-based access (ignored for type-safe access)

        Returns:
            Service instance for type-safe access, or attribute value for string access
        """
        if isinstance(key_or_type, str):
            # Backward compatibility: string-based access
            return getattr(self, key_or_type, default)
        else:
            # Type-safe access
            return self.get_service(key_or_type)

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Get attribute by string name - for backward compatibility.

        Args:
            key: String attribute name
            default: Default value if attribute not found

        Returns:
            Attribute value or default
        """
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Backward compatibility: Allow dictionary-style access."""
        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Backward compatibility: Allow dictionary-style assignment."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Backward compatibility: Support 'key in context' checks."""
        return hasattr(self, key) and getattr(self, key) is not None

    def keys(self) -> list[str]:
        """Backward compatibility: Return list of available service keys."""
        return [
            attr
            for attr in dir(self)
            if not attr.startswith("_")
            and not callable(getattr(self, attr))
            and getattr(self, attr) is not None
        ]


class PluginRuntimeProtocol(Protocol):
    """Protocol for plugin runtime instances."""

    async def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with runtime context."""
        ...

    async def shutdown(self) -> None:
        """Cleanup on shutdown."""
        ...

    async def validate(self) -> bool:
        """Validate plugin is ready."""
        ...

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        ...
