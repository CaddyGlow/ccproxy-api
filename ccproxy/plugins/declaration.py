"""Plugin declaration system for static plugin specification.

This module provides the declaration layer of the plugin system, allowing plugins
to specify their requirements and capabilities at declaration time (app creation)
rather than runtime (lifespan).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

import httpx
import structlog
from fastapi import APIRouter
from pydantic import BaseModel


if TYPE_CHECKING:
    from starlette.middleware.base import BaseHTTPMiddleware

    from ccproxy.config.settings import Settings
    from ccproxy.hooks.base import Hook
    from ccproxy.plugins.protocol import OAuthClientProtocol
    from ccproxy.scheduler.core import Scheduler
    from ccproxy.scheduler.tasks import BaseScheduledTask
    from ccproxy.services.adapters.base import BaseAdapter
    from ccproxy.services.cli_detection import CLIDetectionService
    from ccproxy.services.proxy_service import ProxyService
else:
    # Runtime import - mypy doesn't have stubs for fastapi.middleware
    from starlette.middleware.base import BaseHTTPMiddleware


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

    # Static specifications
    middleware: list[MiddlewareSpec] = field(default_factory=list)
    routes: list[RouteSpec] = field(default_factory=list)
    tasks: list[TaskSpec] = field(default_factory=list)
    hooks: list[HookSpec] = field(default_factory=list)
    auth_commands: list[AuthCommandSpec] = field(default_factory=list)

    # Configuration
    config_class: type[BaseModel] | None = None

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

    def validate_dependencies(self, available_plugins: set[str]) -> list[str]:
        """Validate that all dependencies are available.

        Args:
            available_plugins: Set of available plugin names

        Returns:
            List of missing dependencies
        """
        return [dep for dep in self.dependencies if dep not in available_plugins]

    def get_sorted_middleware(self) -> list[MiddlewareSpec]:
        """Get middleware sorted by priority."""
        return sorted(self.middleware)


class PluginContext(TypedDict, total=False):
    """Context provided to plugin runtime during initialization."""

    settings: "Settings"  # Application settings
    http_client: httpx.AsyncClient  # Shared HTTP client
    logger: structlog.BoundLogger  # Structured logger
    proxy_service: "ProxyService"  # ProxyService instance
    scheduler: "Scheduler"  # Scheduler instance
    config: BaseModel | None  # Plugin-specific configuration
    cli_detection_service: "CLIDetectionService"  # Shared CLI detection service

    # Provider-specific
    adapter: "BaseAdapter"  # BaseAdapter instance
    detection_service: Any  # Detection service instance (provider-specific)
    credentials_manager: Any  # Credentials manager (plugin-specific)


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

    # Provider plugin methods
    async def get_profile_info(self) -> dict[str, Any] | None:
        """Get provider profile information."""
        ...

    async def get_auth_summary(self) -> dict[str, Any]:
        """Get authentication summary."""
        ...
