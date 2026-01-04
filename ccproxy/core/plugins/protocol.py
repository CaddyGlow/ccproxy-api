"""Plugin protocol for provider plugins."""

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from fastapi import APIRouter
from pydantic import BaseModel
from typing_extensions import TypedDict

from ccproxy.core.plugins.hooks.base import Hook
from ccproxy.models.provider import ProviderConfig
from ccproxy.services.adapters.base import BaseAdapter


if TYPE_CHECKING:
    from ccproxy.scheduler.tasks import BaseScheduledTask
    from ccproxy.services.container import ServiceContainer


@runtime_checkable
class OAuthClientProtocol(Protocol):
    """Protocol for OAuth client implementations."""

    async def authenticate(self, open_browser: bool = True) -> Any:
        """Perform OAuth authentication flow.

        Args:
            open_browser: Whether to automatically open browser

        Returns:
            Provider-specific credentials object
        """
        ...

    async def refresh_access_token(self, refresh_token: str) -> Any:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New token response
        """
        ...


class AuthCommandDefinition(TypedDict, total=False):
    """Definition for provider-specific auth command extensions."""

    command_name: str  # Required: Command name (e.g., 'validate', 'profile')
    description: str  # Required: Command description
    handler: Any  # Required: Async command handler function
    options: dict[str, Any]  # Optional: Additional command options


class HealthCheckResult(BaseModel):
    """Standardized health check result following IETF format."""

    status: Literal["pass", "warn", "fail"]
    componentId: str  # noqa: N815
    componentType: str = "provider_plugin"  # noqa: N815
    output: str | None = None
    version: str | None = None
    details: dict[str, Any] | None = None


class ScheduledTaskDefinition(TypedDict, total=False):
    """Definition for a scheduled task from a plugin."""

    task_name: str  # Required: Unique name for the task instance
    task_type: str  # Required: Type identifier for task registry
    task_class: type["BaseScheduledTask"]  # Required: Task class
    interval_seconds: float  # Required: Interval between executions
    enabled: bool  # Optional: Whether task is enabled (default: True)
    # Additional kwargs can be passed for task initialization


@runtime_checkable
class BasePlugin(Protocol):
    """Base protocol for all plugins."""

    @property
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    def version(self) -> str:
        """Plugin version."""
        ...

    @property
    def dependencies(self) -> list[str]:
        """List of plugin names this plugin depends on."""
        ...

    @property
    def router_prefix(self) -> str:
        """Unique route prefix for this plugin."""
        ...

    async def initialize(self, services: "ServiceContainer") -> None:
        """Initialize plugin with shared services. Called once on startup."""
        ...

    async def shutdown(self) -> None:
        """Perform graceful shutdown. Called once on app shutdown."""
        ...

    async def validate(self) -> bool:
        """Validate plugin is ready."""
        ...

    def get_routes(self) -> APIRouter | dict[str, APIRouter] | None:
        """Get plugin-specific routes (optional)."""
        ...

    async def health_check(self) -> HealthCheckResult:
        """Perform health check following IETF format."""
        ...

    def get_scheduled_tasks(self) -> list[ScheduledTaskDefinition] | None:
        """Get scheduled task definitions for this plugin (optional).

        Returns:
            List of task definitions or None if no scheduled tasks needed
        """
        ...

    def get_config_class(self) -> type[BaseModel] | None:
        """Get the Pydantic configuration model for this plugin.

        Returns:
            Pydantic BaseModel class for plugin configuration or None if no configuration needed
        """
        ...

    def get_hooks(self) -> list[Hook] | None:
        """Get hooks provided by this plugin (optional).

        Returns:
            List of hook instances or None if no hooks
        """
        ...


@runtime_checkable
class SystemPlugin(BasePlugin, Protocol):
    """Protocol for system plugins (non-provider plugins).

    System plugins inherit all methods from BasePlugin and don't add
    any additional requirements. They don't proxy to external providers
    and therefore don't need adapters or provider configurations.
    """

    # SystemPlugin has no additional methods beyond BasePlugin
    pass


@runtime_checkable
class ProviderPlugin(BasePlugin, Protocol):
    """Enhanced protocol for provider plugins.

    Provider plugins proxy requests to external API providers and therefore
    need additional methods for creating adapters and configurations.
    """

    def create_adapter(self) -> BaseAdapter:
        """Create adapter instance for handling provider requests."""
        ...

    def create_config(self) -> ProviderConfig:
        """Create provider configuration from settings."""
        ...

    async def get_oauth_client(self) -> OAuthClientProtocol | None:
        """Get OAuth client for this plugin if it supports OAuth authentication.

        Returns:
            OAuth client instance or None if plugin doesn't support OAuth
        """
