"""Core services container for shared services passed to plugins."""

from typing import TYPE_CHECKING, Any

import structlog
from httpx import AsyncClient

from ccproxy.config.settings import Settings


if TYPE_CHECKING:
    from ccproxy.plugins.factory import PluginRegistry
    from ccproxy.scheduler.core import Scheduler
    from ccproxy.services.interfaces import IRequestHandler


class CoreServices:
    """Container for shared services passed to plugins."""

    def __init__(
        self,
        http_client: AsyncClient,
        logger: structlog.BoundLogger,
        settings: Settings,
        scheduler: "Scheduler | None" = None,
        plugin_registry: "PluginRegistry | None" = None,
        proxy_service: "IRequestHandler | None" = None,
    ):
        """Initialize core services.

        Args:
            http_client: Shared HTTP client for plugins
            logger: Shared logger instance
            settings: Application settings
            scheduler: Optional scheduler for plugin tasks
            plugin_registry: Optional plugin registry for config introspection
            proxy_service: Optional request handler reference for adapter initialization
        """
        self.http_client = http_client
        self.logger = logger
        self.settings = settings
        self.scheduler = scheduler
        self.plugin_registry = plugin_registry
        self.proxy_service = proxy_service

    def get_plugin_config(self, plugin_name: str) -> dict[str, Any]:
        """Get configuration for a specific plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            dict: Plugin-specific configuration or empty dict
        """
        # Try to get config from plugin's config class if registry is available
        if self.plugin_registry:
            runtime = self.plugin_registry.get_runtime(plugin_name)
            if runtime and hasattr(runtime, "get_config_class"):
                config_class = runtime.get_config_class()
                if config_class:
                    # Get raw config from settings.plugins dictionary
                    raw_config = self.settings.plugins.get(plugin_name, {})

                    # Validate and return config using plugin's schema
                    try:
                        validated_config = config_class(**raw_config)
                        return validated_config.model_dump()  # type: ignore[no-any-return]
                    except (ValueError, TypeError) as e:
                        self.logger.error(
                            "config_validation_error",
                            plugin_name=plugin_name,
                            error=str(e),
                            exc_info=e,
                        )
                        return {}
                    except Exception as e:
                        self.logger.error(
                            "config_unexpected_error",
                            plugin_name=plugin_name,
                            error=str(e),
                            exc_info=e,
                        )
                        return {}

        # Default: look in plugins dictionary
        return self.settings.plugins.get(plugin_name, {})
