"""Core services container for shared services passed to plugins."""

from typing import TYPE_CHECKING, Any, cast

import structlog

from ccproxy.config.settings import Settings


if TYPE_CHECKING:
    from ccproxy.core.plugins import PluginRegistry
    from ccproxy.scheduler.core import Scheduler
    from ccproxy.services.adapters.format_registry import FormatAdapterRegistry
    from ccproxy.http.pool import HTTPPoolManager


class CoreServices:
    """Container for shared services passed to plugins."""

    def __init__(
        self,
        http_pool_manager: "HTTPPoolManager",
        logger: structlog.BoundLogger,
        settings: Settings,
        scheduler: "Scheduler | None" = None,
        plugin_registry: "PluginRegistry | None" = None,
        format_registry: "FormatAdapterRegistry | None" = None,
    ):
        """Initialize core services.

        Args:
            http_pool_manager: HTTP pool manager for plugins to get clients
            logger: Shared logger instance
            settings: Application settings
            scheduler: Optional scheduler for plugin tasks
            plugin_registry: Optional plugin registry for config introspection
            format_registry: Optional format adapter registry for declarative adapters
        """
        self.http_pool_manager = http_pool_manager
        self.logger = logger
        self.settings = settings
        self.scheduler = scheduler
        self.plugin_registry = plugin_registry
        self.format_registry = format_registry

    def is_plugin_logging_enabled(self, plugin_name: str) -> bool:
        """Check if logging is enabled for a specific plugin.

        Args:
            plugin_name: Name of the plugin to check

        Returns:
            bool: True if plugin logging is enabled
        """
        # Check global kill switch first
        if not self.settings.logging.enable_plugin_logging:
            return False

        # Check per-plugin override (defaults to True if not specified)
        return self.settings.logging.plugin_overrides.get(plugin_name, True)

    def get_plugin_config(self, plugin_name: str) -> dict[str, Any]:
        """Get configuration for a specific plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            dict: Plugin-specific configuration or empty dict
        """
        # Check if this is a logging plugin and if logging is disabled for it
        if plugin_name.endswith("_logger") and not self.is_plugin_logging_enabled(
            plugin_name
        ):
            return {"enabled": False}

        # Try to get config from plugin's config class if registry is available
        if self.plugin_registry:
            runtime = self.plugin_registry.get_runtime(plugin_name)
            if runtime and hasattr(runtime, "get_config_class"):
                config_class = runtime.get_config_class()
                if config_class:
                    # Get raw config from settings.plugins dictionary
                    raw_config = self.settings.plugins.get(plugin_name, {})

                    # Apply shared base directory for logging plugins if not set
                    if plugin_name == "raw_http_logger" and "log_dir" not in raw_config:
                        raw_config["log_dir"] = (
                            f"{self.settings.logging.plugin_log_base_dir}/raw"
                        )

                    # Validate and return config using plugin's schema
                    try:
                        validated_config = config_class(**raw_config)
                        return cast(dict[str, Any], validated_config.model_dump())
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
        config = self.settings.plugins.get(plugin_name, {})

        # Apply shared base directory for logging plugins if not set
        if plugin_name == "raw_http_logger" and "log_dir" not in config:
            config["log_dir"] = f"{self.settings.logging.plugin_log_base_dir}/raw"

        return config

    def get_format_registry(self) -> "FormatAdapterRegistry":
        """Get format adapter registry service instance.

        Returns:
            FormatAdapterRegistry: The format adapter registry service

        Raises:
            RuntimeError: If format registry is not available
        """
        if self.format_registry is None:
            raise RuntimeError("Format adapter registry is not available")
        return self.format_registry
