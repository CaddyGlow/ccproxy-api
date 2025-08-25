"""Plugin discovery system for finding and loading plugins.

This module provides mechanisms to discover plugins from the filesystem
and dynamically load their factories.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import structlog

from .factory import PluginFactory


logger = structlog.get_logger(__name__)


class PluginDiscovery:
    """Discovers and loads plugins from the filesystem."""

    def __init__(self, plugins_dir: Path):
        """Initialize plugin discovery.

        Args:
            plugins_dir: Directory containing plugin packages
        """
        self.plugins_dir = plugins_dir
        self.discovered_plugins: dict[str, Path] = {}

    def discover_plugins(self) -> dict[str, Path]:
        """Discover all plugins in the plugins directory.

        Returns:
            Dictionary mapping plugin names to their paths
        """
        self.discovered_plugins.clear()

        if not self.plugins_dir.exists():
            logger.warning(
                "plugins_directory_not_found",
                path=str(self.plugins_dir),
                category="plugin",
            )
            return {}

        # Collect all plugin discoveries first
        discovered = []
        for item in self.plugins_dir.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                # Check for plugin.py file
                plugin_file = item / "plugin.py"
                if plugin_file.exists():
                    self.discovered_plugins[item.name] = plugin_file
                    discovered.append(item.name)
                    # Log individual discoveries at TRACE level
                    if hasattr(logger, "trace"):
                        logger.trace(
                            "plugin_found",
                            name=item.name,
                            path=str(plugin_file),
                            category="plugin",
                        )

        # Single consolidated log for all discoveries
        logger.info(
            "plugins_discovered",
            count=len(discovered),
            names=discovered if discovered else [],
            category="plugin",
        )
        return self.discovered_plugins

    def load_plugin_factory(self, name: str) -> PluginFactory | None:
        """Load a plugin factory by name.

        Args:
            name: Plugin name

        Returns:
            Plugin factory or None if not found or failed to load
        """
        if name not in self.discovered_plugins:
            logger.warning("plugin_not_discovered", name=name, category="plugin")
            return None

        plugin_path = self.discovered_plugins[name]

        try:
            # Create module spec and load the module
            spec = importlib.util.spec_from_file_location(
                f"plugins.{name}.plugin", plugin_path
            )

            if not spec or not spec.loader:
                logger.error(
                    "plugin_spec_creation_failed", name=name, category="plugin"
                )
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the factory from the module
            if not hasattr(module, "factory"):
                logger.error(
                    "plugin_factory_not_found",
                    name=name,
                    msg="Module must export 'factory' variable",
                    category="plugin",
                )
                return None

            factory = module.factory

            if not isinstance(factory, PluginFactory):
                logger.error(
                    "plugin_factory_invalid_type",
                    name=name,
                    type=type(factory).__name__,
                    category="plugin",
                )
                return None

            logger.debug(
                "plugin_factory_loaded",
                name=name,
                version=factory.get_manifest().version,
                category="plugin",
            )

            return factory

        except Exception as e:
            logger.error(
                "plugin_load_failed",
                name=name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )
            return None

    def load_all_factories(self) -> dict[str, PluginFactory]:
        """Load all discovered plugin factories.

        Returns:
            Dictionary mapping plugin names to their factories
        """
        factories = {}

        for name in self.discovered_plugins:
            factory = self.load_plugin_factory(name)
            if factory:
                factories[name] = factory

        logger.info(
            "plugin_factories_loaded",
            count=len(factories),
            names=list(factories.keys()),
            category="plugin",
        )

        return factories


class PluginFilter:
    """Filter plugins based on configuration."""

    def __init__(
        self,
        enabled_plugins: list[str] | None = None,
        disabled_plugins: list[str] | None = None,
    ):
        """Initialize plugin filter.

        Args:
            enabled_plugins: List of explicitly enabled plugins (None = all)
            disabled_plugins: List of explicitly disabled plugins
        """
        self.enabled_plugins = set(enabled_plugins) if enabled_plugins else None
        self.disabled_plugins = set(disabled_plugins) if disabled_plugins else set()

    def is_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled.

        Args:
            plugin_name: Plugin name

        Returns:
            True if plugin is enabled
        """
        # First check if explicitly disabled
        if plugin_name in self.disabled_plugins:
            return False

        # If we have an explicit enabled list, check if in it
        if self.enabled_plugins is not None:
            return plugin_name in self.enabled_plugins

        # Otherwise, enabled by default
        return True

    def filter_factories(
        self, factories: dict[str, PluginFactory]
    ) -> dict[str, PluginFactory]:
        """Filter plugin factories based on configuration.

        Args:
            factories: All discovered factories

        Returns:
            Filtered factories
        """
        filtered = {}

        for name, factory in factories.items():
            if self.is_enabled(name):
                filtered[name] = factory
            else:
                logger.info("plugin_disabled", name=name, category="plugin")

        return filtered


def discover_and_load_plugins(settings: Any) -> dict[str, PluginFactory]:
    """Discover and load all configured plugins.

    Args:
        settings: Application settings

    Returns:
        Dictionary of loaded plugin factories
    """
    # Get plugins directory - go up to project root then to plugins/
    plugins_dir = Path(__file__).parent.parent.parent / "plugins"

    # Discover plugins
    discovery = PluginDiscovery(plugins_dir)
    discovery.discover_plugins()

    # Load all factories
    all_factories = discovery.load_all_factories()

    # Filter based on settings
    filter_config = PluginFilter(
        enabled_plugins=getattr(settings, "enabled_plugins", None),
        disabled_plugins=getattr(settings, "disabled_plugins", None),
    )

    filtered_factories = filter_config.filter_factories(all_factories)

    logger.info(
        "plugins_ready",
        discovered=len(all_factories),
        enabled=len(filtered_factories),
        category="plugin",
    )

    return filtered_factories
