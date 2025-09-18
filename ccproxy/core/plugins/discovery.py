"""Plugin discovery system for finding and loading plugins.

This module provides mechanisms to discover plugins from the filesystem
and dynamically load their factories.
"""

import importlib
import importlib.util
from pathlib import Path
from typing import Any, cast

import structlog


try:
    # Python 3.10+
    from importlib.metadata import EntryPoint, entry_points
except Exception:  # pragma: no cover
    # Fallback for very old environments
    entry_points = None  # type: ignore
    EntryPoint = Any  # type: ignore

from .interfaces import PluginFactory


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
                f"ccproxy.plugins.{name}.plugin", plugin_path
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

            # logger.debug(
            #     "plugin_factory_loaded",
            #     name=name,
            #     version=factory.get_manifest().version,
            #     category="plugin",
            # )

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
        factories: dict[str, PluginFactory] = {}

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

    def load_entry_point_factories(
        self, skip_names: set[str] | None = None
    ) -> dict[str, PluginFactory]:
        """Load plugin factories from installed entry points.

        Returns:
            Dictionary mapping plugin names to their factories
        """
        factories: dict[str, PluginFactory] = {}
        if entry_points is None:
            logger.debug("entry_points_not_available", category="plugin")
            return factories

        try:
            groups = entry_points()
            eps = []
            # importlib.metadata API differences across Python versions
            if hasattr(groups, "select"):
                eps = list(groups.select(group="ccproxy.plugins"))
            else:  # pragma: no cover
                eps = list(groups.get("ccproxy.plugins", []))

            skip_logged: set[str] = set()
            for ep in eps:
                name = ep.name
                # Skip entry points that collide with existing filesystem plugins
                if skip_names and name in skip_names:
                    if name not in skip_logged:
                        logger.debug(
                            "entry_point_skipped_preexisting_filesystem",
                            name=name,
                            category="plugin",
                        )
                        skip_logged.add(name)
                    continue
                # Skip duplicates within entry points themselves
                if name in factories:
                    if name not in skip_logged:
                        logger.debug(
                            "entry_point_duplicate_ignored",
                            name=name,
                            category="plugin",
                        )
                        skip_logged.add(name)
                    continue
                try:
                    # Primary load
                    obj = ep.load()
                except Exception as e:
                    # Fallback: import module and get 'factory'
                    try:
                        import importlib

                        module_name = getattr(ep, "module", None)
                        if not module_name:
                            value = getattr(ep, "value", "")
                            module_name = value.split(":")[0] if ":" in value else None
                        if not module_name:
                            raise e
                        mod = importlib.import_module(module_name)
                        if hasattr(mod, "factory"):
                            obj = mod.factory
                        else:
                            raise e
                    except Exception as e2:
                        logger.error(
                            "entry_point_load_failed",
                            name=name,
                            error=str(e2),
                            exc_info=e2,
                            category="plugin",
                        )
                        continue

                factory: PluginFactory | None = None

                # If the object already looks like a factory (duck typing)
                if hasattr(obj, "get_manifest") and hasattr(obj, "create_runtime"):
                    factory = cast(PluginFactory, obj)
                # If it's callable, try to call to get a factory
                elif callable(obj):
                    try:
                        maybe = obj()
                        if hasattr(maybe, "get_manifest") and hasattr(
                            maybe, "create_runtime"
                        ):
                            factory = cast(PluginFactory, maybe)
                    except Exception:
                        factory = None

                if not factory:
                    logger.warning(
                        "entry_point_not_factory",
                        name=name,
                        obj_type=type(obj).__name__,
                        category="plugin",
                    )
                    continue

                factories[name] = factory
                # logger.debug(
                #     "entry_point_factory_loaded",
                #     name=name,
                #     version=factory.get_manifest().version,
                #     category="plugin",
                # )
        except Exception as e:  # pragma: no cover
            logger.error("entry_points_enumeration_failed", error=str(e), exc_info=e)
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
    # Get plugins directory - go up to project root then to ccproxy/plugins/
    plugins_dir = Path(__file__).parent.parent.parent / "plugins"

    # Discover plugins
    discovery = PluginDiscovery(plugins_dir)

    # Determine whether to use local filesystem discovery
    disable_local = bool(getattr(settings, "plugins_disable_local_discovery", False))
    if disable_local:
        logger.info(
            "plugins_local_discovery_disabled",
            category="plugin",
            reason="settings.plugins_disable_local_discovery",
        )

    all_factories: dict[str, PluginFactory] = {}
    if not disable_local:
        discovery.discover_plugins()
        # Load factories from local filesystem
        all_factories = discovery.load_all_factories()

    # Load factories from installed entry points and merge. If local discovery
    # is disabled, do not skip any names.
    ep_factories = discovery.load_entry_point_factories(
        skip_names=set(all_factories.keys()) if not disable_local else None
    )
    for name, factory in ep_factories.items():
        if name in all_factories:
            logger.debug(
                "entry_point_factory_ignored",
                name=name,
                reason="filesystem_plugin_with_same_name",
                category="plugin",
            )
            continue
        all_factories[name] = factory

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
        names=list(filtered_factories.keys()),
        category="plugin",
    )

    return filtered_factories
