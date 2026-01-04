"""Lightweight CLI discovery for plugin command registration.

This module provides minimal plugin discovery specifically for CLI command
registration, loading only plugin manifests without full initialization.
"""

import importlib.util
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import structlog

from ccproxy.core.plugins.declaration import PluginManifest
from ccproxy.core.plugins.discovery import (
    PluginFilter,
    build_combined_plugin_denylist,
)
from ccproxy.core.plugins.interfaces import PluginFactory


logger = structlog.get_logger(__name__)


def discover_plugin_cli_extensions(
    settings: Any | None = None,
) -> list[tuple[str, PluginManifest]]:
    """Lightweight discovery of plugin CLI extensions.

    Only loads plugin factories and manifests, no runtime initialization.
    Used during CLI app creation to register plugin commands/arguments.

    Args:
        settings: Optional settings object to filter plugins

    Returns:
        List of (plugin_name, manifest) tuples for plugins with CLI extensions.
    """
    plugin_manifests = []

    # Discover from filesystem (plugins/ directory)
    try:
        filesystem_manifests = _discover_filesystem_cli_extensions()
        plugin_manifests.extend(filesystem_manifests)
    except Exception as e:
        logger.debug("filesystem_cli_discovery_failed", error=str(e))

    # Discover from entry points
    try:
        entry_point_manifests = _discover_entry_point_cli_extensions()
        plugin_manifests.extend(entry_point_manifests)
    except Exception as e:
        logger.debug("entry_point_cli_discovery_failed", error=str(e))

    # Remove duplicates (filesystem takes precedence)
    seen_names = set()
    unique_manifests = []
    for name, manifest in plugin_manifests:
        if name not in seen_names:
            unique_manifests.append((name, manifest))
            seen_names.add(name)

    # Apply plugin filtering if settings provided
    if settings is not None:
        combined_denylist = build_combined_plugin_denylist(
            getattr(settings, "disabled_plugins", None),
            getattr(settings, "plugins", None),
        )

        plugin_filter = PluginFilter(
            enabled_plugins=getattr(settings, "enabled_plugins", None),
            disabled_plugins=combined_denylist,
        )

        filtered_manifests = []
        for name, manifest in unique_manifests:
            if plugin_filter.is_enabled(name):
                filtered_manifests.append((name, manifest))
            else:
                logger.debug(
                    "plugin_cli_extension_disabled", plugin=name, category="plugin"
                )

        return filtered_manifests

    return unique_manifests


def _discover_filesystem_cli_extensions() -> list[tuple[str, PluginManifest]]:
    """Discover CLI extensions from filesystem plugins/ directories."""
    manifests: list[tuple[str, PluginManifest]] = []

    # Check local plugins/
    plugins_dirs = [
        Path("plugins"),
    ]

    for plugins_dir in plugins_dirs:
        if not plugins_dir.exists():
            continue

        manifests.extend(_discover_plugins_in_directory(plugins_dir))

    return manifests


def _discover_plugins_in_directory(
    plugins_dir: Path,
) -> list[tuple[str, PluginManifest]]:
    """Discover CLI extensions from a specific plugins directory."""
    manifests: list[tuple[str, PluginManifest]] = []

    for plugin_path in plugins_dir.iterdir():
        if not plugin_path.is_dir() or plugin_path.name.startswith("_"):
            continue

        plugin_file = plugin_path / "plugin.py"
        if not plugin_file.exists():
            continue

        try:
            factory = _load_plugin_factory_from_file(plugin_file)
            if factory:
                manifest = factory.get_manifest()
                if manifest.cli_commands or manifest.cli_arguments:
                    manifests.append((manifest.name, manifest))
        except Exception as e:
            logger.debug(
                "filesystem_plugin_cli_discovery_failed",
                plugin=plugin_path.name,
                error=str(e),
            )

    return manifests


def _discover_entry_point_cli_extensions() -> list[tuple[str, PluginManifest]]:
    """Discover CLI extensions from installed entry points."""
    manifests: list[tuple[str, PluginManifest]] = []

    try:
        plugin_entries = entry_points(group="ccproxy.plugins")
    except Exception:
        return manifests

    for entry_point in plugin_entries:
        try:
            factory_or_callable = entry_point.load()

            # Handle both factory instances and factory callables
            if callable(factory_or_callable) and not isinstance(
                factory_or_callable, PluginFactory
            ):
                factory = factory_or_callable()
            else:
                factory = factory_or_callable

            if isinstance(factory, PluginFactory):
                manifest = factory.get_manifest()
                if manifest.cli_commands or manifest.cli_arguments:
                    manifests.append((manifest.name, manifest))
        except Exception as e:
            logger.debug(
                "entry_point_plugin_cli_discovery_failed",
                entry_point=entry_point.name,
                error=str(e),
            )

    return manifests


def _load_plugin_factory_from_file(plugin_file: Path) -> PluginFactory | None:
    """Load plugin factory from a plugin.py file."""
    try:
        # Use proper package naming for ccproxy plugins
        plugin_name = plugin_file.parent.name

        # Check if it's in ccproxy/plugins/ structure
        if "ccproxy/plugins" in str(plugin_file):
            module_name = f"ccproxy.plugins.{plugin_name}.plugin"
        else:
            module_name = f"plugin_{plugin_name}"

        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        if not spec or not spec.loader:
            return None

        module = importlib.util.module_from_spec(spec)

        # Temporarily add to sys.modules for relative imports
        old_module = sys.modules.get(spec.name)
        sys.modules[spec.name] = module

        try:
            spec.loader.exec_module(module)
            factory = getattr(module, "factory", None)

            if isinstance(factory, PluginFactory):
                return factory
        finally:
            # Restore original module or remove
            if old_module is not None:
                sys.modules[spec.name] = old_module
            else:
                sys.modules.pop(spec.name, None)

    except Exception:
        pass

    return None
