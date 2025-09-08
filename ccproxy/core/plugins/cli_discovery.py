"""Lightweight CLI discovery for plugin command registration.

This module provides minimal plugin discovery specifically for CLI command
registration, loading only plugin manifests without full initialization.
"""

import importlib.util
import sys
from importlib.metadata import entry_points
from pathlib import Path

import structlog

from ccproxy.core.plugins.declaration import PluginManifest
from ccproxy.core.plugins.factory import PluginFactory


logger = structlog.get_logger(__name__)


def discover_plugin_cli_extensions() -> list[tuple[str, PluginManifest]]:
    """Lightweight discovery of plugin CLI extensions.

    Only loads plugin factories and manifests, no runtime initialization.
    Used during CLI app creation to register plugin commands/arguments.

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

    return unique_manifests


def _discover_filesystem_cli_extensions() -> list[tuple[str, PluginManifest]]:
    """Discover CLI extensions from filesystem plugins/ directory."""
    manifests: list[tuple[str, PluginManifest]] = []
    plugins_dir = Path("plugins")

    if not plugins_dir.exists():
        return manifests

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
        spec = importlib.util.spec_from_file_location(
            f"plugin_{plugin_file.parent.name}", plugin_file
        )
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
