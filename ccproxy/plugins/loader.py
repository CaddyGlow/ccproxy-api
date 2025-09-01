"""Centralized plugin loader.

Provides a single entry to discover factories, build a `PluginRegistry`, and
prepare `MiddlewareManager` based on settings. This isolates loader usage to
one place and reinforces import boundaries (core should not import concrete
plugin modules directly).
"""

from __future__ import annotations

from typing import Any

import structlog

from ccproxy.plugins.discovery import discover_and_load_plugins
from ccproxy.plugins.factory import PluginFactory, PluginRegistry
from ccproxy.plugins.middleware import MiddlewareManager


logger = structlog.get_logger(__name__)


def load_plugin_system(settings: Any) -> tuple[PluginRegistry, MiddlewareManager]:
    """Discover plugins and build a registry + middleware manager.

    This function is the single entry point to set up the plugin layer for
    the application factory. It avoids scattering discovery/registry logic.

    Args:
        settings: Application settings (with plugin config)

    Returns:
        Tuple of (PluginRegistry, MiddlewareManager)
    """
    # Discover factories (filesystem + entry points) with existing helper
    factories: dict[str, PluginFactory] = discover_and_load_plugins(settings)

    # Create registry and register all factories
    registry = PluginRegistry()
    for _name, factory in factories.items():
        registry.register_factory(factory)

    # Prepare middleware manager; plugins will populate via manifests during
    # app creation (manifest population stage) and at runtime as needed
    middleware_manager = MiddlewareManager()

    logger.debug(
        "plugin_system_loaded",
        factory_count=len(factories),
        plugins=list(factories.keys()),
        category="plugin",
    )

    return registry, middleware_manager


__all__ = ["load_plugin_system"]
