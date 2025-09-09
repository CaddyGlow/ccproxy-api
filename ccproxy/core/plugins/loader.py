"""Centralized plugin loader.

Provides a single entry to discover factories, build a `PluginRegistry`, and
prepare `MiddlewareManager` based on settings. This isolates loader usage to
one place and reinforces import boundaries (core should not import concrete
plugin modules directly).
"""

from __future__ import annotations

from typing import Any

import structlog

from ccproxy.core.plugins.discovery import discover_and_load_plugins
from ccproxy.core.plugins.factories import PluginRegistry
from ccproxy.core.plugins.interfaces import (
    AuthProviderPluginFactory,
    PluginFactory,
)
from ccproxy.core.plugins.middleware import MiddlewareManager


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


def load_cli_plugins(
    settings: Any,
    auth_provider: str | None = None,
    allow_plugins: list[str] | None = None,
) -> PluginRegistry:
    """Load filtered plugins for CLI operations.

    This function creates a lightweight plugin registry for CLI commands that:
    - Includes only CLI-safe plugins (marked with cli_safe = True)
    - Optionally includes a specific auth provider plugin if requested
    - Excludes heavy provider plugins that cause DuckDB locks, task manager errors, etc.

    Args:
        settings: Application settings
        auth_provider: Name of auth provider to include (e.g., "codex", "claude-api")
        allow_plugins: Additional plugins to explicitly allow (beyond cli_safe ones)

    Returns:
        Filtered PluginRegistry containing only CLI-appropriate plugins
    """
    # Discover all available factories
    all_factories: dict[str, PluginFactory] = discover_and_load_plugins(settings)

    # Start with CLI-safe plugins
    cli_factories: dict[str, PluginFactory] = {}

    for name, factory in all_factories.items():
        # Include plugins explicitly marked as CLI-safe
        if getattr(factory, "cli_safe", False):
            cli_factories[name] = factory

    # Add specific auth provider if requested
    if auth_provider:
        auth_plugin_name = _resolve_auth_provider_plugin_name(auth_provider)
        if auth_plugin_name and auth_plugin_name in all_factories:
            cli_factories[auth_plugin_name] = all_factories[auth_plugin_name]
        else:
            logger.warning(
                "auth_provider_not_found",
                provider=auth_provider,
                resolved_name=auth_plugin_name,
                available_auth_providers=[
                    name
                    for name, factory in all_factories.items()
                    if isinstance(factory, AuthProviderPluginFactory)
                ],
            )

    # Add explicitly allowed plugins
    if allow_plugins:
        for plugin_name in allow_plugins:
            if plugin_name in all_factories and plugin_name not in cli_factories:
                cli_factories[plugin_name] = all_factories[plugin_name]

    # Create filtered registry
    registry = PluginRegistry()
    for _name, factory in cli_factories.items():
        registry.register_factory(factory)

    logger.debug(
        "cli_plugin_system_loaded",
        total_available=len(all_factories),
        cli_safe_count=len(
            [f for f in all_factories.values() if getattr(f, "cli_safe", False)]
        ),
        loaded_count=len(cli_factories),
        loaded_plugins=list(cli_factories.keys()),
        auth_provider=auth_provider,
        allow_plugins=allow_plugins or [],
        category="plugin",
    )

    return registry


def _resolve_auth_provider_plugin_name(provider: str) -> str | None:
    """Map CLI provider name to auth plugin name.

    Args:
        provider: CLI provider name (e.g., "codex", "claude-api")

    Returns:
        Plugin name (e.g., "oauth_codex", "oauth_claude") or None
    """
    provider_key = provider.strip().lower().replace("_", "-")

    mapping: dict[str, str] = {
        "codex": "oauth_codex",
        "openai": "oauth_codex",
        "openai-api": "oauth_codex",
        "claude": "oauth_claude",
        "claude-api": "oauth_claude",
        "claude_api": "oauth_claude",
        "copilot": "copilot",
    }

    resolved = mapping.get(provider_key)
    if resolved:
        return resolved
    # Fallback: build dynamically as oauth_<provider>
    fallback = "oauth_" + provider_key.replace("-", "_")
    return fallback


__all__ = ["load_plugin_system", "load_cli_plugins"]
