"""CCProxy Plugin System v2.

This module provides a modern plugin system with proper lifecycle management,
supporting middleware, routes, scheduled tasks, hooks, and more.
"""

from .declaration import (
    AuthCommandSpec,
    HookSpec,
    MiddlewareLayer,
    MiddlewareSpec,
    PluginContext,
    PluginManifest,
    PluginRuntimeProtocol,
    RouteSpec,
    TaskSpec,
)
from .discovery import PluginDiscovery, PluginFilter, discover_and_load_plugins
from .factory import (
    BasePluginFactory,
    PluginFactory,
    PluginRegistry,
    ProviderPluginFactory,
    SystemPluginFactory,
)
from .middleware import CoreMiddlewareSpec, MiddlewareManager, setup_default_middleware
from .runtime import BasePluginRuntime, ProviderPluginRuntime, SystemPluginRuntime


__all__ = [
    # Declaration
    "PluginManifest",
    "PluginContext",
    "PluginRuntimeProtocol",
    "MiddlewareSpec",
    "MiddlewareLayer",
    "RouteSpec",
    "TaskSpec",
    "HookSpec",
    "AuthCommandSpec",
    # Runtime
    "BasePluginRuntime",
    "SystemPluginRuntime",
    "ProviderPluginRuntime",
    # Factory
    "PluginFactory",
    "BasePluginFactory",
    "SystemPluginFactory",
    "ProviderPluginFactory",
    "PluginRegistry",
    # Discovery
    "PluginDiscovery",
    "PluginFilter",
    "discover_and_load_plugins",
    # Middleware
    "MiddlewareManager",
    "CoreMiddlewareSpec",
    "setup_default_middleware",
]
