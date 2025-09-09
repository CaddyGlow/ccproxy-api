"""CCProxy Plugin System public API (minimal re-exports).

This module exposes the common symbols used by plugins and app code while
keeping imports straightforward to avoid circular dependencies.
"""

from .declaration import (
    AuthCommandSpec,
    FormatAdapterSpec,
    FormatPair,
    HookSpec,
    MiddlewareLayer,
    MiddlewareSpec,
    PluginContext,
    PluginManifest,
    PluginRuntimeProtocol,
    RouteSpec,
    TaskSpec,
)
from .factories import (
    BaseProviderPluginFactory,
    PluginRegistry,
)
from .interfaces import (
    AuthProviderPluginFactory,
    BasePluginFactory,
    PluginFactory,
    ProviderPluginFactory,
    SystemPluginFactory,
    factory_type_name,
)
from .loader import load_cli_plugins, load_plugin_system
from .middleware import CoreMiddlewareSpec, MiddlewareManager, setup_default_middleware
from .runtime import (
    AuthProviderPluginRuntime,
    BasePluginRuntime,
    ProviderPluginRuntime,
    SystemPluginRuntime,
)


__all__ = [
    # Declarations
    "PluginManifest",
    "PluginContext",
    "PluginRuntimeProtocol",
    "MiddlewareSpec",
    "MiddlewareLayer",
    "RouteSpec",
    "TaskSpec",
    "HookSpec",
    "AuthCommandSpec",
    "FormatAdapterSpec",
    "FormatPair",
    # Runtime
    "BasePluginRuntime",
    "SystemPluginRuntime",
    "ProviderPluginRuntime",
    "AuthProviderPluginRuntime",
    # Base factory
    "BaseProviderPluginFactory",
    # Factory and registry
    "PluginFactory",
    "BasePluginFactory",
    "SystemPluginFactory",
    "ProviderPluginFactory",
    "AuthProviderPluginFactory",
    "PluginRegistry",
    "factory_type_name",
    # Middleware
    "MiddlewareManager",
    "CoreMiddlewareSpec",
    "setup_default_middleware",
    # Loader functions
    "load_plugin_system",
    "load_cli_plugins",
]
