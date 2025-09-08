"""CCProxy Plugin System public API (lazy facade).

This package re-exports the stable plugin API while avoiding heavy imports
and circular dependencies at module import time. Names are loaded lazily
via ``__getattr__`` on first access.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Declarations
    "PluginManifest": ("ccproxy.core.plugins.declaration", "PluginManifest"),
    "PluginContext": ("ccproxy.core.plugins.declaration", "PluginContext"),
    "PluginRuntimeProtocol": (
        "ccproxy.core.plugins.declaration",
        "PluginRuntimeProtocol",
    ),
    "MiddlewareSpec": ("ccproxy.core.plugins.declaration", "MiddlewareSpec"),
    "MiddlewareLayer": ("ccproxy.core.plugins.declaration", "MiddlewareLayer"),
    "RouteSpec": ("ccproxy.core.plugins.declaration", "RouteSpec"),
    "TaskSpec": ("ccproxy.core.plugins.declaration", "TaskSpec"),
    "HookSpec": ("ccproxy.core.plugins.declaration", "HookSpec"),
    "AuthCommandSpec": ("ccproxy.core.plugins.declaration", "AuthCommandSpec"),
    # Runtime classes
    "BasePluginRuntime": ("ccproxy.core.plugins.runtime", "BasePluginRuntime"),
    "SystemPluginRuntime": ("ccproxy.core.plugins.runtime", "SystemPluginRuntime"),
    "ProviderPluginRuntime": (
        "ccproxy.core.plugins.runtime",
        "ProviderPluginRuntime",
    ),
    "AuthProviderPluginRuntime": (
        "ccproxy.core.plugins.runtime",
        "AuthProviderPluginRuntime",
    ),
    # Factories and registry
    "PluginFactory": ("ccproxy.core.plugins.factory", "PluginFactory"),
    "BasePluginFactory": ("ccproxy.core.plugins.factory", "BasePluginFactory"),
    "SystemPluginFactory": ("ccproxy.core.plugins.factory", "SystemPluginFactory"),
    "ProviderPluginFactory": ("ccproxy.core.plugins.factory", "ProviderPluginFactory"),
    "AuthProviderPluginFactory": (
        "ccproxy.core.plugins.factory",
        "AuthProviderPluginFactory",
    ),
    "PluginRegistry": ("ccproxy.core.plugins.factory", "PluginRegistry"),
    "factory_type_name": ("ccproxy.core.plugins.factory", "factory_type_name"),
    # Base provider convenience factory
    "BaseProviderPluginFactory": (
        "ccproxy.core.plugins.base_factory",
        "BaseProviderPluginFactory",
    ),
    # Loader / discovery
    "load_plugin_system": ("ccproxy.core.plugins.loader", "load_plugin_system"),
    "load_cli_plugins": ("ccproxy.core.plugins.loader", "load_cli_plugins"),
    "PluginDiscovery": ("ccproxy.core.plugins.discovery", "PluginDiscovery"),
    "PluginFilter": ("ccproxy.core.plugins.discovery", "PluginFilter"),
    "discover_and_load_plugins": (
        "ccproxy.core.plugins.discovery",
        "discover_and_load_plugins",
    ),
    # Middleware manager
    "MiddlewareManager": ("ccproxy.core.plugins.middleware", "MiddlewareManager"),
    "CoreMiddlewareSpec": ("ccproxy.core.plugins.middleware", "CoreMiddlewareSpec"),
    "setup_default_middleware": (
        "ccproxy.core.plugins.middleware",
        "setup_default_middleware",
    ),
}


__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:  # pragma: no cover - import facade
    if name not in _EXPORTS:
        raise AttributeError(f"module 'ccproxy.core.plugins' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:  # pragma: no cover - tooling aid
    return sorted(list(globals().keys()) + __all__)
