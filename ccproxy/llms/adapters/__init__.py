"""LLM format adapters with typed interfaces."""

from .base import APIAdapter, BaseAPIAdapter
from .formatter_adapter import FormatterRegistryAdapter, create_formatter_adapter_factory
from .formatter_registry import (
    FormatterRegistration,
    FormatterRegistry,
    formatter,
    iter_registered_formatters,
    load_builtin_formatter_modules,
    register_formatter,
)
from .registry import (
    AdapterRegistration,
    get_registered_adapter_map,
    iter_adapter_registrations,
)
from .shim import AdapterShim


__all__ = [
    "APIAdapter",
    "AdapterRegistration",
    "FormatterRegistration",
    "FormatterRegistry",
    "FormatterRegistryAdapter",
    "AdapterShim",
    "BaseAPIAdapter",
    "create_formatter_adapter_factory",
    "formatter",
    "iter_registered_formatters",
    "get_registered_adapter_map",
    "iter_adapter_registrations",
    "load_builtin_formatter_modules",
    "register_formatter",
]
