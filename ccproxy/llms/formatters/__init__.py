"""LLM format adapters with typed interfaces."""

from .base import APIAdapter, BaseAPIAdapter

# Legacy FormatterRegistry APIs are deprecated/removed from internal usage.
from .formatter_registry import (
    FormatterRegistration,
    formatter,
    register_formatter,
)
from .shim import AdapterShim


__all__ = [
    "APIAdapter",
    "FormatterRegistration",
    "AdapterShim",
    "BaseAPIAdapter",
    "formatter",
    # FormatterRegistry and module auto-loading not exposed for internal use
    "register_formatter",
]
