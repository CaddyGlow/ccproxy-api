"""LLM format adapters with typed interfaces."""

from .base import APIAdapter, BaseAPIAdapter
from .registry import (
    AdapterRegistration,
    get_registered_adapter_map,
    iter_adapter_registrations,
)
from .shim import AdapterShim

__all__ = [
    "APIAdapter",
    "AdapterRegistration",
    "AdapterShim",
    "BaseAPIAdapter",
    "get_registered_adapter_map",
    "iter_adapter_registrations",
]
