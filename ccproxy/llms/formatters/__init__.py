"""LLM format adapters with typed interfaces."""

from .base import APIAdapter, BaseAPIAdapter
from .shim import AdapterShim


__all__ = [
    "APIAdapter",
    "AdapterShim",
    "BaseAPIAdapter",
]
