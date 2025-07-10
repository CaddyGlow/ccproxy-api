"""Adapter modules for API format conversion."""

from ccproxy.core.interfaces import APIAdapter

from .openai import OpenAIAdapter


__all__ = [
    "APIAdapter",
    "OpenAIAdapter",
]
