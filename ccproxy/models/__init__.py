"""Pydantic models for Claude Proxy API Server.

This package now re-exports Anthropic models from ccproxy.llms.models.anthropic
for backward compatibility, while keeping provider-agnostic models here.
"""

from .provider import ProviderConfig


__all__ = [
    # Provider models
    "ProviderConfig",
]
