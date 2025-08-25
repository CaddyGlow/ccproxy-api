"""Streaming response utilities for header preservation.

This package provides streaming response handling that preserves
upstream headers through deferred response creation.
"""

from .deferred_streaming import DeferredStreaming


__all__ = [
    "DeferredStreaming",
]
