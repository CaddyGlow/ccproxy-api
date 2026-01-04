"""Generic streaming utilities for CCProxy.

This package provides transport-agnostic streaming functionality:
- Stream interfaces and handlers
- Buffer management
- Deferred streaming for header preservation
"""

from .buffer import StreamingBufferService
from .buffer import StreamingBufferService as BufferService
from .deferred import DeferredStreaming
from .handler import StreamingHandler
from .interfaces import IStreamingMetricsCollector, StreamingMetrics


__all__ = [
    "BufferService",
    "StreamingBufferService",
    "StreamingMetrics",
    "IStreamingMetricsCollector",
    "StreamingHandler",
    "DeferredStreaming",
]
