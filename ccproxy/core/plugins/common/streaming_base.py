"""Compatibility layer for streaming metrics interfaces.

This module preserves the historical import path for streaming metrics
interfaces while delegating to the new core streaming interfaces.
"""

from ccproxy.streaming.interfaces import (
    IStreamingMetricsCollector as StreamingMetricsCollector,
    StreamingMetrics,
)

__all__ = ["StreamingMetricsCollector", "StreamingMetrics"]

