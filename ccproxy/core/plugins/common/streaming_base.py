"""Compatibility layer for streaming metrics interfaces.

This module preserves the historical import path for streaming metrics
interfaces while delegating to the new core streaming interfaces.
"""

from ccproxy.streaming import (
    IStreamingMetricsCollector as StreamingMetricsCollector,
)
from ccproxy.streaming import (
    StreamingMetrics,
)


__all__ = ["StreamingMetricsCollector", "StreamingMetrics"]
