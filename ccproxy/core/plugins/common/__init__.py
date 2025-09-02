"""Common helpers for plugins (PEP 420 migration).

This package hosts shared utilities that used to live under
`plugins/common`. It provides stable import locations for built-in and
external plugins targeting `ccproxy.core.plugins.common`.
"""

__all__ = [
    # Provided by streaming_base for compatibility
    "StreamingMetrics",
    "StreamingMetricsCollector",
]
