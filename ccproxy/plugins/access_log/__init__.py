"""Access log plugin for CCProxy.

Provides structured access logging for both client and provider requests
using the hook system.
"""

from .config import AccessLogConfig
from .hook import AccessLogHook
from .plugin import AccessLogFactory, AccessLogRuntime


__all__ = [
    "AccessLogConfig",
    "AccessLogFactory",
    "AccessLogHook",
    "AccessLogRuntime",
]

# Export the factory instance for plugin loading
factory = AccessLogFactory()
