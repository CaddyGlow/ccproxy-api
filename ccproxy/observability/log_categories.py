"""Log categories for structured logging and filtering."""

from enum import Enum


class LogCategory(str, Enum):
    """Log categories for filtering and organization.

    Used to categorize logs for easier filtering and analysis.
    Can be filtered via CCPROXY_LOG_CHANNELS and CCPROXY_LOG_EXCLUDE_CHANNELS.
    """

    LIFECYCLE = "lifecycle"  # Request start/end, major milestones
    PLUGIN = "plugin"  # Plugin discovery, initialization, registration
    HTTP = "http"  # External HTTP calls to provider APIs
    STREAMING = "streaming"  # SSE/streaming events and chunks
    AUTH = "auth"  # Authentication, token refresh, OAuth flows
    TRANSFORM = "transform"  # Request/response transformations
    CACHE = "cache"  # Cache hits, misses, invalidations
    MIDDLEWARE = "middleware"  # Middleware execution and processing
    CONFIG = "config"  # Configuration loading and validation
    METRICS = "metrics"  # Performance metrics and measurements
    ACCESS = "access"  # Access logging and request tracking
    REQUEST = "request"  # Request processing and handling
    DEFAULT = "general"  # Uncategorized logs


# Export category constants to avoid circular imports
LIFECYCLE = LogCategory.LIFECYCLE.value
PLUGIN = LogCategory.PLUGIN.value
HTTP = LogCategory.HTTP.value
STREAMING = LogCategory.STREAMING.value
AUTH = LogCategory.AUTH.value
TRANSFORM = LogCategory.TRANSFORM.value
CACHE = LogCategory.CACHE.value
MIDDLEWARE = LogCategory.MIDDLEWARE.value
CONFIG = LogCategory.CONFIG.value
METRICS = LogCategory.METRICS.value
ACCESS = LogCategory.ACCESS.value
REQUEST = LogCategory.REQUEST.value
DEFAULT = LogCategory.DEFAULT.value
