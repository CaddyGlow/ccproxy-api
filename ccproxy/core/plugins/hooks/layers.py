"""Standard hook execution layers for priority ordering."""

from enum import IntEnum


class HookLayer(IntEnum):
    """Standard hook execution priority layers.

    Hooks execute in priority order from lowest to highest value.
    Within the same priority, hooks execute in registration order.
    """

    # Pre-processing: Core system setup
    CRITICAL = 0  # System-critical hooks (request ID generation, core context)
    VALIDATION = 100  # Input validation and sanitization

    # Context building: Authentication and enrichment
    AUTH = 200  # Authentication and authorization
    ENRICHMENT = 300  # Context enrichment (session data, user info, metadata)

    # Core processing: Business logic
    PROCESSING = 500  # Main request/response processing

    # Observation: Metrics and logging
    OBSERVATION = 700  # Metrics collection, access logging, tracing

    # Post-processing: Cleanup and finalization
    CLEANUP = 900  # Resource cleanup, connection management
    FINALIZATION = 1000  # Final operations before response


# Convenience aliases for common use cases
BEFORE_AUTH = HookLayer.AUTH - 10
AFTER_AUTH = HookLayer.AUTH + 10

BEFORE_PROCESSING = HookLayer.PROCESSING - 10
AFTER_PROCESSING = HookLayer.PROCESSING + 10

# Observation layer ordering (metrics first, logging last)
METRICS = HookLayer.OBSERVATION  # 700: Collect metrics
TRACING = HookLayer.OBSERVATION + 20  # 720: Request tracing
ACCESS_LOGGING = (
    HookLayer.OBSERVATION + 50
)  # 750: Access logs (last to capture all data)
