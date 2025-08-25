"""Extended request context for proxy service."""

from typing import Any

from ccproxy.observability.context import RequestContext as BaseRequestContext


class ProxyRequestContext:
    """Extended request context with proxy-specific attributes."""

    def __init__(self, base_context: BaseRequestContext):
        """Initialize with base context and proxy-specific attributes."""
        self.base_context = base_context
        self.request_id = base_context.request_id
        self.logger = base_context.logger
        self.metadata = base_context.metadata

        # Proxy-specific attributes
        self.provider: str | None = None
        self.endpoint: str | None = None
        self.method: str | None = None
        self.model: str | None = None
        self.metrics: dict[str, Any] = {}

    @property
    def duration_ms(self) -> float:
        """Get current duration in milliseconds."""
        return self.base_context.duration_ms

    @property
    def duration_seconds(self) -> float:
        """Get current duration in seconds."""
        return self.base_context.duration_seconds

    def add_metadata(self, **kwargs: Any) -> None:
        """Add metadata to the request context."""
        self.base_context.add_metadata(**kwargs)

    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log an event with current context and timing."""
        self.base_context.log_event(event, **kwargs)


def create_proxy_context(
    request_id: str, provider: str, endpoint: str, method: str
) -> ProxyRequestContext:
    """Create a proxy request context."""
    import time

    import structlog

    # Create base context
    base_ctx = BaseRequestContext(
        request_id=request_id,
        start_time=time.perf_counter(),
        logger=structlog.get_logger(),
        metadata={},
    )

    # Create proxy context
    ctx = ProxyRequestContext(base_ctx)
    ctx.provider = provider
    ctx.endpoint = endpoint
    ctx.method = method

    return ctx
