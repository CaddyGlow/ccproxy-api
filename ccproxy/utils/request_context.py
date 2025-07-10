"""Request context for passing metrics data between layers."""

from contextvars import ContextVar
from typing import Any

from ccproxy.utils.token_extractor import TokenUsage


# Context variables for request-scoped data
_model_context: ContextVar[str | None] = ContextVar("model", default=None)
_endpoint_context: ContextVar[str | None] = ContextVar("endpoint", default=None)
_token_usage_context: ContextVar[TokenUsage | None] = ContextVar(
    "token_usage", default=None
)
_streaming_context: ContextVar[bool] = ContextVar("streaming", default=False)


def set_model(model: str) -> None:
    """Set the model name in request context."""
    _model_context.set(model)


def get_model() -> str | None:
    """Get the model name from request context."""
    return _model_context.get()


def set_endpoint(endpoint: str) -> None:
    """Set the endpoint in request context."""
    _endpoint_context.set(endpoint)


def get_endpoint() -> str | None:
    """Get the endpoint from request context."""
    return _endpoint_context.get()


def set_token_usage(usage: TokenUsage | None) -> None:
    """Set token usage in request context."""
    _token_usage_context.set(usage)


def get_token_usage() -> TokenUsage | None:
    """Get token usage from request context."""
    return _token_usage_context.get()


def set_streaming(streaming: bool) -> None:
    """Set streaming flag in request context."""
    _streaming_context.set(streaming)


def is_streaming() -> bool:
    """Check if request is streaming."""
    return _streaming_context.get()


def clear_context() -> None:
    """Clear all context variables."""
    _model_context.set(None)
    _endpoint_context.set(None)
    _token_usage_context.set(None)
    _streaming_context.set(False)


class RequestContextManager:
    """Context manager for request-scoped data."""

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        streaming: bool = False,
    ) -> None:
        """Initialize context manager."""
        self.model = model
        self.endpoint = endpoint
        self.streaming = streaming
        self._tokens: list[Any] = []

    def __enter__(self) -> "RequestContextManager":
        """Enter context and set values."""
        if self.model:
            self._tokens.append(_model_context.set(self.model))
        if self.endpoint:
            self._tokens.append(_endpoint_context.set(self.endpoint))
        self._tokens.append(_streaming_context.set(self.streaming))
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and reset values."""
        import contextlib

        for token in reversed(self._tokens):
            with contextlib.suppress(Exception):
                token.var.reset(token)
        self._tokens.clear()
