"""Handler configuration for unified request handling."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ccproxy.adapters.base import APIAdapter


@runtime_checkable
class PluginTransformerProtocol(Protocol):
    """Protocol for plugin-based transformers with header and body methods."""

    def transform_headers(
        self, headers: dict[str, str], *args: Any, **kwargs: Any
    ) -> dict[str, str]:
        """Transform request headers."""
        ...

    def transform_body(self, body: Any) -> Any:
        """Transform request body."""
        ...


@dataclass(frozen=True)
class HandlerConfig:
    """Processing pipeline configuration for HTTP/streaming handlers.

    This simplified config only contains universal processing concerns,
    not plugin-specific parameters like session_id or access_token.

    Following the Parameter Object pattern, this groups related processing
    components while maintaining clean separation of concerns. Plugin-specific
    parameters should be passed directly as method parameters.
    """

    # Format conversion (e.g., OpenAI ↔ Anthropic)
    request_adapter: APIAdapter | None = None
    response_adapter: APIAdapter | None = None

    # Header/body transformation
    request_transformer: PluginTransformerProtocol | None = None
    response_transformer: PluginTransformerProtocol | None = None

    # Feature flag
    supports_streaming: bool = True
