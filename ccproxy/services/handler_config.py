"""Handler configuration for request handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ccproxy.services.adapters.format_context import FormatContext


if TYPE_CHECKING:
    from ccproxy.services.adapters.format_adapter import FormatAdapterProtocol


@runtime_checkable
class PluginTransformerProtocol(Protocol):
    """Protocol for plugin-based transformers with header and body methods."""

    def transform_headers(
        self, headers: dict[str, str], *args: Any, **kwargs: Any
    ) -> dict[str, str]:
        """Transform request headers."""
        ...


@runtime_checkable
class SSEParserProtocol(Protocol):
    """Protocol for SSE parsers to extract a final JSON response.

    Implementations should return a parsed dict for the final response, or
    None if no final response could be determined.
    """

    def __call__(
        self, raw: str
    ) -> dict[str, Any] | None:  # pragma: no cover - protocol
        ...

    def transform_body(self, body: Any) -> Any:
        """Transform request body."""
        ...


@dataclass(frozen=True)
class HandlerConfig:
    """Processing pipeline configuration for HTTP/streaming handlers.

    This config only contains universal processing concerns,
    not plugin-specific parameters like session_id or access_token.

    Following the Parameter Object pattern, this groups related processing
    components while maintaining clean separation of concerns. Plugin-specific
    parameters should be passed directly as method parameters.
    """

    # Format conversion (e.g., OpenAI ↔ Anthropic)
    request_adapter: FormatAdapterProtocol | None = None
    response_adapter: FormatAdapterProtocol | None = None

    # Header/body transformation
    request_transformer: PluginTransformerProtocol | None = None
    response_transformer: PluginTransformerProtocol | None = None

    # Feature flag
    supports_streaming: bool = True

    # Header case preservation toggle for upstream requests
    # When True, the HTTP handler will not canonicalize header names and will
    # forward them with their original casing/order as produced by transformers.
    preserve_header_case: bool = False

    # Optional SSE parser provided by plugins that return SSE streams
    sse_parser: SSEParserProtocol | None = None

    # Format context for adapter selection
    format_context: FormatContext | None = None
