"""Core abstractions for the Claude Code Proxy API."""

from ccproxy.core.errors import (
    MiddlewareError,
    ProxyAuthenticationError,
    ProxyConnectionError,
    ProxyError,
    ProxyTimeoutError,
    TransformationError,
)
from ccproxy.core.middleware import (
    BaseMiddleware,
    CompositeMiddleware,
    MiddlewareChain,
    MiddlewareProtocol,
    NextMiddleware,
)
from ccproxy.core.proxy import (
    BaseProxy,
    HTTPProxy,
    ProxyProtocol,
    WebSocketProxy,
)
from ccproxy.core.transformers import (
    BaseTransformer,
    ChainedTransformer,
    RequestTransformer,
    ResponseTransformer,
    TransformerProtocol,
)
from ccproxy.core.types import (
    MiddlewareConfig,
    ProxyConfig,
    ProxyMethod,
    ProxyRequest,
    ProxyResponse,
    TransformContext,
)
from ccproxy.core.types import (
    ProxyProtocol as ProxyProtocolEnum,
)


__all__ = [
    # Proxy abstractions
    "BaseProxy",
    "HTTPProxy",
    "WebSocketProxy",
    "ProxyProtocol",
    # Transformer abstractions
    "BaseTransformer",
    "RequestTransformer",
    "ResponseTransformer",
    "TransformerProtocol",
    "ChainedTransformer",
    # Middleware abstractions
    "BaseMiddleware",
    "MiddlewareChain",
    "MiddlewareProtocol",
    "CompositeMiddleware",
    "NextMiddleware",
    # Error types
    "ProxyError",
    "TransformationError",
    "MiddlewareError",
    "ProxyConnectionError",
    "ProxyTimeoutError",
    "ProxyAuthenticationError",
    # Type definitions
    "ProxyRequest",
    "ProxyResponse",
    "TransformContext",
    "ProxyMethod",
    "ProxyProtocolEnum",
    "ProxyConfig",
    "MiddlewareConfig",
]
