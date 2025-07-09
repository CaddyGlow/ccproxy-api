"""Core type definitions for the proxy system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class ProxyMethod(str, Enum):
    """HTTP methods supported by the proxy."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    CONNECT = "CONNECT"
    TRACE = "TRACE"


class ProxyProtocol(str, Enum):
    """Protocols supported by the proxy."""

    HTTP = "http"
    HTTPS = "https"
    WS = "ws"
    WSS = "wss"


@dataclass
class ProxyRequest:
    """Represents a proxy request."""

    method: ProxyMethod
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    body: str | bytes | dict[str, Any] | None = None
    protocol: ProxyProtocol = ProxyProtocol.HTTPS
    timeout: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize the request."""
        if isinstance(self.method, str):
            self.method = ProxyMethod(self.method.upper())
        if isinstance(self.protocol, str):
            self.protocol = ProxyProtocol(self.protocol.lower())


@dataclass
class ProxyResponse:
    """Represents a proxy response."""

    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: str | bytes | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if the response indicates success."""
        return 200 <= self.status_code < 300

    @property
    def is_error(self) -> bool:
        """Check if the response indicates an error."""
        return self.status_code >= 400


@dataclass
class TransformContext:
    """Context passed to transformers during transformation."""

    request: ProxyRequest | None = None
    response: ProxyResponse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata."""
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in metadata."""
        self.metadata[key] = value


class ProxyConfig(BaseModel):
    """Configuration for proxy behavior."""

    timeout: float = Field(default=30.0, description="Default timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )
    verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates"
    )
    follow_redirects: bool = Field(
        default=True, description="Whether to follow redirects"
    )
    max_redirects: int = Field(
        default=10, description="Maximum number of redirects to follow"
    )

    class Config:
        """Pydantic configuration."""

        extra = "forbid"


class MiddlewareConfig(BaseModel):
    """Configuration for middleware behavior."""

    enabled: bool = Field(default=True, description="Whether the middleware is enabled")
    priority: int = Field(
        default=0, description="Middleware execution priority (lower = earlier)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional middleware configuration"
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"
