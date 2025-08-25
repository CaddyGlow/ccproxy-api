"""Core proxy abstractions for handling HTTP and WebSocket connections."""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ccproxy.core.types import ProxyRequest, ProxyResponse


if TYPE_CHECKING:
    from ccproxy.core.http import HTTPClient


@runtime_checkable
class ProxyProtocol(Protocol):
    """Protocol defining the proxy interface."""

    async def forward(self, request: ProxyRequest) -> ProxyResponse:
        """Forward a request and return the response."""
        ...

    async def close(self) -> None:
        """Close any resources held by the proxy."""
        ...


class HTTPProxy(ProxyProtocol):
    """HTTP proxy implementation using HTTPClient abstractions."""

    def __init__(self, http_client: "HTTPClient") -> None:
        """Initialize with an HTTP client.

        Args:
            http_client: The HTTP client to use for requests
        """
        self.http_client = http_client

    async def forward(self, request: ProxyRequest) -> ProxyResponse:
        """Forward an HTTP request using the HTTP client.

        Args:
            request: The proxy request to forward

        Returns:
            The proxy response

        Raises:
            ProxyError: If the request cannot be forwarded
        """
        from ccproxy.core.errors import ProxyError
        from ccproxy.core.http import HTTPError

        try:
            # Convert ProxyRequest to HTTP client format
            body_bytes = None
            if request.body is not None:
                if isinstance(request.body, bytes):
                    body_bytes = request.body
                elif isinstance(request.body, str):
                    body_bytes = request.body.encode("utf-8")
                elif isinstance(request.body, dict):
                    import json

                    body_bytes = json.dumps(request.body).encode("utf-8")

            # Make the HTTP request
            status_code, headers, response_body = await self.http_client.request(
                method=request.method.value,
                url=request.url,
                headers=request.headers,
                body=body_bytes,
                timeout=request.timeout,
            )

            # Convert response body to appropriate format
            body: str | bytes | dict[str, Any] | None = response_body
            if response_body:
                # Try to decode as JSON if content-type suggests it
                content_type = headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        import json

                        body = json.loads(response_body.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Keep as bytes if JSON parsing fails
                        body = response_body
                elif "text/" in content_type:
                    try:
                        body = response_body.decode("utf-8")
                    except UnicodeDecodeError:
                        # Keep as bytes if text decoding fails
                        body = response_body

            return ProxyResponse(
                status_code=status_code,
                headers=headers,
                body=body,
            )

        except HTTPError as e:
            raise ProxyError(f"HTTP request failed: {e}") from e
        except (ValueError, TypeError) as e:
            raise ProxyError(f"Invalid request data: {e}") from e
        except Exception as e:
            raise ProxyError(f"Unexpected error during HTTP request: {e}") from e

    async def close(self) -> None:
        """Close HTTP proxy resources."""
        await self.http_client.close()
