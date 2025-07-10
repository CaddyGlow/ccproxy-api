"""HTTP-level transformers for proxy service."""

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


class HTTPRequestTransformer:
    """Basic HTTP request transformer for proxy service."""

    def transform_path(self, path: str, proxy_mode: str = "full") -> str:
        """Transform request path."""
        # Basic path transformation - pass through for now
        return path

    def create_proxy_headers(
        self, headers: dict[str, str], access_token: str, proxy_mode: str = "full"
    ) -> dict[str, str]:
        """Create proxy headers from original headers with Claude CLI identity."""
        proxy_headers = {}

        # Copy important headers
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key not in ["host", "authorization"]:
                proxy_headers[key] = value

        # Add authentication
        proxy_headers["Authorization"] = f"Bearer {access_token}"

        # Set content type if not present
        if "content-type" not in proxy_headers:
            proxy_headers["Content-Type"] = "application/json"

        # Critical Claude/Anthropic headers for tools and beta features
        proxy_headers["anthropic-beta"] = (
            "claude-code-20250219,oauth-2025-04-20,"
            "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"
        )
        proxy_headers["anthropic-version"] = "2023-06-01"
        proxy_headers["anthropic-dangerous-direct-browser-access"] = "true"

        # Claude CLI identity headers
        proxy_headers["x-app"] = "cli"
        proxy_headers["User-Agent"] = "claude-cli/1.0.43 (external, cli)"

        # Stainless SDK compatibility headers
        proxy_headers["X-Stainless-Retry-Count"] = "0"
        proxy_headers["X-Stainless-Timeout"] = "60"
        proxy_headers["X-Stainless-Lang"] = "js"
        proxy_headers["X-Stainless-Package-Version"] = "0.55.1"
        proxy_headers["X-Stainless-OS"] = "Linux"
        proxy_headers["X-Stainless-Arch"] = "x64"
        proxy_headers["X-Stainless-Runtime"] = "node"
        proxy_headers["X-Stainless-Runtime-Version"] = "v22.14.0"

        # Standard HTTP headers for proper API interaction
        proxy_headers["Connection"] = "keep-alive"
        proxy_headers["accept-language"] = "*"
        proxy_headers["sec-fetch-mode"] = "cors"
        proxy_headers["accept-encoding"] = "gzip, deflate"

        return proxy_headers

    def transform_request_body(
        self, body: bytes, path: str, proxy_mode: str = "full"
    ) -> bytes:
        """Transform request body."""
        # Basic body transformation - pass through for now
        return body


class HTTPResponseTransformer:
    """Basic HTTP response transformer for proxy service."""

    def transform_response_body(
        self, body: bytes, path: str, proxy_mode: str = "full"
    ) -> bytes:
        """Transform response body."""
        # Basic body transformation - pass through for now
        return body

    def transform_response_headers(
        self,
        headers: dict[str, str],
        path: str,
        content_length: int,
        proxy_mode: str = "full",
    ) -> dict[str, str]:
        """Transform response headers."""
        transformed_headers = {}

        # Copy important headers
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key not in ["content-length", "transfer-encoding"]:
                transformed_headers[key] = value

        # Set content length
        transformed_headers["Content-Length"] = str(content_length)

        # Add CORS headers
        transformed_headers["Access-Control-Allow-Origin"] = "*"
        transformed_headers["Access-Control-Allow-Headers"] = "*"
        transformed_headers["Access-Control-Allow-Methods"] = "*"

        return transformed_headers

    def _is_openai_request(self, path: str) -> bool:
        """Check if this is an OpenAI API request."""
        return "/openai/" in path or "/chat/completions" in path
