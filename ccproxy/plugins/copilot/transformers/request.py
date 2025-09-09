"""Request transformer for GitHub Copilot API."""

from typing import Any
from uuid import uuid4

from ccproxy.core.logging import get_plugin_logger
from ccproxy.utils.headers import filter_request_headers

from ..config import CopilotConfig


logger = get_plugin_logger()


class CopilotRequestTransformer:
    """Transform requests for GitHub Copilot API."""

    def __init__(self, config: CopilotConfig | None = None):
        """Initialize request transformer.

        Args:
            config: Plugin configuration
        """
        self.config = config or CopilotConfig()

    def transform_headers(
        self, headers: dict[str, str], access_token: str | None = None, **kwargs: Any
    ) -> dict[str, str]:
        """Transform request headers for Copilot API.

        Args:
            headers: Original headers
            access_token: Copilot access token
            **kwargs: Additional context

        Returns:
            Transformed headers
        """
        # Use common filter utility (don't preserve auth since we'll add our own)
        transformed = filter_request_headers(headers, preserve_auth=False)

        # Add required Copilot headers
        copilot_headers = self.config.api_headers.copy()

        # Add authentication header if token provided
        if access_token:
            copilot_headers["authorization"] = f"Bearer {access_token}"

        # Add unique request ID
        copilot_headers["x-request-id"] = str(uuid4())

        # Merge headers (copilot headers take precedence)
        transformed.update(copilot_headers)

        logger.debug(
            "headers_transformed",
            original_count=len(headers),
            transformed_count=len(transformed),
            has_auth=bool(access_token),
            keys=list(transformed.keys()),
        )

        return transformed

    def transform_body(self, body: bytes | str | dict[str, Any]) -> bytes:
        """Transform request body for Copilot API.

        Args:
            body: Original request body

        Returns:
            Transformed body as bytes
        """
        # For Copilot, we usually pass through the body as-is
        # since format conversion is handled by format adapters

        if isinstance(body, dict):
            import json

            return json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            return body.encode("utf-8")
        elif isinstance(body, bytes):
            return body
        else:
            logger.warning(
                "unexpected_body_type",
                body_type=type(body).__name__,
            )
            return str(body).encode("utf-8")

    def get_target_url(self, endpoint: str, base_url: str | None = None) -> str:
        """Get target URL for Copilot API endpoint.

        Args:
            endpoint: API endpoint path
            base_url: Base URL (uses config default if not provided)

        Returns:
            Complete target URL
        """
        if base_url is None:
            base_url = self.config.provider.get_base_url()

        # Ensure base_url doesn't end with slash and endpoint starts with slash
        base_url = base_url.rstrip("/")
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        target_url = f"{base_url}{endpoint}"

        logger.debug(
            "target_url_generated",
            base_url=base_url,
            endpoint=endpoint,
            target_url=target_url,
        )

        return target_url

    def prepare_request_context(
        self,
        endpoint: str,
        method: str,
        headers: dict[str, str],
        body: bytes,
        access_token: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare complete request context for proxy service.

        Args:
            endpoint: API endpoint
            method: HTTP method
            headers: Request headers
            body: Request body
            access_token: Copilot access token
            **kwargs: Additional context

        Returns:
            Request context dictionary
        """
        # Transform headers and body
        transformed_headers = self.transform_headers(headers, access_token, **kwargs)
        transformed_body = self.transform_body(body)

        # Get target URL
        target_url = self.get_target_url(endpoint)

        context = {
            "method": method,
            "url": target_url,
            "headers": transformed_headers,
            "body": transformed_body,
            "endpoint": endpoint,
            "provider": "copilot",
            "account_type": self.config.provider.account_type,
        }

        # Add optional context
        context.update(kwargs)

        logger.debug(
            "request_context_prepared",
            method=method,
            endpoint=endpoint,
            url=target_url,
            body_size=len(transformed_body),
            header_count=len(transformed_headers),
        )

        return context
