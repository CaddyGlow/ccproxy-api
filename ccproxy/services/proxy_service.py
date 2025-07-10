"""Proxy service for orchestrating Claude API requests with business logic."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ccproxy.core.http import (
    BaseProxyClient,
    HTTPClient,
    get_proxy_url,
    get_ssl_context,
)
from ccproxy.core.http_transformers import (
    HTTPRequestTransformer,
    HTTPResponseTransformer,
)
from ccproxy.services.credentials.manager import CredentialsManager


logger = logging.getLogger(__name__)


class ProxyService:
    """Claude-specific proxy orchestration with business logic.

    This service orchestrates the complete proxy flow including:
    - Authentication management
    - Request/response transformations
    - Metrics collection (future)
    - Error handling and logging

    Pure HTTP forwarding is delegated to BaseProxyClient.
    """

    def __init__(
        self,
        proxy_client: BaseProxyClient,
        credentials_manager: CredentialsManager,
        proxy_mode: str = "full",
        target_base_url: str = "https://api.anthropic.com",
    ) -> None:
        """Initialize the proxy service.

        Args:
            proxy_client: HTTP client for pure forwarding
            credentials_manager: Authentication manager
            proxy_mode: Transformation mode - "minimal" or "full"
            target_base_url: Base URL for the target API
        """
        self.proxy_client = proxy_client
        self.credentials_manager = credentials_manager
        self.proxy_mode = proxy_mode
        self.target_base_url = target_base_url.rstrip("/")

        # Create concrete transformers
        self.request_transformer = HTTPRequestTransformer()
        self.response_transformer = HTTPResponseTransformer()

    async def handle_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes | None = None,
        query_params: dict[str, Any] | None = None,
        timeout: float = 240.0,
    ) -> tuple[int, dict[str, str], bytes] | StreamingResponse:
        """Handle a proxy request with full business logic orchestration.

        Args:
            method: HTTP method
            path: Request path (without /unclaude prefix)
            headers: Request headers
            body: Request body
            query_params: Query parameters
            timeout: Request timeout in seconds

        Returns:
            Tuple of (status_code, headers, body) or StreamingResponse for streaming

        Raises:
            HTTPException: If request fails
        """
        try:
            # 1. Authentication - get access token
            logger.debug("Retrieving OAuth access token...")
            access_token = await self._get_access_token()

            # 2. Request transformation
            logger.debug("Transforming request...")
            transformed_request = await self._transform_request(
                method, path, headers, body, query_params, access_token
            )

            # 3. Forward request using pure proxy client
            logger.debug(f"Forwarding request to: {transformed_request['url']}")

            # Check if this will be a streaming response
            if self._should_stream_response(transformed_request["headers"]):
                return await self._handle_streaming_request(
                    transformed_request, path, timeout
                )

            # Handle regular request
            (
                status_code,
                response_headers,
                response_body,
            ) = await self.proxy_client.forward(
                method=transformed_request["method"],
                url=transformed_request["url"],
                headers=transformed_request["headers"],
                body=transformed_request["body"],
                timeout=timeout,
            )

            # 4. Response transformation
            logger.debug("Transforming response...")
            transformed_response = await self._transform_response(
                status_code, response_headers, response_body, path
            )

            # 5. Metrics collection (future)
            await self._collect_metrics(transformed_request, transformed_response)

            return (
                transformed_response["status_code"],
                transformed_response["headers"],
                transformed_response["body"],
            )

        except Exception as e:
            logger.exception(f"Error in proxy request: {method} {path}")
            await self._handle_error(e, method, path)
            raise

    async def _get_access_token(self) -> str:
        """Get OAuth access token from credentials manager.

        Returns:
            Valid access token

        Raises:
            HTTPException: If no valid token is available
        """
        try:
            access_token = await self.credentials_manager.get_access_token()
            if not access_token:
                logger.error("No OAuth access token available")

                # Try to get more details about credential status
                try:
                    validation = await self.credentials_manager.validate()
                    if validation.get("valid"):
                        logger.debug(
                            "Found credentials but access token is invalid/expired"
                        )
                        logger.debug(f"Expired: {validation.get('expired')}")
                        logger.debug(f"Expires at: {validation.get('expires_at')}")
                    else:
                        logger.debug(f"Credentials invalid: {validation.get('error')}")
                except Exception as e:
                    logger.debug(f"Could not check credential details: {e}")

                raise HTTPException(
                    status_code=401,
                    detail="No valid OAuth credentials found. Please run 'ccproxy auth login'.",
                )

            logger.debug("Successfully retrieved OAuth access token")
            return access_token

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            raise HTTPException(
                status_code=401,
                detail="Authentication failed",
            ) from e

    async def _transform_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes | None,
        query_params: dict[str, Any] | None,
        access_token: str,
    ) -> dict[str, Any]:
        """Transform request using the transformer pipeline.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            body: Request body
            query_params: Query parameters
            access_token: OAuth access token

        Returns:
            Transformed request data
        """
        # Transform path
        transformed_path = self.request_transformer.transform_path(
            path, self.proxy_mode
        )
        target_url = f"{self.target_base_url}{transformed_path}"

        # Add beta=true query parameter for /v1/messages requests if not already present
        if transformed_path == "/v1/messages":
            if query_params is None:
                query_params = {}
            elif "beta" not in query_params:
                query_params = dict(query_params)  # Make a copy

            if "beta" not in query_params:
                query_params["beta"] = "true"
                logger.debug("Added beta=true query parameter to /v1/messages request")

        # Transform body first (as it might change size)
        proxy_body = None
        if body:
            proxy_body = self.request_transformer.transform_request_body(
                body, path, self.proxy_mode
            )

        # Transform headers (and update Content-Length if body changed)
        proxy_headers = self.request_transformer.create_proxy_headers(
            headers, access_token, self.proxy_mode
        )

        # Update Content-Length if body was transformed and size changed
        if proxy_body and body and len(proxy_body) != len(body):
            # Remove any existing content-length headers (case-insensitive)
            proxy_headers = {
                k: v for k, v in proxy_headers.items() if k.lower() != "content-length"
            }
            proxy_headers["Content-Length"] = str(len(proxy_body))
        elif proxy_body and not body:
            # New body was created where none existed
            proxy_headers["Content-Length"] = str(len(proxy_body))

        # Add query parameters to URL if present
        if query_params:
            import urllib.parse

            query_string = urllib.parse.urlencode(query_params)
            target_url = f"{target_url}?{query_string}"

        return {
            "method": method,
            "url": target_url,
            "headers": proxy_headers,
            "body": proxy_body,
        }

    async def _transform_response(
        self,
        status_code: int,
        headers: dict[str, str],
        body: bytes,
        original_path: str,
    ) -> dict[str, Any]:
        """Transform response using the transformer pipeline.

        Args:
            status_code: HTTP status code
            headers: Response headers
            body: Response body
            original_path: Original request path for context

        Returns:
            Transformed response data
        """
        transformed_body = self.response_transformer.transform_response_body(
            body, original_path, self.proxy_mode
        )

        transformed_headers = self.response_transformer.transform_response_headers(
            headers, original_path, len(transformed_body), self.proxy_mode
        )

        return {
            "status_code": status_code,
            "headers": transformed_headers,
            "body": transformed_body,
        }

    def _should_stream_response(self, headers: dict[str, str]) -> bool:
        """Check if response should be streamed based on request headers.

        Args:
            headers: Request headers

        Returns:
            True if response should be streamed
        """
        # Check if client requested streaming
        accept_header = headers.get("accept", "").lower()
        return "text/event-stream" in accept_header or "stream" in accept_header

    async def _handle_streaming_request(
        self,
        request_data: dict[str, Any],
        original_path: str,
        timeout: float,
    ) -> StreamingResponse:
        """Handle streaming request with transformation.

        Args:
            request_data: Transformed request data
            original_path: Original request path for context
            timeout: Request timeout

        Returns:
            StreamingResponse
        """

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            try:
                # Use httpx directly for streaming since we need the stream context manager
                import os
                from pathlib import Path

                import httpx

                # Get proxy and SSL settings
                proxy_url = get_proxy_url()
                verify = get_ssl_context()

                async with (
                    httpx.AsyncClient(
                        timeout=timeout, proxy=proxy_url, verify=verify
                    ) as client,
                    client.stream(
                        method=request_data["method"],
                        url=request_data["url"],
                        headers=request_data["headers"],
                        content=request_data["body"],
                    ) as response,
                ):
                    # Check for errors
                    if response.status_code >= 400:
                        error_content = await response.aread()
                        logger.info(f"Streaming error {response.status_code}")
                        logger.debug(
                            f"Streaming error detail: {error_content.decode('utf-8', errors='replace')}"
                        )
                        yield error_content
                        return

                    # Transform streaming response
                    is_openai = self.response_transformer._is_openai_request(
                        original_path
                    )

                    if is_openai:
                        # Transform Anthropic SSE to OpenAI SSE format using adapter
                        logger.info(
                            f"Transforming Anthropic SSE to OpenAI format for {original_path}"
                        )

                        async for (
                            transformed_chunk
                        ) in self._transform_anthropic_to_openai_stream(
                            response, original_path
                        ):
                            yield transformed_chunk
                    else:
                        # Stream as-is for Anthropic endpoints
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk

            except Exception as e:
                logger.exception("Error in streaming response")
                error_message = f'data: {{"error": "Streaming error: {str(e)}"}}\\n\\n'
                yield error_message.encode("utf-8")

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    def _get_proxy_url(self) -> str | None:
        """Get proxy URL from environment variables.

        Returns:
            str or None: Proxy URL if any proxy is set
        """
        import os

        # Check for standard proxy environment variables
        # For HTTPS requests, prioritize HTTPS_PROXY
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        all_proxy = os.environ.get("ALL_PROXY")
        http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")

        proxy_url = https_proxy or all_proxy or http_proxy

        if proxy_url:
            logger.debug(f"Using proxy: {proxy_url}")

        return proxy_url

    def _get_ssl_context(self) -> str | bool:
        """Get SSL context configuration from environment variables.

        Returns:
            SSL verification configuration:
            - Path to CA bundle file
            - True for default verification
            - False to disable verification (insecure)
        """
        import os
        from pathlib import Path

        # Check for custom CA bundle
        ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get(
            "SSL_CERT_FILE"
        )

        # Check if SSL verification should be disabled (NOT RECOMMENDED)
        ssl_verify = os.environ.get("SSL_VERIFY", "true").lower()

        if ca_bundle and Path(ca_bundle).exists():
            logger.info(f"Using custom CA bundle: {ca_bundle}")
            return ca_bundle
        elif ssl_verify in ("false", "0", "no"):
            logger.warning("SSL verification disabled - this is insecure!")
            return False
        else:
            logger.debug("Using default SSL verification")
            return True

    async def _collect_metrics(
        self,
        request_data: dict[str, Any],
        response_data: dict[str, Any],
    ) -> None:
        """Collect metrics for the request/response.

        Args:
            request_data: Request data
            response_data: Response data
        """
        # Placeholder for future metrics collection
        # This will integrate with MetricsCollector when implemented
        logger.debug("Metrics collection placeholder")

    async def _handle_error(
        self,
        error: Exception,
        method: str,
        path: str,
    ) -> None:
        """Handle errors in proxy requests.

        Args:
            error: The error that occurred
            method: HTTP method
            path: Request path
        """
        # Convert known exceptions to HTTP exceptions
        if isinstance(error, httpx.TimeoutException):
            raise HTTPException(status_code=504, detail="Gateway timeout") from error
        elif isinstance(error, httpx.HTTPStatusError):
            raise HTTPException(
                status_code=error.response.status_code, detail=str(error)
            ) from error
        elif isinstance(error, HTTPException):
            # Re-raise HTTP exceptions as-is
            raise
        else:
            # Generic server error
            raise HTTPException(
                status_code=500, detail="Internal server error"
            ) from error

    async def _transform_anthropic_to_openai_stream(
        self, response: httpx.Response, original_path: str
    ) -> AsyncGenerator[bytes, None]:
        """Transform Anthropic SSE stream to OpenAI SSE format.

        Args:
            response: Streaming response from Anthropic
            original_path: Original request path for context

        Yields:
            Transformed OpenAI SSE format chunks
        """
        # Use OpenAI streaming formatter from adapters
        from ccproxy.adapters.openai.streaming import OpenAISSEFormatter

        # Simple streaming transformation for now
        # TODO: Implement proper stream transformation using adapters
        async for chunk in response.aiter_bytes():
            if chunk:
                yield chunk

    async def close(self) -> None:
        """Close any resources held by the proxy service."""
        if self.proxy_client:
            await self.proxy_client.close()
        if self.credentials_manager:
            await self.credentials_manager.__aexit__(None, None, None)
