"""Reverse proxy service for forwarding requests to api.anthropic.com."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from claude_code_proxy.services.credentials import CredentialsManager
from claude_code_proxy.services.request_transformer import RequestTransformer
from claude_code_proxy.services.response_transformer import ResponseTransformer
from claude_code_proxy.utils.logging import get_logger


logger = get_logger(__name__)


class ReverseProxyService:
    """Service for proxying requests to Anthropic API with transformations."""

    def __init__(
        self,
        target_base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
        proxy_mode: str = "full",
        credentials_manager: CredentialsManager | None = None,
    ):
        """Initialize the reverse proxy service.

        Args:
            target_base_url: Base URL for the target API
            timeout: Request timeout in seconds
            proxy_mode: Transformation mode - "minimal", "full", or "passthrough"
            credentials_manager: Optional credentials manager instance
        """
        self.target_base_url = target_base_url.rstrip("/")
        self.timeout = timeout
        self.proxy_mode = proxy_mode
        self.request_transformer = RequestTransformer()
        self.response_transformer = ResponseTransformer()
        self._credentials_manager = credentials_manager

    async def proxy_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, str], bytes] | StreamingResponse:
        """Proxy a request to the target API.

        Args:
            method: HTTP method
            path: Request path (without /unclaude prefix)
            headers: Request headers
            body: Request body
            query_params: Query parameters

        Returns:
            Tuple of (status_code, headers, body) or StreamingResponse for streaming

        Raises:
            HTTPException: If proxy request fails
        """
        try:
            # Get OAuth access token
            logger.debug("Attempting to retrieve OAuth access token...")

            # Use provided credentials manager or create a temporary one
            if self._credentials_manager:
                credentials_manager = self._credentials_manager
            else:
                credentials_manager = CredentialsManager()

            try:
                access_token = await credentials_manager.get_access_token()
            except Exception as e:
                logger.error(f"Failed to get access token: {e}")
                access_token = None

            if not access_token:
                logger.error("No OAuth access token available")
                # Try to get more details about credential status
                try:
                    validation = await credentials_manager.validate()
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
            logger.debug(f"Access token (first 20 chars): {access_token[:20]}...")

            # Transform request path (remove /openai prefix)
            transformed_path = self.request_transformer.transform_path(
                path, self.proxy_mode
            )
            target_url = f"{self.target_base_url}{transformed_path}"

            # Add beta=true query parameter for /v1/messages requests if not already present
            if transformed_path == "/v1/messages":
                if query_params is None:
                    query_params = {}
                elif "beta" not in query_params:
                    query_params = dict(
                        query_params
                    )  # Make a copy to avoid modifying original

                if "beta" not in query_params:
                    query_params["beta"] = "true"
                    logger.debug(
                        "Added beta=true query parameter to /v1/messages request"
                    )

            proxy_headers = self.request_transformer.create_proxy_headers(
                headers, access_token, self.proxy_mode
            )

            # Log the headers being sent (safely)
            logger.debug("Request headers prepared:")
            for key, value in proxy_headers.items():
                if key.lower() == "authorization":
                    # Show only first part of auth header for security
                    logger.debug(f"  - {key}: {value[:20]}...")
                else:
                    logger.debug(f"  - {key}: {value}")

            # Transform request body if present
            proxy_body = None
            if body:
                proxy_body = self.request_transformer.transform_request_body(
                    body, path, self.proxy_mode
                )

            logger.debug(f"Making request to: {method} {target_url}")
            logger.debug(
                f"Request body size: {len(proxy_body) if proxy_body else 0} bytes"
            )

            # Make the request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug(f"Sending {method} request to Anthropic API...")
                response = await client.request(
                    method=method,
                    url=target_url,
                    headers=proxy_headers,
                    content=proxy_body,
                    params=query_params,
                )

                # Log response status
                logger.debug(
                    f"Anthropic API response: {response.status_code} {response.reason_phrase}"
                )
                if response.status_code >= 400:
                    logger.info(
                        f"API error: {response.status_code} {response.reason_phrase}"
                    )
                    logger.debug(
                        f"API error response headers: {dict(response.headers)}"
                    )
                    # Log first part of error response for debugging
                    try:
                        error_content = response.content[:500]  # First 500 bytes
                        logger.debug(
                            f"Error response preview: {error_content.decode('utf-8', errors='replace')}"
                        )
                    except Exception:
                        logger.debug("Could not read error response content")

            # Handle streaming responses
            if self._is_streaming_response(response):
                return await self._handle_streaming_response(
                    method, target_url, proxy_headers, proxy_body, query_params, path
                )

            # Handle regular responses
            response_body = self.response_transformer.transform_response_body(
                response.content, path, self.proxy_mode
            )
            response_headers = self.response_transformer.transform_response_headers(
                dict(response.headers), path, len(response_body), self.proxy_mode
            )

            return response.status_code, response_headers, response_body

        except httpx.TimeoutException as e:
            logger.error(f"Timeout proxying {method} {path}")
            raise HTTPException(status_code=504, detail="Gateway timeout") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error proxying {method} {path}: {e}")
            raise HTTPException(
                status_code=e.response.status_code, detail=str(e)
            ) from e
        except Exception as e:
            logger.exception(f"Error proxying {method} {path}")
            raise HTTPException(status_code=500, detail="Internal server error") from e

    def _is_streaming_response(self, response: httpx.Response) -> bool:
        """Check if response is a streaming response.

        Args:
            response: HTTP response

        Returns:
            True if response is streaming
        """
        content_type = response.headers.get("content-type", "")
        return "text/event-stream" in content_type or "stream" in content_type

    async def _handle_streaming_response(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        params: dict[str, Any] | None,
        original_path: str,
    ) -> StreamingResponse:
        """Handle streaming response from the target API.

        Args:
            method: HTTP method
            url: Target URL
            headers: Request headers
            body: Request body
            params: Query parameters
            original_path: Original request path for context

        Returns:
            StreamingResponse
        """

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            try:
                async with (
                    httpx.AsyncClient(timeout=self.timeout) as client,
                    client.stream(
                        method=method,
                        url=url,
                        headers=headers,
                        content=body,
                        params=params,
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

                    # Stream the response
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk

            except Exception as e:
                logger.exception("Error in streaming response")
                error_message = f'data: {{"error": "Streaming error: {str(e)}"}}\n\n'
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
