"""Reverse proxy service for forwarding requests to api.anthropic.com."""

import asyncio
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Union

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from ccproxy.services.credentials import CredentialsManager
from ccproxy.services.request_transformer import RequestTransformer
from ccproxy.services.response_transformer import ResponseTransformer
from ccproxy.utils.logging import get_logger
from ccproxy.utils.openai import is_openai_request


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
            proxy_mode: Transformation mode - "minimal" or "full"
            credentials_manager: Optional credentials manager instance
        """
        self.target_base_url = target_base_url.rstrip("/")
        self.timeout = timeout
        self.proxy_mode = proxy_mode
        self.request_transformer = RequestTransformer()
        self.response_transformer = ResponseTransformer()
        self._credentials_manager = credentials_manager

    def _get_proxy_url(self) -> str | None:
        """Get proxy URL from environment variables.

        Returns:
            str or None: Proxy URL if any proxy is set
        """
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
            logger.debug(f"Request headers prepared {len(proxy_headers)}")
            # for key, value in proxy_headers.items():
            #     if key.lower() == "authorization":
            #         # Show only first part of auth header for security
            #         logger.debug(f"  - {key}: {value[:20]}...")
            #     else:
            #         logger.debug(f"  - {key}: {value}")

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
            proxy_url = self._get_proxy_url()
            verify = self._get_ssl_context()

            logger.debug(f"HTTP client config - proxy: {proxy_url}, verify: {verify}")

            async with httpx.AsyncClient(
                timeout=self.timeout, proxy=proxy_url, verify=verify
            ) as client:
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
                proxy_url = self._get_proxy_url()
                verify = self._get_ssl_context()
                async with (
                    httpx.AsyncClient(
                        timeout=self.timeout, proxy=proxy_url, verify=verify
                    ) as client,
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

                    # Check if this is an OpenAI endpoint that needs transformation
                    is_openai = self.response_transformer._is_openai_request(
                        original_path
                    )
                    logger.debug(
                        f"Streaming response for path: {original_path}, is_openai: {is_openai}, mode: {self.proxy_mode}"
                    )

                    if is_openai:
                        # Transform Anthropic SSE to OpenAI SSE format
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
                        # Stream the response as-is for Anthropic endpoints
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

    async def _transform_anthropic_to_openai_stream(
        self, response: httpx.Response, original_path: str
    ) -> AsyncGenerator[bytes, None]:
        """Transform Anthropic SSE stream to OpenAI SSE format.

        Uses the unified stream transformer for consistent behavior across all OpenAI
        streaming implementations. Supports tool calls, usage info, and thinking blocks.

        Args:
            response: The streaming response from Anthropic
            original_path: Original request path for extracting model info

        Yields:
            Transformed OpenAI SSE format chunks
        """
        from ccproxy.services.stream_transformer import (
            OpenAIStreamTransformer,
            StreamingConfig,
        )

        # Configure streaming for reverse proxy
        # Note: Tool calls and usage are now supported!
        config = StreamingConfig(
            enable_text_chunking=False,  # Don't chunk text in reverse proxy
            enable_tool_calls=True,  # Now we support tool calls
            enable_usage_info=True,  # Now we support usage info
            chunk_delay_ms=0,  # No artificial delays
            chunk_size_words=1,
        )

        # Extract model from request body if possible, otherwise use default
        # For reverse proxy, we might not have access to the original request model
        # So we'll use a generic model name
        model = "gpt-4o"  # Default model name for OpenAI format

        # Create transformer
        transformer = OpenAIStreamTransformer.from_sse_stream(
            response,
            model=model,
            config=config,
        )

        # Transform and yield as bytes
        async for chunk in transformer.transform():
            yield chunk.encode("utf-8")
