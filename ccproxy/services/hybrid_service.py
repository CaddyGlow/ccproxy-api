"""Hybrid service combining proxy and SDK functionality.

This module provides a hybrid service that can dynamically choose between
proxy and SDK modes based on configuration and request requirements.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Optional, Union

from ccproxy.core.types import ProxyRequest, ProxyResponse
from ccproxy.services.claude_sdk_service import ClaudeSDKService
from ccproxy.services.proxy_service import ProxyService


logger = logging.getLogger(__name__)


class HybridService:
    """Hybrid service that orchestrates between proxy and SDK approaches.

    This service implements the business logic for choosing between proxy
    and SDK modes based on configuration, request characteristics, and
    runtime conditions.
    """

    def __init__(
        self,
        claude_sdk_service: ClaudeSDKService,
        proxy_service: ProxyService,
        default_proxy_mode: str = "proxy",
    ):
        """Initialize the hybrid service.

        Args:
            claude_sdk_service: Claude SDK client instance
            proxy_service: Reverse proxy service instance
            default_proxy_mode: Default proxy mode setting
        """
        self.claude_sdk_service = claude_sdk_service
        self.proxy_service = proxy_service
        self.default_proxy_mode = default_proxy_mode

        # Service selection strategy
        self._use_sdk_for_tools = True
        self._use_proxy_for_simple_requests = True
        self._sdk_timeout_threshold = 30.0

        logger.info("Hybrid service initialized")

    async def handle_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes | None = None,
        query_params: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> tuple[int, dict[str, str], bytes] | AsyncIterator[dict[str, Any]]:
        """Handle a request using the appropriate service.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            body: Request body
            query_params: Query parameters
            user_id: Optional user identifier

        Returns:
            Response from the selected service

        Raises:
            ValueError: If no suitable service is available
        """
        # Parse request body for decision making
        request_body = self._parse_request_body(body)

        # Determine which service to use based on request characteristics
        should_use_sdk = await self._should_use_sdk(request_body, path)

        if should_use_sdk:
            return await self._handle_with_sdk(
                method, path, headers, request_body, query_params, user_id
            )
        else:
            return await self._handle_with_proxy(
                method, path, headers, body, query_params, user_id
            )

    def _parse_request_body(self, body: bytes | None) -> dict[str, Any]:
        """Parse request body from bytes to dict.

        Args:
            body: Request body as bytes

        Returns:
            Parsed request body as dict
        """
        if not body:
            return {}

        try:
            result = json.loads(body.decode("utf-8"))
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    async def _should_use_sdk(self, request_body: dict[str, Any], path: str) -> bool:
        """Determine whether to use SDK or proxy for this request.

        Args:
            request_body: The parsed request body to analyze
            path: The API path being requested

        Returns:
            True if SDK should be used, False if proxy should be used
        """
        # Check if it's a messages/chat endpoint
        if not (path.endswith("/messages") or path.endswith("/chat/completions")):
            # Non-chat requests should use proxy
            return False

        # Use SDK if tools are present
        if self._use_sdk_for_tools and self._has_tools(request_body):
            logger.debug("Using SDK for tool-enabled request")
            return True

        # Use SDK if streaming is requested and we have SDK preference
        if self._has_streaming(request_body) and self._prefer_sdk_for_streaming():
            logger.debug("Using SDK for streaming request")
            return True

        # Use proxy for simple requests if configured
        if self._use_proxy_for_simple_requests and self._is_simple_request(
            request_body
        ):
            logger.debug("Using proxy for simple request")
            return False

        # Default to SDK for complex requests
        logger.debug("Using SDK for complex request")
        return True

    def _has_tools(self, request_body: dict[str, Any]) -> bool:
        """Check if the request includes tool usage.

        Args:
            request_body: The request body to analyze

        Returns:
            True if tools are present in the request
        """
        # Check for standard OpenAI tools field
        if (
            "tools" in request_body
            and isinstance(request_body["tools"], list)
            and len(request_body["tools"]) > 0
        ):
            return True

        # Check for Claude Code specific tools field
        return bool("allowed_tools" in request_body and isinstance(request_body["allowed_tools"], list) and len(request_body["allowed_tools"]) > 0)

    def _has_streaming(self, request_body: dict[str, Any]) -> bool:
        """Check if the request is for streaming.

        Args:
            request_body: The request body to analyze

        Returns:
            True if streaming is requested
        """
        stream_value = request_body.get("stream", False)
        return bool(stream_value)

    def _prefer_sdk_for_streaming(self) -> bool:
        """Check if SDK is preferred for streaming requests.

        Returns:
            True if SDK should be used for streaming
        """
        # This could be based on configuration or runtime conditions
        return self.default_proxy_mode == "sdk"

    def _is_simple_request(self, request_body: dict[str, Any]) -> bool:
        """Check if this is a simple request suitable for proxy.

        Args:
            request_body: The request body to analyze

        Returns:
            True if this is a simple request
        """
        # Simple requests are those without tools, function calling, or complex features
        return (
            not self._has_tools(request_body)
            and not request_body.get("function_call")
            and not request_body.get("functions")
            and len(request_body.get("messages", []))
            < 10  # Arbitrary complexity threshold
        )

    async def _handle_with_sdk(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        request_body: dict[str, Any],
        query_params: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> tuple[int, dict[str, str], bytes] | AsyncIterator[dict[str, Any]]:
        """Handle request using the Claude SDK.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            request_body: Parsed request body
            query_params: Query parameters
            user_id: Optional user identifier

        Returns:
            Response from the Claude SDK
        """
        logger.debug(f"Handling request with SDK: {path}")

        # Extract messages and options
        messages = request_body.get("messages", [])
        stream = request_body.get("stream", False)

        # Create default options - this needs to be properly implemented
        # For now, we'll create a basic options dict
        from ccproxy.core.async_utils import patched_typing

        with patched_typing():
            from claude_code_sdk import ClaudeCodeOptions

        options = ClaudeCodeOptions()

        # Create completion using SDK
        model = request_body.get("model", "claude-3-5-sonnet-20241022")  # Default model
        sdk_response = await self.claude_sdk_service.create_completion(
            messages=messages,
            model=model,
            stream=stream,
        )

        if stream:
            # For streaming, return the async iterator directly
            # This should be type: AsyncIterator[dict[str, Any]]
            return sdk_response  # type: ignore[return-value]
        else:
            # For non-streaming, convert to proxy response format
            response_body = json.dumps(sdk_response).encode("utf-8")
            response_headers = {
                "content-type": "application/json",
                "content-length": str(len(response_body)),
            }
            return 200, response_headers, response_body

    async def _handle_with_proxy(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes | None,
        query_params: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> tuple[int, dict[str, str], bytes] | AsyncIterator[dict[str, Any]]:
        """Handle request using the reverse proxy.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            body: Request body
            query_params: Query parameters
            user_id: Optional user identifier

        Returns:
            Response from the reverse proxy
        """
        logger.debug(f"Handling request with proxy: {path}")

        # Forward request to reverse proxy
        proxy_response = await self.proxy_service.handle_request(
            method=method,
            path=path,
            headers=headers,
            body=body,
            query_params=query_params,
        )

        # Handle different response types from reverse proxy
        from fastapi.responses import StreamingResponse

        if isinstance(proxy_response, StreamingResponse):
            # Convert StreamingResponse to AsyncIterator for consistency
            async def stream_generator() -> AsyncIterator[dict[str, Any]]:
                async for chunk in proxy_response.body_iterator:
                    if isinstance(chunk, bytes):
                        try:
                            # Try to parse as JSON
                            chunk_str = chunk.decode("utf-8")
                            if chunk_str.strip():
                                yield {"data": chunk_str}
                        except UnicodeDecodeError:
                            yield {"data": chunk.hex()}
                    else:
                        yield {"data": str(chunk)}

            return stream_generator()
        else:
            # Return tuple response as-is
            return proxy_response

    async def health_check(self) -> dict[str, Any]:
        """Check the health of both underlying services.

        Returns:
            Health status of the hybrid service
        """
        # Check SDK health
        sdk_health = True
        sdk_error = None
        try:
            await self.claude_sdk_service.validate_health()
        except Exception as e:
            sdk_health = False
            sdk_error = str(e)

        # Check proxy health (if available)
        proxy_health = True
        proxy_error = None
        try:
            # Proxy service doesn't have a health check method in the current implementation
            # This would need to be added if required
            pass
        except Exception as e:
            proxy_health = False
            proxy_error = str(e)

        return {
            "hybrid_service": {
                "status": "healthy" if (sdk_health or proxy_health) else "unhealthy",
                "sdk": {
                    "status": "healthy" if sdk_health else "unhealthy",
                    "error": sdk_error,
                },
                "proxy": {
                    "status": "healthy" if proxy_health else "unhealthy",
                    "error": proxy_error,
                },
            }
        }

    async def close(self) -> None:
        """Close the hybrid service and its underlying services."""
        await self.claude_sdk_service.close()
        # Reverse proxy service doesn't have a close method in the current implementation
        logger.info("Hybrid service closed")
