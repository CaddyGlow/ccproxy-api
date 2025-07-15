"""Proxy service for orchestrating Claude API requests with business logic."""

import json
import logging
import os
import time
import urllib.parse
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import httpx
import structlog
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from ccproxy.core.http import BaseProxyClient
from ccproxy.core.http_transformers import (
    HTTPRequestTransformer,
    HTTPResponseTransformer,
)
from ccproxy.observability import (
    PrometheusMetrics,
    get_metrics,
    request_context,
    timed_operation,
)
from ccproxy.services.credentials.manager import CredentialsManager


if TYPE_CHECKING:
    from ccproxy.observability.context import RequestContext


class RequestData(TypedDict):
    """Typed structure for transformed request data."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None


class ResponseData(TypedDict):
    """Typed structure for transformed response data."""

    status_code: int
    headers: dict[str, str]
    body: bytes


logger = structlog.get_logger(__name__)


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
        metrics: PrometheusMetrics | None = None,
    ) -> None:
        """Initialize the proxy service.

        Args:
            proxy_client: HTTP client for pure forwarding
            credentials_manager: Authentication manager
            proxy_mode: Transformation mode - "minimal" or "full"
            target_base_url: Base URL for the target API
            metrics: Prometheus metrics collector (optional)
        """
        self.proxy_client = proxy_client
        self.credentials_manager = credentials_manager
        self.proxy_mode = proxy_mode
        self.target_base_url = target_base_url.rstrip("/")
        self.metrics = metrics or get_metrics()

        # Create concrete transformers
        self.request_transformer = HTTPRequestTransformer()
        self.response_transformer = HTTPResponseTransformer()

        # Create OpenAI adapter for stream transformation
        from ccproxy.adapters.openai.adapter import OpenAIAdapter

        self.openai_adapter = OpenAIAdapter()

        # Cache environment-based configuration
        self._proxy_url = self._init_proxy_url()
        self._ssl_context = self._init_ssl_context()
        self._verbose_streaming = (
            os.environ.get("CCPROXY_VERBOSE_STREAMING", "false").lower() == "true"
        )

    def _init_proxy_url(self) -> str | None:
        """Initialize proxy URL from environment variables."""
        # Check for standard proxy environment variables
        # For HTTPS requests, prioritize HTTPS_PROXY
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        all_proxy = os.environ.get("ALL_PROXY")
        http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")

        proxy_url = https_proxy or all_proxy or http_proxy

        if proxy_url:
            logger.debug(f"Using proxy: {proxy_url}")

        return proxy_url

    def _init_ssl_context(self) -> str | bool:
        """Initialize SSL context configuration from environment variables."""
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

    async def handle_request(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes | None = None,
        query_params: dict[str, str | list[str]] | None = None,
        timeout: float = 240.0,
        request: Request | None = None,  # Optional FastAPI Request object
    ) -> tuple[int, dict[str, str], bytes] | StreamingResponse:
        """Handle a proxy request with full business logic orchestration.

        Args:
            method: HTTP method
            path: Request path (without /unclaude prefix)
            headers: Request headers
            body: Request body
            query_params: Query parameters
            timeout: Request timeout in seconds
            request: Optional FastAPI Request object for accessing request context

        Returns:
            Tuple of (status_code, headers, body) or StreamingResponse for streaming

        Raises:
            HTTPException: If request fails
        """
        # Extract request metadata
        model, streaming = self._extract_request_metadata(body)
        endpoint = path.split("/")[-1] if path else "unknown"

        # Use existing context from request if available, otherwise create new one
        if request and hasattr(request, "state") and hasattr(request.state, "context"):
            # Use existing context from middleware
            ctx = request.state.context
            # Add service-specific metadata
            ctx.add_metadata(
                endpoint=endpoint,
                model=model,
                streaming=streaming,
                service_type="proxy_service",
            )
            # Use a no-op context manager since we're using existing context
            from contextlib import nullcontext

            context_manager: Any = nullcontext(ctx)
        else:
            # Create new context for observability
            context_manager = request_context(
                method=method,
                path=path,
                endpoint=endpoint,
                model=model,
                streaming=streaming,
                service_type="proxy_service",
            )

        async with context_manager as ctx:
            # Record Prometheus metrics
            self.metrics.inc_active_requests()

            try:
                # 1. Authentication - get access token
                async with timed_operation("oauth_token", ctx.request_id):
                    logger.debug("Retrieving OAuth access token...")
                    access_token = await self._get_access_token()

                # 2. Request transformation
                async with timed_operation("request_transform", ctx.request_id):
                    logger.debug("Transforming request...")
                    transformed_request = await self._transform_request(
                        method, path, headers, body, query_params, access_token
                    )

                # 3. Forward request using proxy client
                logger.debug(f"Forwarding request to: {transformed_request['url']}")

                # Check if this will be a streaming response
                should_stream = streaming or self._should_stream_response(
                    transformed_request["headers"]
                )

                if should_stream:
                    logger.debug("Streaming response detected, using streaming handler")
                    return await self._handle_streaming_request(
                        transformed_request, path, timeout, ctx
                    )
                else:
                    logger.debug("Non-streaming response, using regular handler")

                # Handle regular request
                async with timed_operation("api_call", ctx.request_id) as api_op:
                    start_time = time.perf_counter()

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

                    end_time = time.perf_counter()
                    api_duration = end_time - start_time
                    api_op["duration_seconds"] = api_duration

                # 4. Response transformation
                async with timed_operation("response_transform", ctx.request_id):
                    logger.debug("Transforming response...")
                    # For error responses, skip transformation to preserve upstream error format
                    transformed_response: ResponseData
                    if status_code >= 400:
                        logger.info(
                            f"Preserving upstream error response: {status_code}",
                            status_code=status_code,
                            has_body=bool(response_body),
                            content_length=len(response_body) if response_body else 0,
                        )
                        transformed_response = ResponseData(
                            status_code=status_code,
                            headers=response_headers,
                            body=response_body,
                        )
                    else:
                        transformed_response = await self._transform_response(
                            status_code, response_headers, response_body, path
                        )

                # 5. Extract response metrics using direct JSON parsing
                tokens_input = tokens_output = cache_read_tokens = (
                    cache_write_tokens
                ) = cost_usd = None
                if transformed_response["body"]:
                    try:
                        response_data = json.loads(
                            transformed_response["body"].decode("utf-8")
                        )
                        usage = response_data.get("usage", {})
                        tokens_input = usage.get("input_tokens")
                        tokens_output = usage.get("output_tokens")
                        cache_read_tokens = usage.get("cache_read_input_tokens")
                        cache_write_tokens = usage.get("cache_creation_input_tokens")

                        # Calculate cost including cache tokens if we have tokens and model
                        from ccproxy.utils.cost_calculator import calculate_token_cost

                        cost_usd = calculate_token_cost(
                            tokens_input,
                            tokens_output,
                            model,
                            cache_read_tokens,
                            cache_write_tokens,
                        )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass  # Keep all values as None if parsing fails

                # 6. Update context with response data
                ctx.add_metadata(
                    status_code=status_code,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    cost_usd=cost_usd,
                )

                # 7. Record Prometheus metrics
                self.metrics.record_request(
                    method, endpoint, model, status_code, "proxy_service"
                )
                self.metrics.record_response_time(
                    ctx.duration_seconds, model, endpoint, "proxy_service"
                )

                if tokens_input:
                    self.metrics.record_tokens(
                        tokens_input, "input", model, "proxy_service"
                    )
                if tokens_output:
                    self.metrics.record_tokens(
                        tokens_output, "output", model, "proxy_service"
                    )
                if cache_read_tokens:
                    self.metrics.record_tokens(
                        cache_read_tokens, "cache_read", model, "proxy_service"
                    )
                if cache_write_tokens:
                    self.metrics.record_tokens(
                        cache_write_tokens, "cache_write", model, "proxy_service"
                    )
                if cost_usd:
                    self.metrics.record_cost(cost_usd, model, "total", "proxy_service")

                return (
                    transformed_response["status_code"],
                    transformed_response["headers"],
                    transformed_response["body"],
                )

            except Exception as e:
                # Record error metrics
                error_type = type(e).__name__
                self.metrics.record_error(error_type, endpoint, model, "proxy_service")

                logger.exception(f"Error in proxy request: {method} {path}")
                # Re-raise the exception without transformation
                # Let higher layers handle specific error types
                raise
            finally:
                self.metrics.dec_active_requests()

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

                    if (
                        validation.valid
                        and validation.expired
                        and validation.credentials
                    ):
                        logger.debug(
                            "Found credentials but access token is invalid/expired"
                        )
                        logger.debug(
                            f"Expired at: {validation.credentials.claude_ai_oauth.expires_at}"
                        )
                except Exception as e:
                    logger.debug(
                        f"Could not check credential details: {e}",
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )

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
        query_params: dict[str, str | list[str]] | None,
        access_token: str,
    ) -> RequestData:
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
    ) -> ResponseData:
        """Transform response using the transformer pipeline.

        Args:
            status_code: HTTP status code
            headers: Response headers
            body: Response body
            original_path: Original request path for context

        Returns:
            Transformed response data
        """
        # For error responses, pass through without transformation
        if status_code >= 400:
            return {
                "status_code": status_code,
                "headers": headers,
                "body": body,
            }

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
        should_stream = (
            "text/event-stream" in accept_header or "stream" in accept_header
        )
        logger.debug(
            f"Stream check - Accept header: {accept_header!r}, Should stream: {should_stream}"
        )
        return should_stream

    def _extract_request_metadata(self, body: bytes | None) -> tuple[str | None, bool]:
        """Extract model and streaming flag from request body.

        Args:
            body: Request body

        Returns:
            Tuple of (model, streaming)
        """
        if not body:
            return None, False

        try:
            body_data = json.loads(body.decode("utf-8"))
            model = body_data.get("model")
            streaming = body_data.get("stream", False)
            return model, streaming
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, False

    async def _handle_streaming_request(
        self,
        request_data: RequestData,
        original_path: str,
        timeout: float,
        ctx: "RequestContext",
    ) -> StreamingResponse:
        """Handle streaming request with transformation.

        Args:
            request_data: Transformed request data
            original_path: Original request path for context
            timeout: Request timeout

        Returns:
            StreamingResponse
        """
        # Store response headers to preserve for errors
        response_headers = {}
        response_status = 200

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            try:
                logger.debug(
                    f"Starting stream generator for request: {request_data['method']} {request_data['url']}"
                )
                logger.debug(f"Request headers: {request_data['headers']}")

                # Use httpx directly for streaming since we need the stream context manager
                # Get proxy and SSL settings from cached configuration
                proxy_url = self._proxy_url
                verify = self._ssl_context

                start_time = time.perf_counter()
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
                    end_time = time.perf_counter()
                    proxy_api_call_ms = (end_time - start_time) * 1000
                    logger.info(
                        f"Proxy streaming API call completed in {proxy_api_call_ms:.2f}ms"
                    )
                    logger.debug(f"Stream response status: {response.status_code}")
                    logger.debug(f"Stream response headers: {dict(response.headers)}")

                    # Store response status and headers
                    nonlocal response_status, response_headers
                    response_status = response.status_code
                    response_headers = dict(response.headers)

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
                    logger.debug(
                        f"Is OpenAI request: {is_openai} for path: {original_path}"
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
                            logger.debug(
                                f"Yielding transformed chunk: {len(transformed_chunk)} bytes"
                            )
                            yield transformed_chunk
                    else:
                        # Stream as-is for Anthropic endpoints
                        logger.debug("Streaming as-is for Anthropic endpoint")
                        chunk_count = 0
                        content_block_delta_count = 0

                        # Use cached verbose streaming configuration
                        verbose_streaming = self._verbose_streaming

                        async for chunk in response.aiter_bytes():
                            if chunk:
                                chunk_count += 1

                                # Compact logging for content_block_delta events
                                chunk_str = chunk.decode("utf-8", errors="replace")
                                if (
                                    "content_block_delta" in chunk_str
                                    and not verbose_streaming
                                ):
                                    content_block_delta_count += 1
                                    # Only log every 10th content_block_delta or when we start/end
                                    if content_block_delta_count == 1:
                                        logger.debug(
                                            "Streaming content_block_delta events (use CCPROXY_VERBOSE_STREAMING=true for all chunks)"
                                        )
                                    elif content_block_delta_count % 10 == 0:
                                        logger.debug(
                                            f"...{content_block_delta_count} content_block_delta events streamed..."
                                        )
                                elif (
                                    verbose_streaming
                                    or "content_block_delta" not in chunk_str
                                ):
                                    # Log non-content_block_delta events normally, or everything if verbose mode
                                    logger.debug(
                                        f"Yielding chunk {chunk_count}: {len(chunk)} bytes - {chunk[:100]!r}..."
                                    )

                                yield chunk

                        # Final summary for content_block_delta events
                        if content_block_delta_count > 0 and not verbose_streaming:
                            logger.debug(
                                f"Completed streaming {content_block_delta_count} content_block_delta events"
                            )

            except Exception as e:
                logger.exception("Error in streaming response")
                error_message = f'data: {{"error": "Streaming error: {str(e)}"}}\\n\\n'
                yield error_message.encode("utf-8")

        # Always use upstream headers as base
        final_headers = response_headers.copy()

        # Ensure critical headers for streaming
        final_headers["Cache-Control"] = "no-cache"
        final_headers["Connection"] = "keep-alive"
        final_headers["Access-Control-Allow-Origin"] = "*"
        final_headers["Access-Control-Allow-Headers"] = "*"

        # Set content-type if not already set by upstream
        if "content-type" not in final_headers:
            final_headers["content-type"] = "text/event-stream"

        return StreamingResponse(
            stream_generator(),
            status_code=response_status,
            headers=final_headers,
        )

    async def _transform_anthropic_to_openai_stream(
        self, response: httpx.Response, original_path: str
    ) -> AsyncGenerator[bytes, None]:
        """Transform Anthropic SSE stream to OpenAI SSE format using adapter.

        Args:
            response: Streaming response from Anthropic
            original_path: Original request path for context

        Yields:
            Transformed OpenAI SSE format chunks
        """

        # Parse SSE chunks from response into dict stream
        async def sse_to_dict_stream() -> AsyncGenerator[dict[str, object], None]:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE chunk: {data_str}")
                            continue

        # Transform using OpenAI adapter and format back to SSE
        async for openai_chunk in self.openai_adapter.adapt_stream(
            sse_to_dict_stream()
        ):
            sse_line = f"data: {json.dumps(openai_chunk)}\n\n"
            yield sse_line.encode("utf-8")

    async def close(self) -> None:
        """Close any resources held by the proxy service."""
        if self.proxy_client:
            await self.proxy_client.close()
        if self.credentials_manager:
            await self.credentials_manager.__aexit__(None, None, None)
