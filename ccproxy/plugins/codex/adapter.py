import contextlib
import json
import uuid
from typing import Any

import httpx
from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.streaming import DeferredStreaming, StreamingBufferService
from ccproxy.utils.headers import (
    extract_request_headers,
    extract_response_headers,
    filter_request_headers,
    filter_response_headers,
)

from .detection_service import CodexDetectionService


logger = get_plugin_logger()


class CodexAdapter(BaseHTTPAdapter):
    """Simplified Codex adapter."""

    def __init__(
        self,
        detection_service: CodexDetectionService,
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.detection_service = detection_service
        self.base_url = self.config.base_url.rstrip("/")

    async def handle_request(
        self, request: Request
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle request with Codex-specific streaming behavior.

        Codex upstream only supports streaming. If the client requests a non-streaming
        response, we internally stream and buffer it, then return a standard Response.
        """
        # Context + request info
        ctx = request.state.context
        endpoint = ctx.metadata.get("endpoint", "")
        body = await request.body()
        headers = extract_request_headers(request)

        # Determine client streaming intent from body flag (fallback to False)
        wants_stream = False
        try:
            import json as _json

            data = _json.loads(body.decode()) if body else {}
            wants_stream = bool(data.get("stream", False))
        except Exception:  # Malformed/missing JSON -> assume non-streaming
            wants_stream = False

        # Explicitly set service_type for downstream helpers
        with contextlib.suppress(Exception):
            ctx.metadata.setdefault("service_type", "codex")

        # If client wants streaming, delegate to streaming handler directly
        if wants_stream and self.streaming_handler:
            return await self.handle_streaming(request, endpoint)

        # Otherwise, buffer the upstream streaming response into a standard one
        if getattr(self.config, "buffer_non_streaming", True):
            # 1) Prepare provider request (adds auth, sets stream=true, etc.)
            # Apply request format conversion if specified
            if ctx.format_chain and len(ctx.format_chain) > 1:
                with contextlib.suppress(Exception):
                    body = await self._execute_format_chain(
                        body, ctx.format_chain, ctx, mode="request"
                    )

            prepared_body, prepared_headers = await self.prepare_provider_request(
                body, headers, endpoint
            )

            # 2) Build handler config (optionally with streaming format adapter)
            streaming_format_adapter = None
            if ctx.format_chain and len(ctx.format_chain) > 1 and self.format_registry:
                from_format = ctx.format_chain[-1]
                to_format = ctx.format_chain[0]
                try:
                    streaming_format_adapter = self.format_registry.get_if_exists(
                        from_format, to_format
                    )
                except Exception:
                    streaming_format_adapter = None

            from ccproxy.services.handler_config import HandlerConfig

            handler_config = HandlerConfig(
                supports_streaming=True,
                request_transformer=None,
                response_adapter=streaming_format_adapter,
                format_context=None,
            )

            # 3) Use StreamingBufferService to convert upstream stream -> regular response
            target_url = await self.get_target_url(endpoint)
            # Try to use a client with base_url for better hook integration
            http_client = await self.http_pool_manager.get_client()
            hook_manager = (
                getattr(self.streaming_handler, "hook_manager", None)
                if self.streaming_handler
                else None
            )
            buffer_service = StreamingBufferService(
                http_client=http_client,
                request_tracer=None,
                hook_manager=hook_manager,
                http_pool_manager=self.http_pool_manager,
            )

            buffered_response = await buffer_service.handle_buffered_streaming_request(
                method=request.method,
                url=target_url,
                headers=prepared_headers,
                body=prepared_body,
                handler_config=handler_config,
                request_context=ctx,
                provider_name="codex",
            )

            # 4) Apply reverse format chain on buffered body if needed
            if ctx.format_chain and len(ctx.format_chain) > 1:
                mode = "error" if buffered_response.status_code >= 400 else "response"
                converted_body = await self._execute_format_chain(
                    buffered_response.body, ctx.format_chain, ctx, mode=mode
                )

                # Filter headers and rebuild response without content-length
                headers_out = filter_response_headers(dict(buffered_response.headers))
                if "content-length" in headers_out:
                    del headers_out["content-length"]
                return Response(
                    content=converted_body,
                    status_code=buffered_response.status_code,
                    headers=headers_out,
                    media_type="application/json",
                )

            # No conversion needed; return buffered response as-is
            return buffered_response

        # Fallback: no buffering requested, use base non-streaming flow
        return await super().handle_request(request)

    async def get_target_url(self, endpoint: str) -> str:
        # Old URL: https://chat.openai.com/backend-anon/responses (308 redirect)
        return f"{self.base_url}/responses"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        # Get auth credentials and profile
        auth_data = await self.auth_manager.load_credentials()
        if not auth_data:
            raise ValueError("No authentication credentials available")

        # Get profile to extract chatgpt_account_id
        profile = await self.auth_manager.get_profile_quick()
        chatgpt_account_id = profile.chatgpt_account_id if profile else None

        # Parse body (format conversion is now handled by format chain)
        body_data = json.loads(body.decode()) if body else {}

        # Inject instructions if not present
        if "instructions" not in body_data or body_data.get("instructions") is None:
            body_data["instructions"] = self._get_instructions()

        # Codex backend requires stream=true, always override
        body_data["stream"] = True
        body_data["store"] = False

        # Remove any prefixed metadata fields that shouldn't be sent to the API
        body_data = self._remove_metadata_fields(body_data)

        # Filter and add headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        base_headers = {
            "authorization": f"Bearer {auth_data.access_token}",
            "session_id": str(uuid.uuid4()),
            "content-type": "application/json",
        }
        # Add chatgpt-account-id only if available
        if chatgpt_account_id is not None:
            base_headers["chatgpt-account-id"] = chatgpt_account_id

        filtered_headers.update(base_headers)

        # Add CLI headers
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.headers:
                cli_headers = cached_data.headers.to_headers_dict()
                for key, value in cli_headers.items():
                    filtered_headers[key.lower()] = value

        return json.dumps(body_data).encode(), filtered_headers

    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response | StreamingResponse:
        """Return a plain Response; streaming handled upstream by BaseHTTPAdapter.

        The BaseHTTPAdapter is responsible for detecting streaming and delegating
        to the shared StreamingHandler. For non-streaming responses, adapters
        should return a simple Starlette Response.
        """
        response_headers = extract_response_headers(response)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
        )

    async def _create_streaming_response(
        self, response: httpx.Response, endpoint: str
    ) -> DeferredStreaming:
        """Create streaming response with format conversion support."""
        # Deprecated: streaming is centrally handled by BaseHTTPAdapter/StreamingHandler
        # Kept for compatibility; not used.
        raise NotImplementedError

    def _needs_format_conversion(self, endpoint: str) -> bool:
        """Deprecated: format conversion handled via format chain in BaseHTTPAdapter."""
        return False

    def _get_response_format_conversion(self, endpoint: str) -> tuple[str, str]:
        """Deprecated: conversion direction decided by format chain upstream."""
        return ("response_api", "openai")

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle streaming with request conversion for Codex.

        Applies request format conversion (e.g., anthropic->response_api) before
        preparing the provider request, then delegates to StreamingHandler with
        a streaming response adapter for reverse conversion as needed.
        """
        if not self.streaming_handler:
            # Fallback to base behavior
            return await super().handle_streaming(request, endpoint, **kwargs)

        # Get context
        ctx = request.state.context

        # Extract body and headers
        body = await request.body()
        headers = extract_request_headers(request)

        # Apply request format conversion if a chain is defined
        if ctx.format_chain and len(ctx.format_chain) > 1:
            with contextlib.suppress(Exception):
                body = await self._execute_format_chain(
                    body, ctx.format_chain, ctx, mode="request"
                )

        # Provider-specific preparation (adds auth, sets stream=true)
        prepared_body, prepared_headers = await self.prepare_provider_request(
            body, headers, endpoint
        )

        # Get format adapter for streaming reverse conversion
        streaming_format_adapter = None
        if ctx.format_chain and len(ctx.format_chain) > 1 and self.format_registry:
            from_format = ctx.format_chain[-1]
            to_format = ctx.format_chain[0]
            try:
                streaming_format_adapter = self.format_registry.get_if_exists(
                    from_format, to_format
                )
            except Exception:
                streaming_format_adapter = None

        from ccproxy.services.handler_config import HandlerConfig

        handler_config = HandlerConfig(
            supports_streaming=True,
            request_transformer=None,
            response_adapter=streaming_format_adapter,
            format_context=None,
        )

        target_url = await self.get_target_url(endpoint)

        from urllib.parse import urlparse

        parsed_url = urlparse(target_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        return await self.streaming_handler.handle_streaming_request(
            method=request.method,
            url=target_url,
            headers=prepared_headers,
            body=prepared_body,
            handler_config=handler_config,
            request_context=ctx,
            client=await self.http_pool_manager.get_client(base_url=base_url),
        )

    # Helper methods
    def _remove_metadata_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove fields that start with '_' as they are internal metadata.

        Args:
            data: Dictionary that may contain metadata fields

        Returns:
            Cleaned dictionary without metadata fields
        """
        if not isinstance(data, dict):
            return data

        # Create a new dict without keys starting with '_'
        cleaned_data: dict[str, Any] = {}
        for key, value in data.items():
            if not key.startswith("_"):
                # Recursively clean nested dictionaries
                if isinstance(value, dict):
                    cleaned_data[key] = self._remove_metadata_fields(value)
                elif isinstance(value, list):
                    # Clean list items if they are dictionaries
                    cleaned_items: list[Any] = []
                    for item in value:
                        if isinstance(item, dict):
                            cleaned_items.append(self._remove_metadata_fields(item))
                        else:
                            cleaned_items.append(item)
                    cleaned_data[key] = cleaned_items
                else:
                    cleaned_data[key] = value

        return cleaned_data

    def _get_instructions(self) -> str:
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.instructions:
                return cached_data.instructions.instructions_field
        return "You are a coding agent..."

    def adapt_error(self, error_body: dict[str, Any]) -> dict[str, Any]:
        """Convert Codex error format to appropriate API error format.

        Args:
            error_body: Codex error response

        Returns:
            API-formatted error response
        """
        # Handle the specific "Stream must be set to true" error
        if isinstance(error_body, dict) and "detail" in error_body:
            detail = error_body["detail"]
            if "Stream must be set to true" in detail:
                # Convert to generic invalid request error
                return {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Invalid streaming parameter",
                    }
                }

        # Handle other error formats that might have "error" key
        if "error" in error_body:
            return error_body

        # Default: wrap non-standard errors
        return {
            "error": {
                "type": "internal_server_error",
                "message": "An error occurred processing the request",
            }
        }
