import contextlib
import json
import uuid
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from fastapi import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from ccproxy.auth.exceptions import OAuthTokenRefreshError
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.interfaces import (
    DetectionServiceProtocol,
    ProfiledTokenManagerProtocol,
)
from ccproxy.services.adapters.chain_composer import compose_from_chain
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.services.handler_config import HandlerConfig
from ccproxy.streaming import DeferredStreaming, StreamingBufferService
from ccproxy.utils.headers import (
    extract_request_headers,
    extract_response_headers,
    filter_request_headers,
    filter_response_headers,
)
from ccproxy.utils.model_mapper import restore_model_aliases


logger = get_plugin_logger()


class CodexAdapter(BaseHTTPAdapter):
    """Simplified Codex adapter."""

    def __init__(
        self,
        detection_service: DetectionServiceProtocol,
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.detection_service: DetectionServiceProtocol = detection_service
        self.token_manager: ProfiledTokenManagerProtocol = cast(
            ProfiledTokenManagerProtocol, self.auth_manager
        )
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
        self._ensure_tool_accumulator(ctx)
        endpoint = ctx.metadata.get("endpoint", "")
        body = await request.body()
        body = await self._map_request_model(ctx, body)
        headers = extract_request_headers(request)

        # Determine client streaming intent from body flag (fallback to False)
        wants_stream = False
        try:
            data = json.loads(body.decode()) if body else {}
            wants_stream = bool(data.get("stream", False))
        except Exception:  # Malformed/missing JSON -> assume non-streaming
            wants_stream = False
        logger.trace(
            "codex_adapter_request_intent",
            wants_stream=wants_stream,
            endpoint=endpoint,
            format_chain=getattr(ctx, "format_chain", []),
            category="streaming",
        )

        # Explicitly set service_type for downstream helpers
        with contextlib.suppress(Exception):
            ctx.metadata.setdefault("service_type", "codex")

        # If client wants streaming, delegate to streaming handler directly
        if wants_stream and self.streaming_handler:
            logger.trace(
                "codex_adapter_delegating_streaming",
                endpoint=endpoint,
                category="streaming",
            )
            return await self.handle_streaming(request, endpoint)

        # Otherwise, buffer the upstream streaming response into a standard one
        if getattr(self.config, "buffer_non_streaming", True):
            # 1) Prepare provider request (adds auth, sets stream=true, etc.)
            # Apply request format conversion if specified
            if ctx.format_chain and len(ctx.format_chain) > 1:
                try:
                    request_payload = self._decode_json_body(
                        body, context="codex_request"
                    )
                    request_payload = await self._apply_format_chain(
                        data=request_payload,
                        format_chain=ctx.format_chain,
                        stage="request",
                    )
                    body = self._encode_json_body(request_payload)
                except Exception as e:
                    logger.error(
                        "codex_format_chain_request_failed",
                        error=str(e),
                        exc_info=e,
                        category="transform",
                    )
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Failed to convert request using format chain",
                                "details": str(e),
                            }
                        },
                    )

            prepared_body, prepared_headers = await self.prepare_provider_request(
                body, headers, endpoint
            )
            logger.trace(
                "codex_adapter_prepared_provider_request",
                header_keys=list(prepared_headers.keys()),
                body_size=len(prepared_body or b""),
                category="http",
            )

            # 2) Build handler config using composed adapter from format_chain (unified path)

            composed_adapter = (
                compose_from_chain(
                    registry=self.format_registry, chain=ctx.format_chain
                )
                if self.format_registry and ctx.format_chain
                else None
            )

            handler_config = HandlerConfig(
                supports_streaming=True,
                request_transformer=None,
                response_adapter=composed_adapter,
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
            logger.trace(
                "codex_adapter_buffered_response_ready",
                status_code=buffered_response.status_code,
                buffer_respones_preview=buffered_response.body[:300],
                category="streaming",
                format_chain=getattr(ctx, "format_chain", []),
            )

            # 4) Apply reverse format chain on buffered body if needed
            if ctx.format_chain and len(ctx.format_chain) > 1:
                from typing import Literal

                mode: Literal["error", "response"] = (
                    "error" if buffered_response.status_code >= 400 else "response"
                )
                try:
                    body_bytes = (
                        buffered_response.body
                        if isinstance(buffered_response.body, bytes)
                        else bytes(buffered_response.body)
                    )
                    response_payload = self._decode_json_body(
                        body_bytes, context=f"codex_{mode}"
                    )
                    response_payload = await self._apply_format_chain(
                        data=response_payload,
                        format_chain=ctx.format_chain,
                        stage=mode,
                    )
                    metadata = getattr(ctx, "metadata", None)
                    alias_map = getattr(ctx, "_model_alias_map", None)
                    if isinstance(metadata, dict):
                        if (
                            isinstance(alias_map, dict)
                            and isinstance(response_payload, dict)
                            and isinstance(response_payload.get("model"), str)
                        ):
                            response_payload["model"] = alias_map.get(
                                response_payload["model"], response_payload["model"]
                            )
                        restore_model_aliases(response_payload, metadata)
                    converted_body = self._encode_json_body(response_payload)
                except Exception as e:
                    logger.error(
                        "codex_format_chain_response_failed",
                        error=str(e),
                        mode=mode,
                        exc_info=e,
                        category="transform",
                    )
                    return JSONResponse(
                        status_code=502,
                        content={
                            "error": {
                                "type": "server_error",
                                "message": "Failed to convert provider response using format chain",
                                "details": str(e),
                            }
                        },
                    )

                headers_out = filter_response_headers(dict(buffered_response.headers))
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
        return f"{self.base_url}/responses"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        token_value = await self._resolve_access_token()

        # Get profile to extract chatgpt_account_id
        profile = await self.token_manager.get_profile_quick()
        chatgpt_account_id = (
            getattr(profile, "chatgpt_account_id", None) if profile else None
        )

        # Parse body (format conversion is now handled by format chain)
        body_data = json.loads(body.decode()) if body else {}

        # Inject instructions mandatory for being allow to
        # to used the Codex API endpoint
        # Fetch detected instructions from detection service
        instructions = self._get_instructions()

        existing_instructions = body_data.get("instructions")
        if isinstance(existing_instructions, str) and existing_instructions:
            if instructions:
                instructions = instructions + "\n" + existing_instructions
            else:
                instructions = existing_instructions

        body_data["instructions"] = instructions

        # Codex backend requires stream=true, always override
        body_data["stream"] = True
        body_data["store"] = False

        # Remove unsupported keys for Codex
        for key in ("max_output_tokens", "max_completion_tokens", "temperature"):
            body_data.pop(key, None)

        list_input = body_data.get("input", [])
        # Remove any input types that Codex does not support
        body_data["input"] = [
            input for input in list_input if input.get("type") != "item_reference"
        ]

        #
        # Remove any prefixed metadata fields that shouldn't be sent to the API
        body_data = self._remove_metadata_fields(body_data)

        # Filter and add headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)

        session_id = filtered_headers.get("session_id") or str(uuid.uuid4())
        conversation_id = filtered_headers.get("conversation_id") or str(uuid.uuid4())

        base_headers = {
            "authorization": f"Bearer {token_value}",
            "content-type": "application/json",
            "session_id": session_id,
            "conversation_id": conversation_id,
        }

        if chatgpt_account_id is not None:
            base_headers["chatgpt-account-id"] = chatgpt_account_id

        filtered_headers.update(base_headers)

        cli_headers = self._collect_cli_headers()
        if cli_headers:
            filtered_headers.update(cli_headers)

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

    async def _resolve_access_token(self) -> str:
        """Resolve an access token suitable for Codex requests.

        If the auth manager/credential balancer is not configured, raise a
        unified AuthenticationError so middleware can return a clean 401
        without leaking stack traces.
        """

        # Guard: token manager must be configured via plugin auth_manager
        if not getattr(self, "token_manager", None):
            from ccproxy.core.errors import AuthenticationError

            logger.warning(
                "auth_manager_override_not_resolved",
                plugin="codex",
                auth_manager_name="codex_credential_balancer",
                category="auth",
            )
            raise AuthenticationError(
                "Authentication manager not configured for Codex provider"
            )

        token_manager = self.token_manager

        async def _snapshot_token() -> str | None:
            snapshot = await token_manager.get_token_snapshot()
            if snapshot and snapshot.access_token:
                return snapshot.access_token
            return None

        credentials = await token_manager.load_credentials()
        if credentials and token_manager.should_refresh(credentials):
            try:
                refreshed = await token_manager.get_access_token_with_refresh()
                if refreshed:
                    return refreshed
            except OAuthTokenRefreshError as exc:
                logger.warning(
                    "codex_token_refresh_failed",
                    error=str(exc),
                    category="auth",
                )
                fallback = await _snapshot_token()
                if fallback:
                    return fallback

        token = None
        try:
            token = await token_manager.get_access_token()
        except OAuthTokenRefreshError as exc:
            logger.warning(
                "codex_token_refresh_failed",
                error=str(exc),
                category="auth",
            )
            fallback = await _snapshot_token()
            if fallback:
                return fallback

        if token:
            return token

        try:
            refreshed = await token_manager.get_access_token_with_refresh()
            if refreshed:
                return refreshed
        except OAuthTokenRefreshError as exc:
            logger.warning(
                "codex_token_refresh_failed",
                error=str(exc),
                category="auth",
            )
            fallback = await _snapshot_token()
            if fallback:
                return fallback

        fallback = await _snapshot_token()
        if fallback:
            return fallback

        raise ValueError("No authentication credentials available")

    def _collect_cli_headers(self) -> dict[str, str]:
        """Collect safe CLI headers from detection cache for forwarding."""

        if not self.detection_service:
            return {}

        headers_data = self.detection_service.get_detected_headers()
        if not headers_data:
            return {}

        ignores = {
            header.lower() for header in self.detection_service.get_ignored_headers()
        }
        redacted = {
            header.lower() for header in self.detection_service.get_redacted_headers()
        }

        return headers_data.filtered(ignores=ignores, redacted=redacted)

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle streaming with request conversion for Codex.

        Applies request format conversion (e.g., anthropic.messages -> openai.responses) before
        preparing the provider request, then delegates to StreamingHandler with
        a streaming response adapter for reverse conversion as needed.
        """
        if not self.streaming_handler:
            # Fallback to base behavior
            return await super().handle_streaming(request, endpoint, **kwargs)

        # Get context
        ctx = request.state.context
        self._ensure_tool_accumulator(ctx)

        # Extract body and headers
        body = await request.body()
        body = await self._map_request_model(ctx, body)
        headers = extract_request_headers(request)

        # Ensure format adapters are available when required
        self._ensure_format_registry(ctx.format_chain, endpoint)

        # Apply request format conversion if a chain is defined
        if ctx.format_chain and len(ctx.format_chain) > 1:
            try:
                request_payload = self._decode_json_body(
                    body, context="codex_stream_request"
                )
                request_payload = await self._apply_format_chain(
                    data=request_payload,
                    format_chain=ctx.format_chain,
                    stage="request",
                )
                self._record_tool_definitions(ctx, request_payload)
                body = self._encode_json_body(request_payload)
            except Exception as e:
                logger.error(
                    "codex_format_chain_request_failed",
                    error=str(e),
                    exc_info=e,
                    category="transform",
                )
                # Convert error to streaming response

                error_content = {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Failed to convert request using format chain",
                        "details": str(e),
                    }
                }
                error_bytes = json.dumps(error_content).encode("utf-8")

                async def error_generator() -> (
                    Any
                ):  # AsyncGenerator[bytes, None] would be more specific
                    yield error_bytes

                return StreamingResponse(
                    content=error_generator(),
                    status_code=400,
                    media_type="application/json",
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

        handler_config = HandlerConfig(
            supports_streaming=True,
            request_transformer=None,
            response_adapter=streaming_format_adapter,
            format_context=None,
        )

        target_url = await self.get_target_url(endpoint)

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
        if not self.detection_service:
            return ""

        prompts = self.detection_service.get_detected_prompts()
        if prompts.has_instructions():
            return prompts.instructions or ""

        injection = self.detection_service.get_system_prompt()
        if isinstance(injection, dict):
            instructions = injection.get("instructions")
            if isinstance(instructions, str):
                return instructions

        fallback = getattr(self.detection_service, "instructions_value", None)
        if isinstance(fallback, str):
            return fallback

        return ""

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
