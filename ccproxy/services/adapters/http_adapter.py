import contextlib
import json
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.models.provider import ProviderConfig
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.adapters.chain_composer import compose_from_chain
from ccproxy.services.handler_config import HandlerConfig
from ccproxy.streaming import DeferredStreaming
from ccproxy.streaming.handler import StreamingHandler
from ccproxy.utils.headers import extract_request_headers, filter_response_headers
from ccproxy.utils.model_mapper import (
    ModelMapper,
    add_model_alias,
    restore_model_aliases,
)


logger = get_plugin_logger()


class BaseHTTPAdapter(BaseAdapter):
    """Simplified HTTP adapter with format chain support."""

    def __init__(
        self,
        config: ProviderConfig,
        auth_manager: Any,
        http_pool_manager: Any,
        streaming_handler: StreamingHandler | None = None,
        **kwargs: Any,
    ) -> None:
        # Call parent constructor to properly initialize config
        super().__init__(config=config, **kwargs)
        self.auth_manager = auth_manager
        self.http_pool_manager = http_pool_manager
        self.streaming_handler = streaming_handler
        self.format_registry = kwargs.get("format_registry")
        self.context = kwargs.get("context")
        self.model_mapper = kwargs.get("model_mapper")

        logger.debug(
            "base_http_adapter_initialized",
            has_streaming_handler=streaming_handler is not None,
            has_format_registry=self.format_registry is not None,
        )

    async def handle_request(
        self, request: Request
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle request with streaming detection and format chain support."""

        # Get context from middleware (already initialized)
        ctx = request.state.context
        self._ensure_tool_accumulator(ctx)

        # Step 1: Extract request data
        body = await request.body()
        body = await self._map_request_model(ctx, body)
        headers = extract_request_headers(request)
        method = request.method
        endpoint = ctx.metadata.get("endpoint", "")

        # Fail fast if a format chain is configured without a registry
        self._ensure_format_registry(ctx.format_chain, endpoint)

        # Extra debug breadcrumbs to confirm code path and detection inputs
        logger.debug(
            "http_adapter_handle_request_entry",
            endpoint=endpoint,
            method=method,
            content_type=headers.get("content-type"),
            has_streaming_handler=bool(self.streaming_handler),
            category="stream_detection",
        )

        # Step 2: Early streaming detection
        if self.streaming_handler:
            logger.debug(
                "checking_should_stream",
                endpoint=endpoint,
                has_streaming_handler=True,
                content_type=headers.get("content-type"),
                category="stream_detection",
            )
            # Detect streaming via Accept header and/or body flag stream:true
            body_wants_stream = False
            parsed_payload: dict[str, Any] | None = None
            try:
                parsed_payload = json.loads(body.decode()) if body else {}
                body_wants_stream = bool(parsed_payload.get("stream", False))
            except Exception:
                body_wants_stream = False
            header_wants_stream = self.streaming_handler.should_stream_response(headers)
            logger.debug(
                "should_stream_results",
                body_wants_stream=body_wants_stream,
                header_wants_stream=header_wants_stream,
                endpoint=endpoint,
                category="stream_detection",
            )
            if body_wants_stream or header_wants_stream:
                logger.debug(
                    "streaming_request_detected",
                    endpoint=endpoint,
                    detected_via=(
                        "content_type_sse"
                        if header_wants_stream
                        else "body_stream_flag"
                    ),
                    category="stream_detection",
                )
                if isinstance(parsed_payload, dict):
                    self._record_tool_definitions(ctx, parsed_payload)
                return await self.handle_streaming(request, endpoint)
            else:
                logger.debug(
                    "not_streaming_request",
                    endpoint=endpoint,
                    category="stream_detection",
                )

        # Step 3: Execute format chain if specified (non-streaming)
        request_payload: dict[str, Any] | None = None
        if ctx.format_chain and len(ctx.format_chain) > 1:
            try:
                request_payload = self._decode_json_body(body, context="request")
            except ValueError as exc:
                logger.error(
                    "format_chain_request_parse_failed",
                    error=str(exc),
                    endpoint=endpoint,
                    category="transform",
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Failed to parse request body for format conversion",
                            "details": str(exc),
                        }
                    },
                )

            self._record_tool_definitions(ctx, request_payload)

            try:
                logger.debug(
                    "format_chain_request_about_to_convert",
                    chain=ctx.format_chain,
                    endpoint=endpoint,
                    category="transform",
                )
                request_payload = await self._apply_format_chain(
                    data=request_payload,
                    format_chain=ctx.format_chain,
                    stage="request",
                )
                body = self._encode_json_body(request_payload)
                logger.trace(
                    "format_chain_request_converted",
                    from_format=ctx.format_chain[0],
                    to_format=ctx.format_chain[-1],
                    keys=list(request_payload.keys()),
                    size_bytes=len(body),
                    category="transform",
                )
                logger.info(
                    "format_chain_applied",
                    stage="request",
                    endpoint=endpoint,
                    chain=ctx.format_chain,
                    steps=len(ctx.format_chain) - 1,
                    category="format",
                )
            except Exception as e:
                logger.error(
                    "format_chain_request_failed",
                    error=str(e),
                    endpoint=endpoint,
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
        # Step 4: Provider-specific preparation
        prepared_body, prepared_headers = await self.prepare_provider_request(
            body, headers, endpoint
        )
        with contextlib.suppress(Exception):
            logger.trace(
                "provider_request_prepared",
                endpoint=endpoint,
                header_keys=list(prepared_headers.keys()),
                body_size=len(prepared_body or b""),
                category="http",
            )

        # Step 5: Execute HTTP request
        target_url = await self.get_target_url(endpoint)
        (
            method,
            target_url,
            prepared_body,
            prepared_headers,
        ) = await self._emit_provider_request_prepared(
            request_obj=request,
            ctx=ctx,
            method=method,
            endpoint=endpoint,
            target_url=target_url,
            prepared_body=prepared_body,
            prepared_headers=prepared_headers,
            is_streaming=False,
        )
        provider_response = await self._execute_http_request(
            method,
            target_url,
            prepared_headers,
            prepared_body,
        )
        logger.trace(
            "provider_response_received",
            status_code=getattr(provider_response, "status_code", None),
            content_type=getattr(provider_response, "headers", {}).get(
                "content-type", None
            ),
            category="http",
        )

        # Step 6: Provider-specific response processing
        response = await self.process_provider_response(provider_response, endpoint)

        # filter out hop-by-hop headers
        headers = filter_response_headers(dict(provider_response.headers))

        # Step 7: Format the response
        if isinstance(response, StreamingResponse):
            logger.debug("process_provider_response_streaming")
            return await self._convert_streaming_response(
                response, ctx.format_chain, ctx
            )
        elif isinstance(response, Response):
            logger.debug("process_provider_response")
            response = self._restore_model_response(response, ctx)

            # httpx has already decoded provider payloads, so strip encoding
            # headers that no longer match the body we forward to clients.
            for header in ("content-encoding", "transfer-encoding", "content-length"):
                with contextlib.suppress(KeyError):
                    del response.headers[header]
            if ctx.format_chain and len(ctx.format_chain) > 1:
                stage: Literal["response", "error"] = (
                    "error" if provider_response.status_code >= 400 else "response"
                )
                try:
                    payload = self._decode_json_body(
                        cast(bytes, response.body), context=stage
                    )
                except ValueError as exc:
                    logger.error(
                        "format_chain_response_parse_failed",
                        error=str(exc),
                        endpoint=endpoint,
                        stage=stage,
                        category="transform",
                    )
                    return response

                try:
                    payload = await self._apply_format_chain(
                        data=payload,
                        format_chain=ctx.format_chain,
                        stage=stage,
                    )
                    metadata = getattr(ctx, "metadata", None)
                    if isinstance(metadata, dict):
                        alias_map = metadata.get("_model_alias_map")
                    else:
                        alias_map = None
                    if not alias_map:
                        alias_map = getattr(ctx, "_model_alias_map", None)
                    if isinstance(metadata, dict):
                        if (
                            isinstance(payload, dict)
                            and isinstance(alias_map, Mapping)
                            and isinstance(payload.get("model"), str)
                        ):
                            payload["model"] = alias_map.get(
                                payload["model"], payload["model"]
                            )
                        restore_model_aliases(payload, metadata)
                    body_bytes = self._encode_json_body(payload)
                    logger.info(
                        "format_chain_applied",
                        stage=stage,
                        endpoint=endpoint,
                        chain=ctx.format_chain,
                        steps=len(ctx.format_chain) - 1,
                        category="format",
                    )
                    restored = Response(
                        content=body_bytes,
                        status_code=provider_response.status_code,
                        headers=headers,
                        media_type=provider_response.headers.get(
                            "content-type", "application/json"
                        ),
                    )
                    return self._restore_model_response(restored, ctx)
                except Exception as e:
                    logger.error(
                        "format_chain_response_failed",
                        error=str(e),
                        endpoint=endpoint,
                        stage=stage,
                        exc_info=e,
                        category="transform",
                    )
                    # Return proper error instead of potentially malformed response
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": {
                                "type": "internal_server_error",
                                "message": "Failed to convert response format",
                                "details": str(e),
                            }
                        },
                    )
            else:
                logger.debug("format_chain_skipped", reason="no forward chain")
                return self._restore_model_response(response, ctx)
        else:
            logger.warning(
                "unexpected_provider_response_type", type=type(response).__name__
            )
        restored = Response(
            content=provider_response.content,
            status_code=provider_response.status_code,
            headers=headers,
            media_type=headers.get("content-type", "application/json"),
        )
        return self._restore_model_response(restored, ctx)
        # raise ValueError(
        #     "process_provider_response must return httpx.Response for non-streaming",
        # )

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle a streaming request using StreamingHandler with format chain support."""

        logger.debug("handle_streaming_called", endpoint=endpoint)

        if not self.streaming_handler:
            logger.error(
                "streaming_handler_missing",
                endpoint=endpoint,
                category="streaming",
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "type": "configuration_error",
                        "message": "Streaming handler is not configured for this provider.",
                        "details": {
                            "endpoint": endpoint,
                        },
                    }
                },
            )

        # Get context from middleware
        ctx = request.state.context
        method = request.method
        self._ensure_tool_accumulator(ctx)

        # Extract request data
        body = await request.body()
        body = await self._map_request_model(ctx, body)
        headers = extract_request_headers(request)

        # Fail fast on missing format registry if chain configured
        self._ensure_format_registry(ctx.format_chain, endpoint)

        # Step 1: Execute request-side format chain if specified (streaming)
        if ctx.format_chain and len(ctx.format_chain) > 1:
            try:
                stream_payload = self._decode_json_body(body, context="stream_request")
                stream_payload = await self._apply_format_chain(
                    data=stream_payload,
                    format_chain=ctx.format_chain,
                    stage="request",
                )
                self._record_tool_definitions(ctx, stream_payload)
                body = self._encode_json_body(stream_payload)
                logger.trace(
                    "format_chain_stream_request_converted",
                    from_format=ctx.format_chain[0],
                    to_format=ctx.format_chain[-1],
                    keys=list(stream_payload.keys()),
                    size_bytes=len(body),
                    category="transform",
                )
                logger.info(
                    "format_chain_applied",
                    stage="stream_request",
                    endpoint=endpoint,
                    chain=ctx.format_chain,
                    steps=len(ctx.format_chain) - 1,
                    category="format",
                )
            except Exception as e:
                logger.error(
                    "format_chain_stream_request_failed",
                    error=str(e),
                    endpoint=endpoint,
                    exc_info=e,
                    category="transform",
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Failed to convert streaming request using format chain",
                            "details": str(e),
                        }
                    },
                )

        # Step 2: Provider-specific preparation (add auth headers, etc.)
        prepared_body, prepared_headers = await self.prepare_provider_request(
            body, headers, endpoint
        )
        try:
            original_payload = json.loads(body.decode()) if body else {}
            if isinstance(original_payload, dict):
                self._record_tool_definitions(ctx, original_payload)
        except Exception:
            pass

        # Get format adapter for streaming if format chain exists
        # Important: Do NOT reverse the chain. Adapters are defined for the
        # declared flow and handle response/streaming internally.
        streaming_format_adapter = None
        if ctx.format_chain and self.format_registry:
            # For streaming responses, we need to reverse the format chain direction
            # Request: client_format → provider_format
            # Stream Response: provider_format → client_format
            from_format = ctx.format_chain[-1]  # provider format (e.g., "anthropic")
            to_format = ctx.format_chain[
                0
            ]  # client format (e.g., "openai.chat_completions")
            streaming_format_adapter = self.format_registry.get_if_exists(
                from_format, to_format
            )

            logger.debug(
                "streaming_adapter_lookup",
                format_chain=ctx.format_chain,
                from_format=from_format,
                to_format=to_format,
                adapter_found=streaming_format_adapter is not None,
                adapter_type=type(streaming_format_adapter).__name__
                if streaming_format_adapter
                else None,
            )

        # Build handler config for streaming with a composed format adapter derived from chain
        # Import here to avoid circular imports
        composed_adapter = (
            compose_from_chain(registry=self.format_registry, chain=ctx.format_chain)
            if self.format_registry and ctx.format_chain
            else streaming_format_adapter
        )

        if ctx.format_chain and len(ctx.format_chain) > 1 and composed_adapter is None:
            logger.error(
                "streaming_adapter_missing",
                endpoint=endpoint,
                chain=ctx.format_chain,
                category="format",
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "type": "configuration_error",
                        "message": "No streaming format adapter available for configured format chain.",
                        "details": {
                            "endpoint": endpoint,
                            "format_chain": ctx.format_chain,
                        },
                    }
                },
            )

        if composed_adapter is not None and ctx.format_chain:
            logger.debug(
                "streaming_format_adapter_selected",
                endpoint=endpoint,
                chain=ctx.format_chain,
                adapter_type=type(composed_adapter).__name__,
                category="format",
            )

        handler_config = HandlerConfig(
            supports_streaming=True,
            request_transformer=None,
            response_adapter=composed_adapter,  # use composed adapter when available
            format_context=None,
        )

        # Get target URL for proper client pool management
        target_url = await self.get_target_url(endpoint)

        (
            method,
            target_url,
            prepared_body,
            prepared_headers,
        ) = await self._emit_provider_request_prepared(
            request_obj=request,
            ctx=ctx,
            method=method,
            endpoint=endpoint,
            target_url=target_url,
            prepared_body=prepared_body,
            prepared_headers=prepared_headers,
            is_streaming=True,
        )

        # Get HTTP client from pool manager with base URL for hook integration
        parsed_url = urlparse(target_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Delegate to StreamingHandler - no format chain needed since adapter is in config
        return await self.streaming_handler.handle_streaming_request(
            method=method,
            url=target_url,
            headers=prepared_headers,  # Use prepared headers with auth
            body=prepared_body,  # Use prepared body
            handler_config=handler_config,
            request_context=ctx,
            client=await self.http_pool_manager.get_client(base_url=base_url),
        )

    async def _convert_streaming_response(
        self, response: StreamingResponse, format_chain: list[str], ctx: Any
    ) -> StreamingResponse:
        """Convert streaming response through reverse format chain."""
        # Streaming responses are already converted inside DeferredStreaming
        # via the configured format adapters; no additional work required here.
        logger.debug(
            "reverse_streaming_format_chain_disabled",
            reason="complex_sse_parsing_disabled",
            format_chain=format_chain,
        )
        return response

    async def _map_request_model(self, ctx: Any, body: bytes) -> bytes:
        """Apply provider model mapping to request payload if configured."""

        mapper = getattr(self, "model_mapper", None)
        if mapper is None and hasattr(self, "config"):
            config_rules = getattr(self.config, "model_mappings", None)
            if config_rules:
                mapper = ModelMapper(config_rules)
                self.model_mapper = mapper
        if mapper is None or not getattr(mapper, "has_rules", False) or not body:
            if body:
                model_value = None
                try:
                    parsed = json.loads(body.decode())
                    if isinstance(parsed, dict):
                        model_value = parsed.get("model")
                except Exception:
                    model_value = None
                logger.debug(
                    "model_mapper_missing",
                    has_mapper=bool(mapper),
                    has_rules=getattr(mapper, "has_rules", False),
                    request_id=getattr(ctx, "request_id", None),
                    client_model=model_value,
                )
            return body

        try:
            payload = json.loads(body.decode())
        except Exception:
            return body

        if not isinstance(payload, dict):
            return body

        model_value = payload.get("model")
        if not isinstance(model_value, str):
            return body

        match = mapper.map(model_value)
        if match.mapped == match.original:
            return body

        metadata = getattr(ctx, "metadata", None)
        if metadata is None or not isinstance(metadata, dict):
            metadata = {}
            ctx.metadata = metadata
            logger.debug(
                "model_mapping_metadata_initialized",
                context_type=type(ctx).__name__,
            )

        add_model_alias(metadata, original=match.original, mapped=match.mapped)
        alias_map_ctx = getattr(ctx, "_model_alias_map", None)
        if not isinstance(alias_map_ctx, dict):
            alias_map_ctx = {}
            ctx._model_alias_map = alias_map_ctx
        alias_map_ctx[match.mapped] = match.original
        metadata["_last_client_model"] = match.original
        metadata["_last_provider_model"] = match.mapped
        payload["model"] = match.mapped

        logger.debug(
            "model_mapping_applied",
            original_model=match.original,
            mapped_model=match.mapped,
            alias_map=alias_map_ctx,
            category="model_mapping",
        )

        return self._encode_json_body(payload)

    async def _emit_provider_request_prepared(
        self,
        *,
        request_obj: Request | None,
        ctx: Any,
        method: str,
        endpoint: str,
        target_url: str,
        prepared_body: bytes,
        prepared_headers: dict[str, str],
        is_streaming: bool,
    ) -> tuple[str, str, bytes, dict[str, str]]:
        """Emit hook before provider request is dispatched, allowing mutation."""

        hook_manager = getattr(self.http_pool_manager, "hook_manager", None)
        if not hook_manager:
            return method, target_url, prepared_body, prepared_headers

        provider_name = getattr(self.config, "name", None)
        body_for_hooks, body_kind = self._prepare_body_for_hook(prepared_body)
        hook_data: dict[str, Any] = {
            "method": method,
            "url": target_url,
            "headers": dict(prepared_headers),
            "body": body_for_hooks,
            "body_raw": None,
            "original_body_raw": prepared_body,
            "body_kind": body_kind,
            "is_streaming": is_streaming,
            "endpoint": endpoint,
        }

        hook_metadata: dict[str, Any] = {}
        request_id = getattr(ctx, "request_id", None)
        if request_id:
            hook_metadata["request_id"] = request_id
        if endpoint:
            hook_metadata["endpoint"] = endpoint

        ctx_metadata = getattr(ctx, "metadata", None)
        if isinstance(ctx_metadata, dict):
            provider_model = ctx_metadata.get(
                "_last_provider_model"
            ) or ctx_metadata.get("model")
            if provider_model:
                hook_metadata.setdefault("provider_model", provider_model)
            client_model = ctx_metadata.get("_last_client_model")
            if client_model:
                hook_metadata.setdefault("client_model", client_model)
            alias_map = ctx_metadata.get("_model_alias_map")
            if isinstance(alias_map, dict) and alias_map:
                hook_metadata.setdefault("_model_alias_map", dict(alias_map))

        hook_context = HookContext(
            event=HookEvent.PROVIDER_REQUEST_PREPARED,
            timestamp=datetime.utcnow(),
            data=hook_data,
            metadata=hook_metadata,
            request=request_obj,
            provider=provider_name,
        )

        try:
            await hook_manager.emit_with_context(hook_context, fire_and_forget=False)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug(
                "provider_request_prepared_hook_failed",
                provider=provider_name,
                error=str(exc),
            )
            return method, target_url, prepared_body, prepared_headers

        mutated = hook_context.data or {}
        mutated_method = str(mutated.get("method", method))
        mutated_url = str(mutated.get("url", target_url))
        mutated_headers = self._coerce_hook_headers(
            mutated.get("headers"),
            prepared_headers,
        )
        mutated_body = self._coerce_hook_body(
            mutated.get("body"),
            mutated.get("body_kind", body_kind),
            mutated.get("body_raw"),
            prepared_body,
        )

        return mutated_method, mutated_url, mutated_body, mutated_headers

    def _prepare_body_for_hook(self, body: bytes) -> tuple[Any, str]:
        """Return hook-friendly body representation and its kind."""

        if not body:
            return b"", "bytes"

        try:
            decoded = body.decode()
        except UnicodeDecodeError:
            return body, "bytes"

        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError:
            return decoded, "text"

        return parsed, "json"

    def _coerce_hook_body(
        self,
        body: Any,
        body_kind: str,
        body_raw: Any,
        original: bytes,
    ) -> bytes:
        """Convert hook-mutated body back to bytes safely."""

        coerced_raw = self._ensure_bytes(body_raw)
        if coerced_raw is not None:
            return coerced_raw

        converted = self._convert_hook_body_payload(body, body_kind)
        if converted is not None and converted != original:
            return converted

        if converted is not None:
            return converted

        return original

    def _ensure_bytes(self, value: Any) -> bytes | None:
        """Best-effort conversion to bytes."""

        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, str):
            return value.encode()
        return None

    def _coerce_hook_headers(
        self,
        headers: Any,
        original: dict[str, str],
    ) -> dict[str, str]:
        """Sanitize hook-mutated headers."""

        if headers is None:
            return original

        items: Sequence[tuple[Any, Any]] | None = None
        if isinstance(headers, Mapping):
            items = list(headers.items())
        elif isinstance(headers, Sequence):
            try:
                items = [tuple(pair) for pair in headers]
            except Exception:  # pragma: no cover - defensive
                items = None

        if not items:
            return original

        coerced: dict[str, str] = {}
        for key, value in items:
            try:
                coerced_key = str(key).lower()
                coerced_value = str(value)
            except Exception:
                logger.debug(
                    "provider_request_prepared_header_dropped",
                    header_key=key,
                )
                continue
            coerced[coerced_key] = coerced_value

        return coerced or original

    def _convert_hook_body_payload(self, body: Any, body_kind: str) -> bytes | None:
        """Convert hook-provided body payload into bytes when possible."""

        if body is None:
            return None

        direct = self._ensure_bytes(body)
        if direct is not None:
            return direct

        try:
            if isinstance(body, dict | list) or body_kind == "json":
                return json.dumps(body).encode()
            if isinstance(body, int | float | bool):
                return json.dumps(body).encode()
            if isinstance(body, str):
                return body.encode()
        except (TypeError, ValueError) as exc:
            logger.debug(
                "provider_request_prepared_body_conversion_failed",
                error=str(exc),
            )
            return None

        logger.debug(
            "provider_request_prepared_body_unmodified",
            reason="unsupported_type",
            body_type=type(body).__name__,
        )
        return None

    def _restore_model_response(self, response: Response, ctx: Any) -> Response:
        """Restore original model identifiers in JSON responses."""

        metadata = getattr(ctx, "metadata", None)
        if not isinstance(metadata, dict) or "_model_alias_map" not in metadata:
            return response

        try:
            payload = self._decode_json_body(
                cast(bytes, response.body), context="restore"
            )
        except ValueError:
            return response

        alias_map = (
            metadata.get("_model_alias_map") if isinstance(metadata, dict) else None
        )
        if not alias_map:
            alias_map = getattr(ctx, "_model_alias_map", None)
        if (
            isinstance(payload, dict)
            and isinstance(alias_map, Mapping)
            and isinstance(payload.get("model"), str)
        ):
            payload["model"] = alias_map.get(payload["model"], payload["model"])

        restore_model_aliases(payload, metadata)
        response.body = self._encode_json_body(payload)
        return response

    @abstractmethod
    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        """Provider prepares request. Headers have lowercase keys."""
        pass

    @abstractmethod
    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response | StreamingResponse:
        """Provider processes response."""
        pass

    @abstractmethod
    async def get_target_url(self, endpoint: str) -> str:
        """Get target URL for this provider."""
        pass

    async def _apply_format_chain(
        self,
        *,
        data: dict[str, Any],
        format_chain: list[str],
        stage: Literal["request", "response", "error"],
    ) -> dict[str, Any]:
        if not self.format_registry:
            raise RuntimeError("Format registry is not configured")

        pairs = self._build_chain_pairs(format_chain, stage)
        current = data
        for step_index, (from_format, to_format) in enumerate(pairs, start=1):
            adapter = self.format_registry.get(from_format, to_format)
            logger.debug(
                "format_chain_step_start",
                from_format=from_format,
                to_format=to_format,
                stage=stage,
                step=step_index,
            )

            if stage == "request":
                current = await adapter.convert_request(current)
            elif stage == "response":
                current = await adapter.convert_response(current)
            elif stage == "error":
                current = await adapter.convert_error(current)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported format chain stage: {stage}")

            logger.debug(
                "format_chain_step_completed",
                from_format=from_format,
                to_format=to_format,
                stage=stage,
                step=step_index,
            )

        return current

    def _build_chain_pairs(
        self, format_chain: list[str], stage: Literal["request", "response", "error"]
    ) -> list[tuple[str, str]]:
        if len(format_chain) < 2:
            return []

        if stage == "response":
            pairs = [
                (format_chain[i + 1], format_chain[i])
                for i in range(len(format_chain) - 1)
            ]
            pairs.reverse()
            return pairs

        return [
            (format_chain[i], format_chain[i + 1]) for i in range(len(format_chain) - 1)
        ]

    def _ensure_format_registry(
        self, format_chain: list[str] | None, endpoint: str
    ) -> None:
        """Ensure format registry is available when a format chain is provided."""

        if format_chain and len(format_chain) > 1 and self.format_registry is None:
            logger.error(
                "format_registry_missing_for_chain",
                endpoint=endpoint,
                chain=format_chain,
                category="format",
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "type": "configuration_error",
                        "message": "Format registry is not configured but a format chain was requested.",
                        "details": {
                            "endpoint": endpoint,
                            "format_chain": format_chain,
                        },
                    }
                },
            )

    def _decode_json_body(self, body: bytes, *, context: str) -> dict[str, Any]:
        if not body:
            return {}

        try:
            parsed = json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:  # pragma: no cover
            raise ValueError(f"{context} body is not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"{context} body must be a JSON object, got {type(parsed).__name__}"
            )

        return parsed

    def _encode_json_body(self, data: dict[str, Any]) -> bytes:
        try:
            return json.dumps(data).encode()
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to serialize format chain output: {exc}") from exc

    async def _execute_http_request(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> httpx.Response:
        """Execute HTTP request."""
        # Convert to canonical headers for HTTP
        canonical_headers = headers

        # Get HTTP client
        client = await self.http_pool_manager.get_client()

        # Execute
        response: httpx.Response = await client.request(
            method=method,
            url=url,
            headers=canonical_headers,
            content=body,
            timeout=120.0,
        )
        return response

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.debug("adapter_cleanup_completed")
