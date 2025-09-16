import json
import time
import uuid
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.llms.openai.models import ResponseObject
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.utils.headers import (
    extract_request_headers,
    extract_response_headers,
    filter_request_headers,
    filter_response_headers,
)

from .config import CopilotConfig
from .oauth.provider import CopilotOAuthProvider


logger = get_plugin_logger()


class CopilotAdapter(BaseHTTPAdapter):
    """Simplified Copilot adapter."""

    def __init__(
        self, oauth_provider: CopilotOAuthProvider, config: CopilotConfig, **kwargs: Any
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.oauth_provider = oauth_provider

        self.base_url = self.config.base_url.rstrip("/")

    async def get_target_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        # Get auth token
        access_token = await self.oauth_provider.ensure_copilot_token()

        # Filter headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)

        # Add Copilot headers (lowercase keys)
        copilot_headers = {}
        for key, value in self.config.api_headers.items():
            copilot_headers[key.lower()] = value

        copilot_headers["authorization"] = f"Bearer {access_token}"
        copilot_headers["x-request-id"] = str(uuid.uuid4())

        # Merge headers
        final_headers = {}
        final_headers.update(filtered_headers)
        final_headers.update(copilot_headers)

        logger.debug("copilot_request_prepared", header_count=len(final_headers))

        return body, final_headers

    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Process provider response with format conversion support."""
        # Streaming detection and handling is centralized in BaseHTTPAdapter.
        # Always return a plain Response for non-streaming flows.
        response_headers = extract_response_headers(response)

        # Normalize Copilot chat completion payloads to include the required
        # OpenAI "created" timestamp field. GitHub's API occasionally omits it,
        # but our OpenAI-compatible schema requires it for validation.
        if (
            response.status_code < 400
            and endpoint.endswith("/chat/completions")
            and "json" in (response.headers.get("content-type", "").lower())
        ):
            try:
                payload = response.json()
                if isinstance(payload, dict) and "choices" in payload:
                    if "created" not in payload or not isinstance(
                        payload["created"], int
                    ):
                        payload["created"] = int(time.time())
                        body = json.dumps(payload).encode()
                        response_headers["content-length"] = str(len(body))
                        return Response(
                            content=body,
                            status_code=response.status_code,
                            headers=response_headers,
                            media_type=response.headers.get("content-type"),
                        )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                # Fall back to the raw payload if normalization fails
                pass

        if (
            response.status_code < 400
            and endpoint.endswith("/responses")
            and "json" in (response.headers.get("content-type", "").lower())
        ):
            try:
                payload = response.json()
                normalized = self._normalize_response_payload(payload)
                if normalized is not None:
                    body = json.dumps(normalized).encode()
                    response_headers["content-length"] = str(len(body))
                    return Response(
                        content=body,
                        status_code=response.status_code,
                        headers=response_headers,
                        media_type=response.headers.get("content-type"),
                    )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                # Fall back to raw payload on normalization errors
                pass

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
        )

    async def _create_streaming_response(
        self, response: httpx.Response, endpoint: str
    ) -> DeferredStreaming:
        # Deprecated: streaming is centrally handled by BaseHTTPAdapter/StreamingHandler
        # Kept for compatibility; not used.
        raise NotImplementedError

    async def handle_request_gh_api(self, request: Request) -> Response:
        """Forward request to GitHub API with proper authentication.

        Args:
            path: API path (e.g., '/copilot_internal/user')
            mode: API mode - 'api' for GitHub API with OAuth token, 'copilot' for Copilot API with Copilot token
            method: HTTP method
            body: Request body
            extra_headers: Additional headers
        """
        access_token = await self.oauth_provider.ensure_oauth_token()
        base_url = "https://api.github.com"

        headers = {
            "authorization": f"Bearer {access_token}",
            "accept": "application/json",
        }
        # Get context from middleware (already initialized)
        ctx = request.state.context

        # Step 1: Extract request data
        body = await request.body()
        headers = extract_request_headers(request)
        method = request.method
        endpoint = ctx.metadata.get("endpoint", "")
        target_url = f"{base_url}{endpoint}"

        provider_response = await self._execute_http_request(
            method,
            target_url,
            headers,
            body,
        )

        filtered_headers = filter_response_headers(dict(provider_response.headers))

        return Response(
            content=provider_response.content,
            status_code=provider_response.status_code,
            headers=filtered_headers,
            media_type=provider_response.headers.get(
                "content-type", "application/json"
            ),
        )

    def _needs_format_conversion(self, endpoint: str) -> bool:
        # Deprecated: conversion handled via format chain in BaseHTTPAdapter
        return False

    def _normalize_response_payload(self, payload: Any) -> dict[str, Any] | None:
        """Normalize Response API payloads to align with OpenAI schema expectations."""
        from pydantic import ValidationError

        if not isinstance(payload, dict):
            return None

        try:
            # If already valid, return canonical dump
            model = ResponseObject.model_validate(payload)
            return model.model_dump(mode="json", exclude_none=True)
        except ValidationError:
            pass

        normalized: dict[str, Any] = {}
        response_id = str(payload.get("id") or f"resp-{uuid.uuid4().hex}")
        normalized["id"] = response_id
        normalized["object"] = payload.get("object") or "response"
        normalized["created_at"] = int(payload.get("created_at") or time.time())

        stop_reason = payload.get("stop_reason")
        status = payload.get("status") or self._map_stop_reason_to_status(stop_reason)
        normalized["status"] = status
        normalized["model"] = payload.get("model") or ""

        parallel_tool_calls = payload.get("parallel_tool_calls")
        normalized["parallel_tool_calls"] = bool(parallel_tool_calls)

        # Normalize usage structure
        usage_raw = payload.get("usage") or {}
        if isinstance(usage_raw, dict):
            input_tokens = int(
                usage_raw.get("input_tokens") or usage_raw.get("prompt_tokens") or 0
            )
            output_tokens = int(
                usage_raw.get("output_tokens")
                or usage_raw.get("completion_tokens")
                or 0
            )
            total_tokens = int(
                usage_raw.get("total_tokens") or (input_tokens + output_tokens)
            )
            cached_tokens = int(
                usage_raw.get("input_tokens_details", {}).get("cached_tokens")
                if isinstance(usage_raw.get("input_tokens_details"), dict)
                else usage_raw.get("cached_tokens", 0)
            )
            reasoning_tokens = int(
                usage_raw.get("output_tokens_details", {}).get("reasoning_tokens")
                if isinstance(usage_raw.get("output_tokens_details"), dict)
                else usage_raw.get("reasoning_tokens", 0)
            )
            normalized["usage"] = {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": cached_tokens},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
                "total_tokens": total_tokens,
            }

        # Normalize output items
        normalized_output: list[dict[str, Any]] = []
        for index, item in enumerate(payload.get("output") or []):
            if not isinstance(item, dict):
                continue
            normalized_item = dict(item)
            normalized_item["id"] = (
                normalized_item.get("id") or f"{response_id}_output_{index}"
            )
            normalized_item["status"] = normalized_item.get("status") or status
            normalized_item["type"] = normalized_item.get("type") or "message"
            normalized_item["role"] = normalized_item.get("role") or "assistant"

            content_blocks = []
            for part in normalized_item.get("content", []) or []:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "output_text" or part_type == "text":
                    text_part = {
                        "type": "output_text",
                        "text": part.get("text", ""),
                        "annotations": part.get("annotations") or [],
                    }
                else:
                    text_part = part
                content_blocks.append(text_part)
            normalized_item["content"] = content_blocks
            normalized_output.append(normalized_item)

        normalized["output"] = normalized_output

        optional_keys = [
            "metadata",
            "instructions",
            "max_output_tokens",
            "previous_response_id",
            "reasoning",
            "store",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "truncation",
            "user",
        ]

        for key in optional_keys:
            if key in payload and payload[key] is not None:
                normalized[key] = payload[key]

        try:
            model = ResponseObject.model_validate(normalized)
            return model.model_dump(mode="json", exclude_none=True)
        except ValidationError:
            logger.debug(
                "response_payload_normalization_failed",
                payload_keys=list(payload.keys()),
            )
            return None

    @staticmethod
    def _map_stop_reason_to_status(stop_reason: Any) -> str:
        mapping = {
            "end_turn": "completed",
            "max_output_tokens": "incomplete",
            "stop_sequence": "completed",
            "cancelled": "cancelled",
        }
        return mapping.get(stop_reason, "completed")
