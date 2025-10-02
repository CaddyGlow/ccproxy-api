import json
import time
import uuid
from typing import Any, cast

import httpx
from starlette.requests import Request
from starlette.responses import Response

from ccproxy.core.logging import get_plugin_logger
from ccproxy.llms.models.openai import ResponseObject
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.utils.headers import (
    extract_request_headers,
    extract_response_headers,
    filter_request_headers,
    filter_response_headers,
)

from .config import CopilotConfig
from .detection_service import CopilotDetectionService
from .manager import CopilotTokenManager
from .oauth.provider import CopilotOAuthProvider


logger = get_plugin_logger()


class CopilotAdapter(BaseHTTPAdapter):
    """Simplified Copilot adapter."""

    def __init__(
        self,
        config: CopilotConfig,
        auth_manager: CopilotTokenManager | None,
        detection_service: CopilotDetectionService,
        http_pool_manager: Any,
        oauth_provider: CopilotOAuthProvider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            auth_manager=auth_manager,
            http_pool_manager=http_pool_manager,
            **kwargs,
        )
        self.oauth_provider = oauth_provider
        self.detection_service = detection_service
        self.token_manager: CopilotTokenManager | None = cast(
            CopilotTokenManager | None, self.auth_manager
        )

        self.base_url = self.config.base_url.rstrip("/")

    async def get_target_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        access_token = await self._resolve_access_token()

        wants_stream = False
        try:
            parsed_body = json.loads(body.decode()) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed_body = None
        else:
            if isinstance(parsed_body, dict):
                wants_stream = bool(parsed_body.get("stream"))

        # Filter headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)

        # Add Copilot headers (lowercase keys)
        copilot_headers = {
            key.lower(): str(value)
            for key, value in self.config.api_headers.items()
            if value is not None
        }

        cli_headers = self._collect_cli_headers()
        for key, value in cli_headers.items():
            copilot_headers.setdefault(key, value)

        copilot_headers["authorization"] = f"Bearer {access_token}"
        copilot_headers["x-request-id"] = str(uuid.uuid4())

        if wants_stream and "accept" not in filtered_headers:
            copilot_headers.setdefault("accept", "text/event-stream")

        # Merge headers
        final_headers = {**filtered_headers, **copilot_headers}

        logger.debug("copilot_request_prepared", header_count=len(final_headers))

        return body, final_headers

    async def _resolve_access_token(self) -> str:
        """Resolve a usable Copilot access token via the configured manager."""

        auth_manager_name = (
            getattr(self.config, "auth_manager", None) or "oauth_copilot"
        )

        token_manager = self.token_manager
        if token_manager is None:
            from ccproxy.core.errors import AuthenticationError

            logger.warning(
                "auth_manager_override_not_resolved",
                plugin="copilot",
                auth_manager_name=auth_manager_name,
                category="auth",
            )
            raise AuthenticationError(
                "Authentication manager not configured for Copilot provider"
            )

        async def _snapshot_token() -> str | None:
            snapshot = await token_manager.get_token_snapshot()
            if snapshot and snapshot.access_token:
                return str(snapshot.access_token)
            return None

        credentials = await token_manager.load_credentials()
        if not credentials:
            fallback = await _snapshot_token()
            if fallback:
                return fallback
            raise ValueError("No Copilot credentials available")

        try:
            if token_manager.should_refresh(credentials):
                logger.debug("copilot_token_refresh_due", category="auth")
                refreshed = await token_manager.get_access_token_with_refresh()
                if refreshed:
                    return refreshed
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "copilot_token_refresh_failed",
                error=str(exc),
                category="auth",
            )
            fallback = await _snapshot_token()
            if fallback:
                return fallback

        try:
            token = await token_manager.get_access_token()
            if token:
                return token
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "copilot_token_fetch_failed",
                error=str(exc),
                category="auth",
            )

        fallback = await _snapshot_token()
        if fallback:
            return fallback

        raise ValueError("No valid Copilot access token available")

    def _collect_cli_headers(self) -> dict[str, str]:
        """Collect additional headers suggested by CLI detection service."""

        if not self.detection_service:
            return {}

        try:
            recommended = self.detection_service.get_recommended_headers()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "copilot_detection_headers_failed",
                error=str(exc),
                category="headers",
            )
            return {}

        if not isinstance(recommended, dict):
            return {}

        headers: dict[str, str] = {}
        blocked = {"authorization", "x-request-id"}
        for key, value in recommended.items():
            if not isinstance(key, str) or value is None:
                continue
            lower_key = key.lower()
            if lower_key in blocked:
                continue
            headers[lower_key] = str(value)

        return headers

    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response:
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

    async def handle_request_gh_api(self, request: Request) -> Response:
        """Forward request to GitHub API with proper authentication.

        Args:
            path: API path (e.g., '/copilot_internal/user')
            mode: API mode - 'api' for GitHub API with OAuth token, 'copilot' for Copilot API with Copilot token
            method: HTTP method
            body: Request body
            extra_headers: Additional headers
        """
        auth_manager_name = (
            getattr(self.config, "auth_manager", None) or "oauth_copilot"
        )

        if self.auth_manager is None:
            from ccproxy.core.errors import AuthenticationError

            logger.warning(
                "auth_manager_override_not_resolved",
                plugin="copilot",
                auth_manager_name=auth_manager_name,
                category="auth",
            )
            raise AuthenticationError(
                "Authentication manager not configured for Copilot provider"
            )
        oauth_provider = self.oauth_provider
        if oauth_provider is None:
            from ccproxy.core.errors import AuthenticationError

            logger.warning(
                "oauth_provider_not_available",
                plugin="copilot",
                category="auth",
            )
            raise AuthenticationError(
                "OAuth provider not configured for Copilot provider"
            )

        access_token = await oauth_provider.ensure_oauth_token()
        base_url = "https://api.github.com"

        base_headers = {
            "authorization": f"Bearer {access_token}",
            "accept": "application/json",
        }
        # Get context from middleware (already initialized)
        ctx = request.state.context

        # Step 1: Extract request data
        body = await request.body()
        request_headers = extract_request_headers(request)
        method = request.method
        endpoint = ctx.metadata.get("endpoint", "")
        target_url = f"{base_url}{endpoint}"

        outgoing_headers = filter_request_headers(request_headers, preserve_auth=False)
        outgoing_headers.update(base_headers)

        provider_response = await self._execute_http_request(
            method,
            target_url,
            outgoing_headers,
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
