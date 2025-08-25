"""Claude API request transformer."""

import json
from typing import Any

from ccproxy.core.logging import get_plugin_logger

from ..detection_service import ClaudeAPIDetectionService


logger = get_plugin_logger()


class ClaudeAPIRequestTransformer:
    """Transform requests for Claude API.

    Handles:
    - Header transformation and auth injection
    - Claude CLI headers injection from detection service
    - System prompt injection from detected Claude CLI data
    """

    def __init__(self, detection_service: ClaudeAPIDetectionService | None = None):
        """Initialize the request transformer.

        Args:
            detection_service: ClaudeAPIDetectionService instance for header/prompt injection
        """
        self.detection_service = detection_service

    def transform_headers(
        self,
        headers: dict[str, str],
        access_token: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Transform request headers.

        Injects detected Claude CLI headers for proper authentication.

        Args:
            headers: Original request headers
            session_id: Optional session ID
            access_token: Optional access token
            **kwargs: Additional parameters

        Returns:
            Transformed headers with Claude CLI headers injected
        """
        # Get logger with request context at the start of the function
        logger = get_plugin_logger()

        # Debug logging
        logger.trace(
            "transform_headers_called",
            has_access_token=access_token is not None,
            access_token_length=len(access_token) if access_token else 0,
            header_count=len(headers),
            has_x_api_key="x-api-key" in headers,
            has_authorization="Authorization" in headers,
            category="transform",
        )

        transformed = headers.copy()

        # Remove hop-by-hop headers and client auth headers
        hop_by_hop = {
            "host",
            "connection",
            "keep-alive",
            "transfer-encoding",
            "content-length",
            "upgrade",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "x-api-key",  # Remove client's x-api-key header
            "authorization",  # Remove client's Authorization header
        }
        transformed = {
            k: v for k, v in transformed.items() if k.lower() not in hop_by_hop
        }

        # Inject detected headers if available
        has_detected_headers = False
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.headers:
                detected_headers = cached_data.headers.to_headers_dict()
                logger.trace(
                    "injecting_detected_headers",
                    version=cached_data.claude_version,
                    header_count=len(detected_headers),
                    category="transform",
                )
                # Detected headers take precedence
                transformed.update(detected_headers)
                has_detected_headers = True

        if not access_token:
            raise RuntimeError("access_token parameter is required")

        # Inject access token in Authentication header
        transformed["Authorization"] = f"Bearer {access_token}"

        # Debug logging - what headers are we returning?
        logger.trace(
            "transform_headers_result",
            has_x_api_key="x-api-key" in transformed,
            has_authorization="Authorization" in transformed,
            header_count=len(transformed),
            detected_headers_used=has_detected_headers,
            category="transform",
        )

        return transformed

    def transform_body(self, body: bytes | None) -> bytes | None:
        """Transform request body.

        Injects detected system prompt from Claude CLI.

        Args:
            body: Original request body

        Returns:
            Transformed body with system prompt injected
        """
        # Get logger with request context at the start of the function
        logger = get_plugin_logger()

        logger.trace(
            "transform_body_called",
            has_body=body is not None,
            body_length=len(body) if body else 0,
            has_detection_service=self.detection_service is not None,
            category="transform",
        )

        if not body:
            return body

        try:
            data = json.loads(body.decode("utf-8"))
            logger.trace(
                "parsed_request_body",
                keys=list(data.keys()),
                category="transform",
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(
                "body_decode_failed",
                error=str(e),
                body_preview=body[:100].decode("utf-8", errors="replace")
                if body
                else None,
                category="transform",
            )
            return body

        # Inject system prompt if available
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            logger.debug(
                "checking_cached_data",
                has_cached_data=cached_data is not None,
                has_system_prompt=cached_data.system_prompt is not None
                if cached_data
                else False,
                has_system_field=cached_data.system_prompt.system_field is not None
                if cached_data and cached_data.system_prompt
                else False,
                system_already_in_data="system" in data,
                category="transform",
            )
            if cached_data and cached_data.system_prompt and "system" not in data:
                # Inject the detected system prompt (as list or string)
                data["system"] = cached_data.system_prompt.system_field
                logger.trace(
                    "injected_system_prompt",
                    version=cached_data.claude_version,
                    system_type=type(cached_data.system_prompt.system_field).__name__,
                    system_length=len(str(cached_data.system_prompt.system_field)),
                    category="transform",
                )
            else:
                logger.debug(
                    "system_prompt_not_injected",
                    reason="no_cached_data"
                    if not cached_data
                    else "no_system_prompt"
                    if not cached_data.system_prompt
                    else "system_already_exists"
                    if "system" in data
                    else "unknown",
                    category="transform",
                )
        else:
            logger.debug("no_detection_service_available", category="transform")

        return json.dumps(data).encode("utf-8")
