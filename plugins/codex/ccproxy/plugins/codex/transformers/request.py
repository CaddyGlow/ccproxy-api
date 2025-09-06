"""Codex request transformer - headers and auth only."""

import json
from typing import Any

from ccproxy.core.logging import get_plugin_logger

from ..detection_service import CodexDetectionService


logger = get_plugin_logger()


class CodexRequestTransformer:
    """Transform requests for Codex API.

    Handles:
    - Header transformation and auth injection
    - Codex CLI headers injection from detection service
    - System prompt injection (instructions field) injection
    """

    def __init__(self, detection_service: CodexDetectionService | None = None):
        """Initialize the request transformer.

        Args:
            detection_service: CodexDetectionService for header/instructions injection
        """
        self.detection_service = detection_service

    def transform_headers(
        self,
        headers: dict[str, str] | Any,
        session_id: str | None = None,
        access_token: str | None = None,
        chatgpt_account_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Transform request headers for Codex API.

        Args:
            headers: Original request headers
            session_id: Codex session ID
            access_token: Optional Bearer token for authorization
            **kwargs: Additional arguments

        Returns:
            Transformed headers with Codex-specific headers
        """
        # Get logger with request context at the start of the function
        logger = get_plugin_logger()

        # Debug logging
        logger.debug(
            "transform_headers",
            has_session_id=session_id is not None,
            has_access_token=access_token is not None,
            access_token_length=len(access_token) if access_token else 0,
            header_count=len(headers),
            has_authorization="Authorization" in headers,
            request_id=kwargs.get("request_id"),
        )

        # Normalize potential HeaderBag to dict
        if hasattr(headers, "to_dict"):
            try:
                headers = headers.to_dict()
            except Exception:
                headers = dict(headers)

        transformed = headers.copy()

        # Remove hop-by-hop headers
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
        }
        transformed = {
            k: v for k, v in transformed.items() if k.lower() not in hop_by_hop
        }

        # Inject detected headers if available, otherwise use fallback headers
        has_detected_headers = False
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.headers:
                # Prefer preserved order/case if available, otherwise fall back
                try:
                    detected_headers = cached_data.headers_ordered_dict()
                except Exception:
                    detected_headers = {}

                # Merge in typed/normalized headers to ensure required fields
                # like chatgpt-account-id are present even if not captured raw.
                typed_headers = cached_data.headers.to_headers_dict()
                for k, v in typed_headers.items():
                    if k not in detected_headers or not detected_headers.get(k):
                        detected_headers[k] = v

                logger.debug(
                    "injecting_detected_headers",
                    version=cached_data.codex_version,
                    header_count=len(detected_headers),
                    request_id=kwargs.get("request_id"),
                )
                # Detected headers take precedence
                transformed.update(detected_headers)
                has_detected_headers = True
        else:
            # Use fallback headers when no detection service
            fallback_headers = {
                "originator": "codex_cli_rs",
                "version": "0.21.0",
                "openai_beta": "responses=experimental",
            }
            transformed.update(fallback_headers)
            logger.debug(
                "injecting_fallback_headers",
                header_count=len(fallback_headers),
                request_id=kwargs.get("request_id"),
            )

        # If an explicit access_token is provided, inject it; otherwise, trust
        # headers prepared upstream (adapter/credential manager) without failing here.

        # TODO: Disabled injection of content-type and accept headers for now
        # it's normally set by the client
        # if "content-type" not in [k.lower() for k in transformed]:
        #     transformed["Content-Type"] = "application/json"
        # if "accept" not in [k.lower() for k in transformed]:
        #     transformed["Accept"] = "application/json"

        # if "accept" not in [k.lower() for k in transformed]:
        #     transformed["Accept"] = "application/json"
        transformed["accept"] = "text/event-stream"
        # Inject session id if provided
        if session_id:
            transformed["session_id"] = session_id

        # Inject access token in Authentication header only when provided
        if access_token:
            transformed["authorization"] = f"Bearer {access_token}"

        # Inject chatgpt_account_id if provided
        if chatgpt_account_id:
            transformed["chatgpt-account-id"] = chatgpt_account_id
            logger.debug(
                "injected_chatgpt_account_id",
                account_id_length=len(chatgpt_account_id),
                request_id=kwargs.get("request_id"),
            )

        # Debug logging - what headers are we returning?
        logger.debug(
            "transform_headers_result",
            has_authorization="Authorization" in transformed,
            has_chatgpt_account_id="chatgpt-account-id" in transformed,
            header_count=len(transformed),
            detected_headers_used=has_detected_headers,
            request_id=kwargs.get("request_id"),
        )

        return transformed

    def transform_body(self, body: bytes | None, **kwargs: Any) -> bytes | None:
        """Minimal body transformation - inject instructions if missing.

        Args:
            body: Original request body

        Returns:
            Body with instructions injected if needed
        """
        logger = get_plugin_logger()

        logger.debug(
            "transform_body",
            has_body=body is not None,
            body_length=len(body) if body else 0,
            has_detection_service=self.detection_service is not None,
            request_id=kwargs.get("request_id"),
        )

        if not body:
            return body

        try:
            # Debug: log raw body before parsing
            raw_body_preview = body.decode("utf-8")[:500] if body else "empty"
            logger.debug(
                "body_transformer_raw_input",
                body_length=len(body) if body else 0,
                raw_body_preview=raw_body_preview,
                category="transform",
            )

            data = json.loads(body.decode("utf-8"))
            logger.debug(
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

        # Only inject instructions if missing or None
        if "instructions" not in data or data.get("instructions") is None:
            instructions = self._get_instructions()
            logger.debug(
                "getting_instructions",
                has_detection_service=bool(self.detection_service),
                instructions_length=len(instructions) if instructions else 0,
                category="transform",
            )
            if instructions:
                data["instructions"] = instructions
                logger.debug(
                    "injected_codex_instructions",
                    instructions_length=len(instructions),
                    instructions_preview=f"{instructions[:100]}..."
                    if len(instructions) > 100
                    else instructions,
                    category="transform",
                )
            else:
                logger.warning("no_codex_instructions_available", category="transform")
        else:
            logger.info(
                "instructions_already_present",
                length=len(data.get("instructions", "")),
                category="transform",
            )

        result = json.dumps(data).encode("utf-8")
        logger.debug(
            "transform_body_result", result_length=len(result), category="transform"
        )
        return result

    def _get_instructions(self) -> str | None:
        """Get Codex instructions from detection service or fallback."""
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.instructions:
                return cached_data.instructions.instructions_field

        # Fallback instructions
        return (
            "You are a coding agent running in the Codex CLI, a terminal-based coding assistant. "
            "Codex CLI is an open source project led by OpenAI."
        )
