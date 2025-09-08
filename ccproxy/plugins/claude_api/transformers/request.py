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

    Modes:
    - none: No system prompt injection
    - minimal: Only inject the first system prompt (basic Claude Code identification)
    - full: Inject complete system prompt with all instructions
    """

    def __init__(
        self,
        detection_service: ClaudeAPIDetectionService | None = None,
        mode: str = "minimal",
    ):
        """Initialize the request transformer.

        Args:
            detection_service: ClaudeAPIDetectionService instance for header/prompt injection
            mode: Prompt injection mode - "none", "minimal" or "full" (default: "minimal")
        """
        self.detection_service = detection_service
        self.mode = mode.lower()
        if self.mode not in ("none", "minimal", "full"):
            self.mode = "minimal"  # Default to minimal if invalid mode

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
            has_authorization="authorization" in headers,
            category="transform",
        )

        transformed = headers.copy()

        # Strip potentially problematic headers (aligned with main branch logic)
        excluded_headers = {
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
            # Additional headers from main branch that cause issues
            "x-forwarded-for",
            "x-forwarded-proto",
            "x-forwarded-host",
            "forwarded",
            # Authentication headers to be replaced
            "x-api-key",
            # Compression headers to avoid decompression issues
            "accept-encoding",
            "content-encoding",
            # CORS headers - should not be forwarded to upstream
            "origin",
            "access-control-request-method",
            "access-control-request-headers",
            "access-control-allow-origin",
            "access-control-allow-methods",
            "access-control-allow-headers",
            "access-control-allow-credentials",
            "access-control-max-age",
            "access-control-expose-headers",
            "authorization",  # Will be re-injected if access_token is provided
        }
        transformed = {
            k: v for k, v in transformed.items() if k.lower() not in excluded_headers
        }

        # Inject detected headers if available
        has_detected_headers = False
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.headers:
                # Prefer preserved order/case if available, otherwise fall back
                try:
                    detected_headers = cached_data.headers_ordered_dict()
                except Exception:
                    detected_headers = {}
                if not detected_headers:
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
        transformed["authorization"] = f"Bearer {access_token}"

        # exclude_headers
        excluded_headers.remove("authorization")  # We may have just added this
        transformed = {
            k: v for k, v in transformed.items() if k.lower() not in excluded_headers
        }

        # Debug logging - what headers are we returning?
        logger.trace(
            "transform_headers_result",
            has_x_api_key="x-api-key" in transformed,
            has_authorization="authorization" in transformed,
            header_count=len(transformed),
            detected_headers_used=has_detected_headers,
            category="transform",
        )

        return transformed

    def _find_cache_control_blocks(self, data: dict[str, Any]) -> list[tuple[str, int, int]]:
        """Find all cache_control blocks in the request with their locations.

        Returns:
            List of tuples (location_type, location_index, block_index) for each cache_control block
            where location_type is 'system' or 'message'
        """
        blocks = []

        # Find in system field
        system = data.get("system")
        if isinstance(system, list):
            for i, block in enumerate(system):
                if isinstance(block, dict) and "cache_control" in block:
                    blocks.append(("system", 0, i))

        # Find in messages
        messages = data.get("messages", [])
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content")
            if isinstance(content, list):
                for block_idx, block in enumerate(content):
                    if isinstance(block, dict) and "cache_control" in block:
                        blocks.append(("message", msg_idx, block_idx))

        return blocks

    def _limit_cache_control_blocks(
        self, data: dict[str, Any], max_blocks: int = 4
    ) -> dict[str, Any]:
        """Limit the number of cache_control blocks to comply with Anthropic's limit.

        Simple algorithm: Remove cache_control from the last N blocks when exceeding the limit.
        This preserves the first blocks (which are typically system prompts) and removes
        from user messages at the end.

        Args:
            data: Request data dictionary
            max_blocks: Maximum number of cache_control blocks allowed (default: 4)

        Returns:
            Modified data dictionary with cache_control blocks limited
        """
        import copy

        # Deep copy to avoid modifying original
        data = copy.deepcopy(data)

        # Find all cache_control blocks
        cache_blocks = self._find_cache_control_blocks(data)
        total_blocks = len(cache_blocks)

        if total_blocks <= max_blocks:
            # No need to remove anything
            return data

        logger = get_plugin_logger()
        logger.warning(
            "cache_control_limit_exceeded",
            total_blocks=total_blocks,
            max_blocks=max_blocks,
            category="transform",
        )

        # Calculate how many to remove
        to_remove = total_blocks - max_blocks

        # Remove cache_control from the last N blocks
        blocks_to_remove = cache_blocks[-to_remove:]
        
        for location_type, location_index, block_index in blocks_to_remove:
            if location_type == "system":
                system = data.get("system")
                if isinstance(system, list) and block_index < len(system):
                    block = system[block_index]
                    if isinstance(block, dict) and "cache_control" in block:
                        del block["cache_control"]
                        logger.debug(
                            "removed_cache_control",
                            location="system",
                            block_index=block_index,
                            category="transform",
                        )
            elif location_type == "message":
                messages = data.get("messages", [])
                if location_index < len(messages):
                    content = messages[location_index].get("content")
                    if isinstance(content, list) and block_index < len(content):
                        block = content[block_index]
                        if isinstance(block, dict) and "cache_control" in block:
                            del block["cache_control"]
                            logger.debug(
                                "removed_cache_control",
                                location="message",
                                message_index=location_index,
                                block_index=block_index,
                                category="transform",
                            )

        return data

    def transform_body(self, body: bytes | None) -> bytes | None:
        """Transform request body.

        Injects detected system prompt from Claude CLI and manages cache_control blocks.

        Args:
            body: Original request body

        Returns:
            Transformed body with system prompt injected and cache_control blocks limited
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

        # Check if injection is disabled
        if self.mode == "none":
            logger.trace(
                "system_prompt_injection_disabled",
                mode=self.mode,
                category="transform",
            )
        # Inject system prompt if available and not in "none" mode
        elif self.detection_service:
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
                mode=self.mode,
                category="transform",
            )
            if cached_data and cached_data.system_prompt:
                system_field = cached_data.system_prompt.system_field

                # Get the system prompt to inject based on mode
                detected_system = None
                if self.mode == "minimal":
                    # In minimal mode, only inject the first system prompt
                    if isinstance(system_field, list) and len(system_field) > 0:
                        # Keep only the first element (Claude Code identification)
                        # Preserve its complete structure including cache_control
                        detected_system = [system_field[0]]
                        logger.trace(
                            "prepared_minimal_system_prompt",
                            version=cached_data.claude_version,
                            system_type="list",
                            system_elements=1,
                            has_cache_control="cache_control" in system_field[0]
                            if isinstance(system_field[0], dict)
                            else False,
                            category="transform",
                        )
                    elif isinstance(system_field, str):
                        # If it's a string, take only the first sentence/line
                        first_line = (
                            system_field.split("\n")[0]
                            if "\n" in system_field
                            else system_field
                        )
                        detected_system = first_line
                        logger.trace(
                            "prepared_minimal_system_prompt",
                            version=cached_data.claude_version,
                            system_type="string",
                            system_length=len(first_line),
                            category="transform",
                        )
                    else:
                        # Fallback to full field if format is unexpected
                        detected_system = system_field
                elif self.mode == "full":
                    # Full mode - inject complete system prompt
                    detected_system = system_field
                    logger.trace(
                        "prepared_full_system_prompt",
                        version=cached_data.claude_version,
                        system_type=type(system_field).__name__,
                        system_length=len(str(system_field)),
                        system_elements=len(system_field)
                        if isinstance(system_field, list)
                        else 1,
                        category="transform",
                    )

                # Always inject the detected system prompt (prepend to existing if present)
                if detected_system is not None:
                    existing_system = data.get("system")

                    if existing_system is None:
                        # No existing system prompt, inject the detected one
                        data["system"] = detected_system
                        logger.debug(
                            "injected_system_prompt_new",
                            version=cached_data.claude_version,
                            mode=self.mode,
                            category="transform",
                        )
                    else:
                        # Request has existing system prompt, prepend the detected one
                        if isinstance(detected_system, str):
                            # Detected system is a string
                            if isinstance(existing_system, str):
                                # Both are strings, convert to list format
                                data["system"] = [
                                    {"type": "text", "text": detected_system},
                                    {"type": "text", "text": existing_system},
                                ]
                            elif isinstance(existing_system, list):
                                # Detected is string, existing is list
                                data["system"] = [
                                    {"type": "text", "text": detected_system}
                                ] + existing_system
                        elif isinstance(detected_system, list):
                            # Detected system is a list
                            if isinstance(existing_system, str):
                                # Detected is list, existing is string
                                data["system"] = detected_system + [
                                    {"type": "text", "text": existing_system}
                                ]
                            elif isinstance(existing_system, list):
                                # Both are lists, concatenate (detected first)
                                data["system"] = detected_system + existing_system

                        logger.debug(
                            "injected_system_prompt_prepended",
                            version=cached_data.claude_version,
                            mode=self.mode,
                            existing_system_type=type(existing_system).__name__,
                            detected_system_type=type(detected_system).__name__,
                            category="transform",
                        )
            else:
                logger.debug(
                    "system_prompt_not_injected",
                    reason="no_cached_data"
                    if not cached_data
                    else "no_system_prompt"
                    if not cached_data.system_prompt
                    else "unknown",
                    mode=self.mode,
                    category="transform",
                )
        else:
            logger.debug("no_detection_service_available", category="transform")

        # Limit cache_control blocks to comply with Anthropic's limit
        data = self._limit_cache_control_blocks(data)

        return json.dumps(data).encode("utf-8")
