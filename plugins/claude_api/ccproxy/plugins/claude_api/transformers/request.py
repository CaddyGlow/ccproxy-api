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

    def _count_cache_control_blocks(self, data: dict[str, Any]) -> dict[str, int]:
        """Count cache_control blocks in different parts of the request.

        Returns:
            Dictionary with counts for 'injected_system', 'user_system', and 'messages'
        """
        counts = {"injected_system": 0, "user_system": 0, "messages": 0}

        # Count in system field
        system = data.get("system")
        if system:
            if isinstance(system, str):
                # String system prompts don't have cache_control
                pass
            elif isinstance(system, list):
                # Count cache_control in system prompt blocks
                # The first block(s) are injected, rest are user's
                injected_count = 0
                for i, block in enumerate(system):
                    if isinstance(block, dict) and "cache_control" in block:
                        # Check if this is the injected prompt (contains Claude Code identity)
                        text = block.get("text", "")
                        if "Claude Code" in text or "Anthropic's official CLI" in text:
                            counts["injected_system"] += 1
                            injected_count = max(injected_count, i + 1)
                        elif i < injected_count:
                            # Part of injected system (multiple blocks)
                            counts["injected_system"] += 1
                        else:
                            counts["user_system"] += 1

        # Count in messages
        messages = data.get("messages", [])
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        counts["messages"] += 1

        return counts

    def _limit_cache_control_blocks(
        self, data: dict[str, Any], max_blocks: int = 4
    ) -> dict[str, Any]:
        """Limit the number of cache_control blocks to comply with Anthropic's limit.

        Priority order:
        1. Injected system prompt cache_control (highest priority - Claude Code identity)
        2. User's system prompt cache_control
        3. User's message cache_control (lowest priority)

        Args:
            data: Request data dictionary
            max_blocks: Maximum number of cache_control blocks allowed (default: 4)

        Returns:
            Modified data dictionary with cache_control blocks limited
        """
        import copy

        # Deep copy to avoid modifying original
        data = copy.deepcopy(data)

        # Count existing blocks
        counts = self._count_cache_control_blocks(data)
        total = counts["injected_system"] + counts["user_system"] + counts["messages"]

        if total <= max_blocks:
            # No need to remove anything
            return data

        logger = get_plugin_logger()
        logger.warning(
            "cache_control_limit_exceeded",
            total_blocks=total,
            max_blocks=max_blocks,
            injected=counts["injected_system"],
            user_system=counts["user_system"],
            messages=counts["messages"],
            category="transform",
        )

        # Calculate how many to remove
        to_remove = total - max_blocks
        removed = 0

        # Remove from messages first (lowest priority)
        if to_remove > 0 and counts["messages"] > 0:
            messages = data.get("messages", [])
            for msg in reversed(messages):  # Remove from end first
                if removed >= to_remove:
                    break
                content = msg.get("content")
                if isinstance(content, list):
                    for block in reversed(content):
                        if removed >= to_remove:
                            break
                        if isinstance(block, dict) and "cache_control" in block:
                            del block["cache_control"]
                            removed += 1
                            logger.debug(
                                "removed_cache_control",
                                location="message",
                                category="transform",
                            )

        # Remove from user system prompts next
        if removed < to_remove and counts["user_system"] > 0:
            system = data.get("system")
            if isinstance(system, list):
                # Find and remove cache_control from user system blocks (non-injected)
                for block in reversed(system):
                    if removed >= to_remove:
                        break
                    if isinstance(block, dict) and "cache_control" in block:
                        text = block.get("text", "")
                        # Skip injected prompts (highest priority)
                        if (
                            "Claude Code" not in text
                            and "Anthropic's official CLI" not in text
                        ):
                            del block["cache_control"]
                            removed += 1
                            logger.debug(
                                "removed_cache_control",
                                location="user_system",
                                category="transform",
                            )

        # In theory, we should never need to remove injected system cache_control
        # but include this for completeness
        if removed < to_remove:
            logger.error(
                "cannot_preserve_injected_cache_control",
                needed_to_remove=to_remove,
                actually_removed=removed,
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
            if cached_data and cached_data.system_prompt and "system" not in data:
                system_field = cached_data.system_prompt.system_field

                # Handle different modes
                if self.mode == "minimal":
                    # In minimal mode, only inject the first system prompt
                    if isinstance(system_field, list) and len(system_field) > 0:
                        # Keep only the first element (Claude Code identification)
                        # Preserve its complete structure including cache_control
                        data["system"] = [system_field[0]]
                        logger.trace(
                            "injected_minimal_system_prompt",
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
                        data["system"] = first_line
                        logger.trace(
                            "injected_minimal_system_prompt",
                            version=cached_data.claude_version,
                            system_type="string",
                            system_length=len(first_line),
                            category="transform",
                        )
                    else:
                        # Fallback to full field if format is unexpected
                        data["system"] = system_field
                elif self.mode == "full":
                    # Full mode - inject complete system prompt
                    data["system"] = system_field
                    logger.trace(
                        "injected_full_system_prompt",
                        version=cached_data.claude_version,
                        system_type=type(system_field).__name__,
                        system_length=len(str(system_field)),
                        system_elements=len(system_field)
                        if isinstance(system_field, list)
                        else 1,
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
                    mode=self.mode,
                    category="transform",
                )
        else:
            logger.debug("no_detection_service_available", category="transform")

        # Limit cache_control blocks to comply with Anthropic's limit
        data = self._limit_cache_control_blocks(data)

        return json.dumps(data).encode("utf-8")
