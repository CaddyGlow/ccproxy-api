import json
import time
import uuid
from typing import Any, cast

import httpx
from starlette.responses import Response, StreamingResponse

from ccproxy.auth.exceptions import CredentialsInvalidError, OAuthTokenRefreshError
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.interfaces import (
    DetectionServiceProtocol,
    TokenManagerProtocol,
)
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.utils.headers import (
    extract_response_headers,
    filter_request_headers,
)

from .config import ClaudeAPISettings


logger = get_plugin_logger()


class ClaudeAPIAdapter(BaseHTTPAdapter):
    """Simplified Claude API adapter."""

    def __init__(
        self,
        detection_service: DetectionServiceProtocol,
        config: ClaudeAPISettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config or ClaudeAPISettings(), **kwargs)
        self.detection_service: DetectionServiceProtocol = detection_service
        self.token_manager: TokenManagerProtocol = cast(
            TokenManagerProtocol, self.auth_manager
        )

        self.base_url = self.config.base_url.rstrip("/")

    async def get_target_url(self, endpoint: str) -> str:
        return f"{self.base_url}/v1/messages"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        # Get a valid access token (auto-refreshes if expired)
        token_value = await self._resolve_access_token()

        # Parse body
        body_data = json.loads(body.decode()) if body else {}

        # Anthropic API rejects null temperature fields, so strip them early
        if body_data.get("temperature") is None:
            body_data.pop("temperature", None)

        # Anthropic API constraint: cannot accept both temperature and top_p
        # Prioritize temperature over top_p when both are present
        if "temperature" in body_data and "top_p" in body_data:
            body_data.pop("top_p", None)

        if self._needs_anthropic_conversion(endpoint):
            body_data = self._convert_openai_to_anthropic(body_data)

        # Inject system prompt based on config mode using detection service helper
        system_mode = self.config.system_prompt_injection_mode
        if self.detection_service and system_mode != "none":
            system_value = self._resolve_system_prompt_value(system_mode)
            if system_value is not None:
                body_data = self._inject_system_prompt(
                    body_data, system_value, mode=system_mode
                )

        # Limit cache_control blocks to comply with Anthropic's limit
        body_data = self._limit_cache_control_blocks(body_data)

        # Remove internal metadata fields like _ccproxy_injected before sending to the API
        body_data = self._remove_metadata_fields(body_data)

        # Filter headers and enforce OAuth Authorization
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        # Always set Authorization from OAuth-managed access token
        filtered_headers["authorization"] = f"Bearer {token_value}"

        # Add CLI headers if available, but never allow overriding auth
        cli_headers = self._collect_cli_headers()
        if cli_headers:
            blocked_overrides = {"authorization", "x-api-key"}
            for key, value in cli_headers.items():
                lk = key.lower()
                if lk in blocked_overrides:
                    logger.debug(
                        "cli_header_override_blocked",
                        header=lk,
                        reason="preserve_oauth_auth_header",
                    )
                    continue
                filtered_headers[lk] = value

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

        body_bytes = response.content
        media_type = response.headers.get("content-type")

        if self._needs_openai_conversion(endpoint):
            converted = self._convert_anthropic_to_openai_response(response)
            if converted is not None:
                body_bytes = json.dumps(converted).encode()
                media_type = "application/json"

        return Response(
            content=body_bytes,
            status_code=response.status_code,
            headers=response_headers,
            media_type=media_type,
        )

    def _needs_openai_conversion(self, endpoint: str) -> bool:
        if not getattr(self.config, "support_openai_format", True):
            return False
        normalized = (endpoint or "").strip().lower()
        return normalized.startswith("/v1/chat/completions")

    def _needs_anthropic_conversion(self, endpoint: str) -> bool:
        if not getattr(self.config, "support_openai_format", True):
            return False
        normalized = (endpoint or "").strip().lower()
        return normalized.startswith("/v1/chat/completions")

    async def _resolve_access_token(self) -> str:
        """Resolve a usable Claude API OAuth token from the token manager.

        If the auth manager is not configured, raise a unified AuthenticationError
        so middleware returns a clean 401 without stack traces.
        """

        if not getattr(self, "token_manager", None):
            from ccproxy.core.errors import AuthenticationError

            logger.warning(
                "auth_manager_override_not_resolved",
                plugin="claude_api",
                auth_manager_name="oauth_claude",
                category="auth",
            )
            raise AuthenticationError(
                "Authentication manager not configured for Claude API provider"
            )

        token_manager = self.token_manager

        async def _snapshot_token() -> str | None:
            snapshot = await token_manager.get_token_snapshot()
            if snapshot and snapshot.access_token:
                return str(snapshot.access_token)
            return None

        credentials = await token_manager.load_credentials()
        if credentials and token_manager.should_refresh(credentials):
            try:
                refreshed = await token_manager.get_access_token_with_refresh()
                if refreshed:
                    return refreshed
            except OAuthTokenRefreshError as exc:
                logger.warning(
                    "claude_token_refresh_failed",
                    error=str(exc),
                    category="auth",
                )
                fallback = await _snapshot_token()
                if fallback:
                    return fallback

        # Primary path: rely on manager contract
        try:
            token = await token_manager.get_access_token()
            if token:
                return token
        except CredentialsInvalidError:
            logger.debug("claude_token_invalid", category="auth")
        except OAuthTokenRefreshError as exc:
            logger.warning(
                "claude_token_refresh_failed",
                error=str(exc),
                category="auth",
            )
            fallback = await _snapshot_token()
            if fallback:
                return fallback
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "claude_token_fetch_failed",
                error=str(exc),
                category="auth",
            )

        # Fallback to explicit refresh helper
        try:
            refreshed = await token_manager.get_access_token_with_refresh()
            if refreshed:
                return refreshed
        except OAuthTokenRefreshError as exc:
            logger.warning(
                "claude_token_refresh_failed",
                error=str(exc),
                category="auth",
            )
            fallback = await _snapshot_token()
            if fallback:
                return fallback
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "claude_token_refresh_failed",
                error=str(exc),
                category="auth",
            )

        fallback = await _snapshot_token()
        if fallback:
            return fallback

        raise ValueError("No valid OAuth access token available for Claude API")

    def _resolve_system_prompt_value(self, mode: str) -> Any:
        """Retrieve system prompt content for injection from detection cache."""

        if not self.detection_service:
            return None

        # Primary path: detection service helper
        try:
            prompt = self.detection_service.get_system_prompt(mode=mode)
        except Exception:
            prompt = {}
        if isinstance(prompt, dict):
            system_value = prompt.get("system")
            if system_value:
                return system_value

        prompts = self.detection_service.get_detected_prompts()
        if prompts.has_system():
            system_payload = prompts.system_payload(mode=mode)
            system_value = system_payload.get("system") if system_payload else None
            if system_value:
                return system_value
            return prompts.system

        cached = self.detection_service.get_cached_data()
        # Backward compatibility: legacy cached.system_prompt object
        system_prompt_obj = getattr(cached, "system_prompt", None) if cached else None
        if system_prompt_obj is not None:
            return getattr(system_prompt_obj, "system_field", system_prompt_obj)

        return None

    def _collect_cli_headers(self) -> dict[str, str]:
        """Collect safe CLI headers from detection cache for request forwarding."""

        if not self.detection_service:
            return {}

        headers_data = self.detection_service.get_detected_headers()
        if not headers_data:
            return {}

        ignores = {h.lower() for h in self.detection_service.get_ignored_headers()}
        redacted = {h.lower() for h in self.detection_service.get_redacted_headers()}

        return headers_data.filtered(ignores=ignores, redacted=redacted)

    def _convert_openai_to_anthropic(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenAI chat.completions style payload to Anthropic format."""

        if not isinstance(payload, dict):
            return payload

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        system_blocks: list[Any] = []
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                block = self._normalize_text_block(content)
                if block is not None:
                    if isinstance(block, list):
                        system_blocks.extend(block)
                    else:
                        system_blocks.append(block)
                continue

            block = self._normalize_text_block(content)
            if block is None:
                block = []

            anthropic_messages.append(
                {
                    "role": role or "user",
                    "content": block if isinstance(block, list) else [block],
                }
            )

        converted = {k: v for k, v in payload.items() if k != "messages"}
        if system_blocks:
            converted["system"] = system_blocks
        converted["messages"] = anthropic_messages
        return converted

    def _normalize_text_block(self, content: Any) -> Any:
        if isinstance(content, list):
            blocks = []
            for part in content:
                if isinstance(part, dict):
                    blocks.append(part)
                elif isinstance(part, str):
                    blocks.append({"type": "text", "text": part})
            return blocks
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            return {"type": "text", "text": content}
        return None

    def _convert_anthropic_to_openai_response(
        self, response: httpx.Response
    ) -> dict[str, Any] | None:
        try:
            payload = json.loads(response.content.decode()) if response.content else {}
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        message = {
            "role": "assistant",
            "content": "",
        }

        content_blocks = payload.get("content")
        if isinstance(content_blocks, list):
            texts = [
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            message["content"] = "".join(texts)
        elif isinstance(content_blocks, str):
            message["content"] = content_blocks

        finish_reason = payload.get("stop_reason") or payload.get("stop_sequence")
        finish_reason_map = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
        }
        if isinstance(finish_reason, str):
            mapped_finish_reason = finish_reason_map.get(finish_reason, "stop")
        else:
            mapped_finish_reason = "stop"

        usage = payload.get("usage")
        usage_converted = None
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("input_tokens") or 0)
            completion_tokens = int(usage.get("output_tokens") or 0)
            total_tokens = int(
                usage.get("total_tokens") or prompt_tokens + completion_tokens
            )
            usage_converted = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        model_value = payload.get("model")
        if not isinstance(model_value, str) or not model_value:
            mappings = getattr(self.config, "model_mappings", None) or []
            if mappings:
                model_value = mappings[0].target
            else:
                model_value = ""

        converted = {
            "id": payload.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_value,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": mapped_finish_reason,
                    "logprobs": None,
                }
            ],
        }

        if usage_converted is not None:
            converted["usage"] = usage_converted

        return converted

    # Helper methods (move from transformers)
    def _inject_system_prompt(
        self, body_data: dict[str, Any], system_prompt: Any, mode: str = "full"
    ) -> dict[str, Any]:
        """Inject system prompt from Claude CLI detection.

        Args:
            body_data: The request body data dict
            system_prompt: System prompt data from detection service
            mode: Injection mode - "full" (all prompts), "minimal" (first prompt only), or "none"

        Returns:
            Modified body data with system prompt injected
        """
        if not system_prompt:
            return body_data

        # Get the system field from the system prompt data
        system_field = (
            system_prompt.system_field
            if hasattr(system_prompt, "system_field")
            else system_prompt
        )

        if not system_field:
            return body_data

        # Apply injection mode filtering
        if mode == "minimal":
            # Only inject the first system prompt block
            if isinstance(system_field, list) and len(system_field) > 0:
                system_field = [system_field[0]]
            # If it's a string, keep as-is (already minimal)
        elif mode == "none":
            # Should not reach here due to earlier check, but handle gracefully
            return body_data
        # For "full" mode, use system_field as-is

        # Mark the detected system prompt as injected for preservation
        marked_system = self._mark_injected_system_prompts(system_field)

        existing_system = body_data.get("system")

        if existing_system is None:
            # No existing system prompt, inject the marked detected one
            body_data["system"] = marked_system
        else:
            # Request has existing system prompt, prepend the marked detected one
            if isinstance(marked_system, list):
                if isinstance(existing_system, str):
                    # Detected is marked list, existing is string
                    body_data["system"] = marked_system + [
                        {"type": "text", "text": existing_system}
                    ]
                elif isinstance(existing_system, list):
                    # Both are lists, concatenate (detected first)
                    body_data["system"] = marked_system + existing_system
            else:
                # Convert both to list format for consistency
                if isinstance(existing_system, str):
                    body_data["system"] = [
                        {
                            "type": "text",
                            "text": str(marked_system),
                            "_ccproxy_injected": True,
                        },
                        {"type": "text", "text": existing_system},
                    ]
                elif isinstance(existing_system, list):
                    body_data["system"] = [
                        {
                            "type": "text",
                            "text": str(marked_system),
                            "_ccproxy_injected": True,
                        }
                    ] + existing_system

        return body_data

    def _mark_injected_system_prompts(self, system_data: Any) -> Any:
        """Mark system prompts as injected by ccproxy for preservation.

        Args:
            system_data: System prompt data to mark

        Returns:
            System data with injected blocks marked with _ccproxy_injected metadata
        """
        if isinstance(system_data, str):
            # String format - convert to list with marking
            return [{"type": "text", "text": system_data, "_ccproxy_injected": True}]
        elif isinstance(system_data, list):
            # List format - mark each block as injected
            marked_data = []
            for block in system_data:
                if isinstance(block, dict):
                    # Copy block and add marking
                    marked_block = block.copy()
                    marked_block["_ccproxy_injected"] = True
                    marked_data.append(marked_block)
                else:
                    # Preserve non-dict blocks as-is
                    marked_data.append(block)
            return marked_data

        return system_data

    def _remove_metadata_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove internal ccproxy metadata from request data before sending to API.

        This method removes:
        - Fields starting with '_' (internal metadata like _ccproxy_injected)
        - Any other internal ccproxy metadata that shouldn't be sent to the API

        Args:
            data: Request data dictionary

        Returns:
            Cleaned data dictionary without internal metadata
        """
        import copy

        # Deep copy to avoid modifying original
        clean_data = copy.deepcopy(data)

        # Clean system field
        system = clean_data.get("system")
        if isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and "_ccproxy_injected" in block:
                    del block["_ccproxy_injected"]

        # Clean messages
        messages = clean_data.get("messages", [])
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "_ccproxy_injected" in block:
                        del block["_ccproxy_injected"]

        # Clean tools (though they shouldn't have _ccproxy_injected, but be safe)
        tools = clean_data.get("tools", [])
        for tool in tools:
            if isinstance(tool, dict) and "_ccproxy_injected" in tool:
                del tool["_ccproxy_injected"]

        return clean_data

    def _find_cache_control_blocks(
        self, data: dict[str, Any]
    ) -> list[tuple[str, int, int]]:
        """Find all cache_control blocks in the request with their locations.

        Returns:
            List of tuples (location_type, location_index, block_index) for each cache_control block
            where location_type is 'system', 'message', 'tool', 'tool_use', or 'tool_result'
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
                        block_type = block.get("type")
                        if block_type == "tool_use":
                            blocks.append(("tool_use", msg_idx, block_idx))
                        elif block_type == "tool_result":
                            blocks.append(("tool_result", msg_idx, block_idx))
                        else:
                            blocks.append(("message", msg_idx, block_idx))

        # Find in tools
        tools = data.get("tools", [])
        for tool_idx, tool in enumerate(tools):
            if isinstance(tool, dict) and "cache_control" in tool:
                blocks.append(("tool", tool_idx, 0))

        return blocks

    def _calculate_content_size(self, data: dict[str, Any]) -> int:
        """Calculate the approximate content size of a block for cache prioritization.

        Args:
            data: Block data dictionary

        Returns:
            Approximate size in characters
        """
        size = 0

        # Count text content
        if "text" in data:
            size += len(str(data["text"]))

        # Count tool use content
        if "name" in data:  # Tool use block
            size += len(str(data["name"]))
        if "input" in data:
            size += len(str(data["input"]))

        # Count tool result content
        if "content" in data and isinstance(data["content"], str | list):
            if isinstance(data["content"], str):
                size += len(data["content"])
            else:
                # Nested content - recursively calculate
                for sub_item in data["content"]:
                    if isinstance(sub_item, dict):
                        size += self._calculate_content_size(sub_item)
                    else:
                        size += len(str(sub_item))

        # Count other string fields
        for key, value in data.items():
            if key not in (
                "text",
                "name",
                "input",
                "content",
                "cache_control",
                "_ccproxy_injected",
                "type",
            ):
                size += len(str(value))

        return size

    def _get_block_at_location(
        self,
        data: dict[str, Any],
        location_type: str,
        location_index: int,
        block_index: int,
    ) -> dict[str, Any] | None:
        """Get the block at a specific location in the data structure.

        Returns:
            Block dictionary or None if not found
        """
        if location_type == "system":
            system = data.get("system")
            if isinstance(system, list) and block_index < len(system):
                block = system[block_index]
                return block if isinstance(block, dict) else None
        elif location_type in ("message", "tool_use", "tool_result"):
            messages = data.get("messages", [])
            if location_index < len(messages):
                content = messages[location_index].get("content")
                if isinstance(content, list) and block_index < len(content):
                    block = content[block_index]
                    return block if isinstance(block, dict) else None
        elif location_type == "tool":
            tools = data.get("tools", [])
            if location_index < len(tools):
                tool = tools[location_index]
                return tool if isinstance(tool, dict) else None

        return None

    def _remove_cache_control_at_location(
        self,
        data: dict[str, Any],
        location_type: str,
        location_index: int,
        block_index: int,
    ) -> bool:
        """Remove cache_control from a block at a specific location.

        Returns:
            True if cache_control was successfully removed, False otherwise
        """
        block = self._get_block_at_location(
            data, location_type, location_index, block_index
        )
        if block and isinstance(block, dict) and "cache_control" in block:
            del block["cache_control"]
            return True
        return False

    def _limit_cache_control_blocks(
        self, data: dict[str, Any], max_blocks: int = 4
    ) -> dict[str, Any]:
        """Limit the number of cache_control blocks using smart algorithm.

        Smart algorithm:
        1. Preserve all injected system prompts (marked with _ccproxy_injected)
        2. Keep the 2 largest remaining blocks by content size
        3. Remove cache_control from smaller blocks when exceeding the limit

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

        logger.warning(
            "cache_control_limit_exceeded",
            total_blocks=total_blocks,
            max_blocks=max_blocks,
            category="transform",
        )

        # Classify blocks as injected vs non-injected and calculate sizes
        injected_blocks = []
        non_injected_blocks = []

        for location in cache_blocks:
            location_type, location_index, block_index = location
            block = self._get_block_at_location(
                data, location_type, location_index, block_index
            )

            if block and isinstance(block, dict):
                if block.get("_ccproxy_injected", False):
                    injected_blocks.append(location)
                    logger.debug(
                        "found_injected_block",
                        location_type=location_type,
                        location_index=location_index,
                        block_index=block_index,
                        category="transform",
                    )
                else:
                    # Calculate content size for prioritization
                    content_size = self._calculate_content_size(block)
                    non_injected_blocks.append((location, content_size))

        # Sort non-injected blocks by size (largest first)
        non_injected_blocks.sort(key=lambda x: x[1], reverse=True)

        # Determine how many non-injected blocks we can keep
        injected_count = len(injected_blocks)
        remaining_slots = max_blocks - injected_count

        logger.info(
            "cache_control_smart_limiting",
            total_blocks=total_blocks,
            injected_blocks=injected_count,
            non_injected_blocks=len(non_injected_blocks),
            remaining_slots=remaining_slots,
            max_blocks=max_blocks,
            category="transform",
        )

        # Keep the largest non-injected blocks up to remaining slots
        blocks_to_keep = set(injected_blocks)  # Always keep injected blocks
        if remaining_slots > 0:
            largest_blocks = non_injected_blocks[:remaining_slots]
            blocks_to_keep.update(location for location, size in largest_blocks)

            logger.debug(
                "keeping_largest_blocks",
                kept_blocks=[(loc, size) for loc, size in largest_blocks],
                category="transform",
            )

        # Remove cache_control from blocks not in the keep set
        blocks_to_remove = [loc for loc in cache_blocks if loc not in blocks_to_keep]

        for location_type, location_index, block_index in blocks_to_remove:
            if self._remove_cache_control_at_location(
                data, location_type, location_index, block_index
            ):
                logger.debug(
                    "removed_cache_control_smart",
                    location=location_type,
                    location_index=location_index,
                    block_index=block_index,
                    category="transform",
                )

        logger.info(
            "cache_control_limiting_complete",
            blocks_removed=len(blocks_to_remove),
            blocks_kept=len(blocks_to_keep),
            injected_preserved=injected_count,
            category="transform",
        )

        return data
