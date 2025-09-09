"""Claude API request transformer."""

import json
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.utils.headers import filter_request_headers

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

        # Use common filter utility (don't preserve auth since we'll add our own)
        transformed = filter_request_headers(headers, preserve_auth=False)

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

    def _mark_injected_system_prompts(
        self, system_data: str | list[dict[str, Any]]
    ) -> str | list[dict[str, Any]]:
        """Mark system prompts as injected by ccproxy for preservation during cache limiting.

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

    def _clean_internal_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove internal ccproxy metadata from request data before sending to API.

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

        logger = get_plugin_logger()
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
                detected_system: str | list[dict[str, Any]] | None = None
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
                    # Mark the detected system prompt as injected
                    marked_detected_system = self._mark_injected_system_prompts(
                        detected_system
                    )
                    existing_system = data.get("system")

                    if existing_system is None:
                        # No existing system prompt, inject the marked detected one
                        data["system"] = marked_detected_system
                        logger.debug(
                            "injected_system_prompt_new",
                            version=cached_data.claude_version,
                            mode=self.mode,
                            category="transform",
                        )
                    else:
                        # Request has existing system prompt, prepend the marked detected one
                        if isinstance(marked_detected_system, list):
                            if isinstance(existing_system, str):
                                # Detected is marked list, existing is string
                                data["system"] = marked_detected_system + [
                                    {"type": "text", "text": existing_system}
                                ]
                            elif isinstance(existing_system, list):
                                # Both are lists, concatenate (detected first)
                                data["system"] = (
                                    marked_detected_system + existing_system
                                )
                        else:
                            # This shouldn't happen since marking always returns a list, but handle it
                            if isinstance(existing_system, str):
                                # Convert both to list format
                                data["system"] = [
                                    {
                                        "type": "text",
                                        "text": str(marked_detected_system),
                                        "_ccproxy_injected": True,
                                    },
                                    {"type": "text", "text": existing_system},
                                ]
                            elif isinstance(existing_system, list):
                                # Detected is not list, existing is list
                                data["system"] = [
                                    {
                                        "type": "text",
                                        "text": str(marked_detected_system),
                                        "_ccproxy_injected": True,
                                    }
                                ] + existing_system

                        logger.debug(
                            "injected_system_prompt_prepended",
                            version=cached_data.claude_version,
                            mode=self.mode,
                            existing_system_type=type(existing_system).__name__,
                            detected_system_type=type(marked_detected_system).__name__,
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

        # Clean internal metadata before sending to API
        data = self._clean_internal_metadata(data)

        return json.dumps(data).encode("utf-8")
