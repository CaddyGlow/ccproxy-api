"""Token estimation utilities for context window management."""

import json
from pathlib import Path
from typing import Any


# Cache for loaded token limits
_token_limits_cache: dict[str, int] | None = None


def estimate_tokens(content: Any) -> int:
    """Estimate token count for content.

    Uses ~3 characters per token heuristic for English text.
    This is a conservative estimate - actual may be lower.

    Args:
        content: Message content (string, list of blocks, or dict)

    Returns:
        Estimated token count
    """
    if content is None:
        return 0

    if isinstance(content, str):
        # ~3 chars per token for English, be conservative
        return max(1, len(content) // 3)

    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif block_type == "tool_use":
                    # Tool name + input
                    total += estimate_tokens(block.get("name", ""))
                    total += estimate_tokens(json.dumps(block.get("input", {})))
                elif block_type == "tool_result":
                    total += estimate_tokens(block.get("content", ""))
                elif block_type == "image":
                    # Images are ~1600 tokens for typical size
                    total += 1600
                else:
                    # Generic block - serialize and estimate
                    total += estimate_tokens(json.dumps(block))
            else:
                total += estimate_tokens(block)
        return total

    if isinstance(content, dict):
        return estimate_tokens(json.dumps(content))

    return estimate_tokens(str(content))


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of messages.

    Args:
        messages: List of message dicts with role and content

    Returns:
        Estimated total token count
    """
    total = 0
    for msg in messages:
        # Role contributes ~2 tokens
        total += 2
        total += estimate_tokens(msg.get("content"))
    return total


def estimate_request_tokens(request_data: dict[str, Any]) -> int:
    """Estimate total input tokens for a request.

    Includes messages, system prompt, and tool definitions.

    Args:
        request_data: The request payload dictionary

    Returns:
        Estimated total input token count
    """
    total = 0

    # Messages
    messages = request_data.get("messages", [])
    total += estimate_messages_tokens(messages)

    # System prompt
    system = request_data.get("system")
    if system:
        total += estimate_tokens(system)

    # Tools
    tools = request_data.get("tools", [])
    if tools:
        total += estimate_tokens(json.dumps(tools))

    return total


def _load_token_limits() -> dict[str, int]:
    """Load token limits from available sources.

    Loads from:
    1. Local token_limits.json in max_tokens plugin
    2. Pricing cache at ~/.cache/ccproxy/model_pricing.json

    Returns:
        Dict mapping model names to max_input_tokens
    """
    global _token_limits_cache
    if _token_limits_cache is not None:
        return _token_limits_cache

    _token_limits_cache = {}

    # Try local token_limits.json first
    local_file = (
        Path(__file__).parent.parent.parent
        / "plugins"
        / "max_tokens"
        / "token_limits.json"
    )
    if local_file.exists():
        try:
            with local_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for model_name, model_data in data.items():
                if model_name.startswith("_"):
                    continue
                if isinstance(model_data, dict):
                    max_input = model_data.get("max_input_tokens")
                    if isinstance(max_input, int):
                        _token_limits_cache[model_name] = max_input
        except Exception:
            pass  # Fall through to pricing cache

    # Also try pricing cache for additional models
    pricing_cache = Path.home() / ".cache" / "ccproxy" / "model_pricing.json"
    if pricing_cache.exists():
        try:
            with pricing_cache.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for model_name, model_data in data.items():
                if model_name in _token_limits_cache:
                    continue  # Local file takes precedence
                if isinstance(model_data, dict):
                    max_input = model_data.get("max_input_tokens")
                    if isinstance(max_input, int):
                        _token_limits_cache[model_name] = max_input
        except Exception:
            pass

    return _token_limits_cache


def get_max_input_tokens(model: str) -> int | None:
    """Get max input tokens for a model.

    Supports pattern matching for model variants:
    - Exact match: "claude-opus-4-5-20251101"
    - Prefix match: "claude-opus-4-5-*" matches "claude-opus-4-5-20251101"

    Args:
        model: Model name or identifier

    Returns:
        Max input tokens if known, None otherwise
    """
    limits = _load_token_limits()

    # Try exact match first
    if model in limits:
        return limits[model]

    # Try prefix matching (for patterns like claude-opus-4-5-*)
    for pattern, max_tokens in limits.items():
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            if model.startswith(prefix):
                return max_tokens

    # Try matching known model families
    model_lower = model.lower()
    for known_model, max_tokens in limits.items():
        if known_model.lower() in model_lower or model_lower in known_model.lower():
            return max_tokens

    return None
