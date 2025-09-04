from typing import Any

from ccproxy.config.constants import (
    ANTHROPIC_MESSAGES_PATH,
    CODEX_RESPONSES_ENDPOINT,
    OPENAI_CHAT_COMPLETIONS_PATH,
)


class FormatDetectionService:
    """Service for detecting API format from requests and endpoints."""

    def detect_request_format(self, request_data: dict[str, Any]) -> str:
        """Detect format from request structure with fail-fast validation."""
        if not isinstance(request_data, dict):
            raise ValueError(f"Invalid request data type: {type(request_data)}")

        # Anthropic: has required 'max_tokens' and 'messages'
        if "max_tokens" in request_data and "messages" in request_data:
            return "anthropic"

        # OpenAI: has 'messages' but uses 'max_completion_tokens' or optional 'max_tokens'
        if "messages" in request_data:
            if (
                "max_completion_tokens" in request_data
                or "max_tokens" not in request_data
            ):
                return "openai"

        # Response API: has 'input' or other Response API specific fields
        if "input" in request_data or "output" in request_data:
            return "response_api"

        raise ValueError(
            f"Unable to detect format from request data keys: {list(request_data.keys())}"
        )

    def get_format_from_endpoint(self, endpoint: str) -> str:
        """Determine expected format from endpoint path with fail-fast validation."""
        if not endpoint:
            raise ValueError("Endpoint cannot be empty")

        # Handle exact matches first
        if endpoint.endswith(ANTHROPIC_MESSAGES_PATH):
            return "anthropic"
        elif endpoint.endswith(OPENAI_CHAT_COMPLETIONS_PATH):
            return "openai"
        elif CODEX_RESPONSES_ENDPOINT in endpoint:
            return "response_api"
        
        # Handle session-based endpoints (e.g., /{session_id}/v1/messages, /{session_id}/chat/completions)
        # Check if the endpoint contains the path patterns after session parameters
        if "/v1/messages" in endpoint:
            return "anthropic"
        elif "/chat/completions" in endpoint or "/v1/chat/completions" in endpoint:
            return "openai"
        elif "/responses" in endpoint:
            return "response_api"

        raise ValueError(f"Unable to detect format from endpoint: {endpoint}")

    def is_streaming_request(self, request_data: dict[str, Any]) -> bool:
        """Check if request is configured for streaming."""
        return request_data.get("stream", False) is True
