"""Request transformation service for reverse proxy."""

import json
from typing import Any

from claude_code_proxy.utils.logging import get_logger


logger = get_logger(__name__)

# Claude Code identification prompt


# To see the request/response in mitmproxy:
# mitmproxy --mode reverse:https://api.anthropic.com@8082
# export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
# claude

# connection: keep-alive
# Accept: application/json
# X-Stainless-Retry-Count: 0
# X-Stainless-Timeout: 60
# X-Stainless-Lang: js
# X-Stainless-Package-Version: 0.55.1
# X-Stainless-OS: Linux
# X-Stainless-Arch: x64
# X-Stainless-Runtime: node
# X-Stainless-Runtime-Version: v22.14.0
# anthropic-dangerous-direct-browser-access: true
# anthropic-version: 2023-06-01
# authorization: Bearer sk-ant-
# x-app: cli
# User-Agent: claude-cli/1.0.43 (external, cli)
# content-type: application/json
# anthropic-beta: claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14
# accept-language: *
# sec-fetch-mode: cors
# accept-encoding: gzip, deflate

# Injection of prompt
# "system": [
#   {
#     "type": "text",
#     "text": "You are Claude Code, Anthropic's official CLI for Claude.",
#     "cache_control": {
#       "type": "ephemeral"
#     }
#   },

# Before
# Hello! I'm Claude, an AI assistant created by Anthropic. However,
# I should clarify that I'm not actually "Claude Code" or
# an official CLI tool - I'm the regular Claude AI assistant
# that you're chatting with through a text interface.
#
# After
# Hello! I'm Claude, Anthropic's AI assistant. I'm here to help you
# with a wide variety of tasks - from answering questions and explaining
# concepts to helping with analysis, writing, coding, math, and creative projects

claude_code_prompt = "You are Claude Code, Anthropic's official CLI for Claude."


def get_claude_code_prompt() -> dict[str, Any]:
    return {
        "type": "text",
        "text": claude_code_prompt,
        "cache_control": {"type": "ephemeral"},
    }


class RequestTransformer:
    """Handles request body and header transformations for reverse proxy."""

    def transform_system_prompt(self, body: bytes) -> bytes:
        """Transform system prompt to ensure Claude Code identification comes first.

        Args:
            body: Original request body as bytes

        Returns:
            Transformed request body as bytes
        """
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Return original if not valid JSON
            return body

        # Check if request has a system prompt
        if "system" not in data or (
            isinstance(data["system"], str) and data["system"] == claude_code_prompt
        ):
            # No system prompt, system prompt but is str, inject Claude Code identification
            data["system"] = [get_claude_code_prompt()]
            return json.dumps(data).encode("utf-8")

        system = data["system"]

        if isinstance(system, str):
            # Handle string system prompt
            if system == claude_code_prompt:
                return body

            data["system"] = [
                get_claude_code_prompt(),
                {"type": "text", "text": system},
            ]

        elif isinstance(system, list):
            # Handle array system prompt
            if len(system) > 0:
                # Check if first element has correct text
                first = system[0]
                if isinstance(first, dict) and first.get("text") == claude_code_prompt:
                    # Already has Claude Code first, return as-is
                    return body
            # Prepend Claude Code prompt
            data["system"] = [get_claude_code_prompt()] + system

        return json.dumps(data).encode("utf-8")

    def transform_path(self, path: str, mode: str = "full") -> str:
        """Transform request path by removing proxy prefixes and converting to Anthropic endpoints.

        Args:
            path: Original request path
            mode: Proxy mode - "minimal", "full", or "passthrough"

        Returns:
            Transformed path with proxy prefixes removed and endpoint conversion
        """
        # In minimal or passthrough mode, don't transform paths
        if mode in ("minimal", "passthrough"):
            return path

        # Full mode - current behavior
        # Remove /unclaude prefix if present
        if path.startswith("/unclaude/"):
            path = path[9:]  # Remove "/unclaude" (9 characters)

        # Remove /openai prefix if present
        elif path.startswith("/openai/"):
            path = path[7:]  # Remove "/openai" (7 characters)

        # Convert OpenAI endpoints to Anthropic endpoints
        if path.endswith("/chat/completions"):
            # OpenAI chat completions → Anthropic messages
            return "/v1/messages"

        # Ensure path starts with /
        if not path.startswith("/"):
            path = f"/{path}"

        return path

    def transform_request_body(
        self, body: bytes, path: str, mode: str = "full"
    ) -> bytes:
        """Apply all necessary transformations to the request body.

        Args:
            body: Original request body
            path: Request path for context
            mode: Proxy mode - "minimal", "full", or "passthrough"

        Returns:
            Transformed request body
        """
        # In passthrough mode, don't transform anything
        if mode == "passthrough":
            return body

        # In minimal mode, skip most transformations
        if mode == "minimal":
            # Don't do format conversion or system prompt injection
            return body

        # Full mode - current behavior
        # Check if this is an OpenAI format request
        is_openai_request = self._is_openai_request(path, body)

        if is_openai_request:
            # Transform OpenAI format to Anthropic format
            return self._transform_openai_to_anthropic(body)

        # Only transform messages endpoint for direct Anthropic requests
        if not path.endswith("/messages"):
            return body

        try:
            # Transform system prompt
            return self.transform_system_prompt(body)

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug(f"Failed to transform request body: {e}")
            return body

    def create_proxy_headers(
        self, original_headers: dict[str, str], access_token: str, mode: str = "full"
    ) -> dict[str, str]:
        """Create headers for the proxied request.

        Security: This function strips authentication headers (Authorization, x-api-key)
        from the original request and replaces them with OAuth authentication.
        This prevents client API keys from being leaked to the Anthropic API.

        Args:
            original_headers: Original request headers (may contain client API keys)
            access_token: OAuth access token for Anthropic API
            mode: Proxy mode - "minimal", "full", or "passthrough"

        Returns:
            Headers for the proxied request with OAuth auth and stripped client tokens
        """
        # Mode-specific header creation
        if mode == "minimal" or mode == "passthrough":
            # Minimal headers - only essentials
            headers = {
                "Authorization": f"Bearer {access_token}",
                "anthropic-version": "2023-06-01",  # Required
                "anthropic-beta": "oauth-2025-04-20",  # Required for OAuth
            }

            # Preserve essential headers
            content_type = self._get_header_case_insensitive(
                original_headers, "content-type"
            )
            headers["Content-Type"] = content_type or "application/json"

            accept = self._get_header_case_insensitive(original_headers, "accept")
            headers["Accept"] = accept or "application/json"

            # Preserve connection and cache control if present
            connection = self._get_header_case_insensitive(
                original_headers, "connection"
            )
            if connection:
                headers["Connection"] = connection

            cache_control = self._get_header_case_insensitive(
                original_headers, "cache-control"
            )
            if cache_control:
                headers["Cache-Control"] = cache_control

        else:
            # Full mode - all Claude CLI characteristics
            # Create fresh headers with OAuth authentication and Claude CLI characteristics
            # This approach naturally strips any client authentication headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
                "anthropic-version": "2023-06-01",
                "anthropic-dangerous-direct-browser-access": "true",
                "x-app": "cli",
                "User-Agent": "claude-cli/1.0.43 (external, cli)",
                "Connection": "keep-alive",
                "X-Stainless-Retry-Count": "0",
                "X-Stainless-Timeout": "60",
                "X-Stainless-Lang": "js",
                "X-Stainless-Package-Version": "0.55.1",
                "X-Stainless-OS": "Linux",
                "X-Stainless-Arch": "x64",
                "X-Stainless-Runtime": "node",
                "X-Stainless-Runtime-Version": "v22.14.0",
                "accept-language": "*",
                "sec-fetch-mode": "cors",
                "accept-encoding": "gzip, deflate",
            }

            # Preserve essential headers with defaults, but allow override
            content_type = self._get_header_case_insensitive(
                original_headers, "content-type"
            )
            headers["Content-Type"] = content_type or "application/json"

            accept = self._get_header_case_insensitive(original_headers, "accept")
            headers["Accept"] = accept or "application/json"

            # Override connection if specifically provided
            connection = self._get_header_case_insensitive(
                original_headers, "connection"
            )
            if connection:
                headers["Connection"] = connection

            # Preserve cache control for SSE
            cache_control = self._get_header_case_insensitive(
                original_headers, "cache-control"
            )
            if cache_control:
                headers["Cache-Control"] = cache_control

        return headers

    def _get_header_case_insensitive(
        self, headers: dict[str, str], key: str
    ) -> str | None:
        """Get header value with case-insensitive lookup.

        Args:
            headers: Headers dictionary
            key: Header key to look for

        Returns:
            Header value or None if not found
        """
        # Direct lookup
        if key in headers:
            return headers[key]

        # Case-insensitive lookup
        key_lower = key.lower()
        for k, v in headers.items():
            if k.lower() == key_lower:
                return v

        return None

    def _is_openai_request(self, path: str, body: bytes) -> bool:
        """Check if request is in OpenAI format based on path only.

        Args:
            path: Request path
            body: Request body (unused, kept for compatibility)

        Returns:
            True if this is an OpenAI format request based on path
        """
        # Only check path - all OpenAI requests should go to /unclaude/openai
        return path.startswith("/openai/") or path.endswith("/chat/completions")

    def _transform_openai_to_anthropic(self, body: bytes) -> bytes:
        """Transform OpenAI format request to Anthropic format.

        Args:
            body: OpenAI format request body

        Returns:
            Anthropic format request body
        """
        try:
            from claude_code_proxy.services.translator import OpenAITranslator

            openai_data = json.loads(body.decode("utf-8"))
            translator = OpenAITranslator()

            # Convert OpenAI request to Anthropic format
            anthropic_data = translator.openai_to_anthropic_request(openai_data)

            # Apply system prompt transformation
            anthropic_body = json.dumps(anthropic_data).encode("utf-8")
            return self.transform_system_prompt(anthropic_body)

        except Exception as e:
            logger.debug(f"Failed to transform OpenAI request to Anthropic: {e}")
            return body
