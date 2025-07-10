"""HTTP-level transformers for proxy service."""

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)

# Claude Code system prompt constants
claude_code_prompt = "You are Claude Code, Anthropic's official CLI for Claude."


def get_claude_code_prompt() -> dict[str, Any]:
    """Get the Claude Code system prompt with cache control."""
    return {
        "type": "text",
        "text": claude_code_prompt,
        "cache_control": {"type": "ephemeral"},
    }


class HTTPRequestTransformer:
    """Basic HTTP request transformer for proxy service."""

    def transform_path(self, path: str, proxy_mode: str = "full") -> str:
        """Transform request path."""
        # Basic path transformation - pass through for now
        return path

    def create_proxy_headers(
        self, headers: dict[str, str], access_token: str, proxy_mode: str = "full"
    ) -> dict[str, str]:
        """Create proxy headers from original headers with Claude CLI identity."""
        proxy_headers = {}

        # Copy important headers
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key not in ["host", "authorization"]:
                proxy_headers[key] = value

        # Add authentication
        proxy_headers["Authorization"] = f"Bearer {access_token}"

        # Set content type if not present
        if "content-type" not in proxy_headers:
            proxy_headers["Content-Type"] = "application/json"

        # Critical Claude/Anthropic headers for tools and beta features
        proxy_headers["anthropic-beta"] = (
            "claude-code-20250219,oauth-2025-04-20,"
            "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"
        )
        proxy_headers["anthropic-version"] = "2023-06-01"
        proxy_headers["anthropic-dangerous-direct-browser-access"] = "true"

        # Claude CLI identity headers
        proxy_headers["x-app"] = "cli"
        proxy_headers["User-Agent"] = "claude-cli/1.0.43 (external, cli)"

        # Stainless SDK compatibility headers
        proxy_headers["X-Stainless-Retry-Count"] = "0"
        proxy_headers["X-Stainless-Timeout"] = "60"
        proxy_headers["X-Stainless-Lang"] = "js"
        proxy_headers["X-Stainless-Package-Version"] = "0.55.1"
        proxy_headers["X-Stainless-OS"] = "Linux"
        proxy_headers["X-Stainless-Arch"] = "x64"
        proxy_headers["X-Stainless-Runtime"] = "node"
        proxy_headers["X-Stainless-Runtime-Version"] = "v22.14.0"

        # Standard HTTP headers for proper API interaction
        proxy_headers["Connection"] = "keep-alive"
        proxy_headers["accept-language"] = "*"
        proxy_headers["sec-fetch-mode"] = "cors"
        proxy_headers["accept-encoding"] = "gzip, deflate"

        return proxy_headers

    def transform_request_body(
        self, body: bytes, path: str, proxy_mode: str = "full"
    ) -> bytes:
        """Transform request body."""
        # Apply system prompt transformation for Claude Code identity
        return self.transform_system_prompt(body)

    def transform_system_prompt(self, body: bytes) -> bytes:
        """Transform system prompt to ensure Claude Code identification comes first.

        Args:
            body: Original request body as bytes

        Returns:
            Transformed request body as bytes with Claude Code system prompt
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
            # No system prompt, inject Claude Code identification
            data["system"] = [get_claude_code_prompt()]
            return json.dumps(data).encode("utf-8")

        system = data["system"]

        if isinstance(system, str):
            # Handle string system prompt
            if system == claude_code_prompt:
                # Already correct, convert to proper array format
                data["system"] = [get_claude_code_prompt()]
                return json.dumps(data).encode("utf-8")

            # Prepend Claude Code prompt to existing string
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
                    # Already has Claude Code first, ensure it has cache_control
                    data["system"][0] = get_claude_code_prompt()
                    return json.dumps(data).encode("utf-8")
            
            # Prepend Claude Code prompt
            data["system"] = [get_claude_code_prompt()] + system

        return json.dumps(data).encode("utf-8")


class HTTPResponseTransformer:
    """Basic HTTP response transformer for proxy service."""

    def transform_response_body(
        self, body: bytes, path: str, proxy_mode: str = "full"
    ) -> bytes:
        """Transform response body."""
        # Basic body transformation - pass through for now
        return body

    def transform_response_headers(
        self,
        headers: dict[str, str],
        path: str,
        content_length: int,
        proxy_mode: str = "full",
    ) -> dict[str, str]:
        """Transform response headers."""
        transformed_headers = {}

        # Copy important headers
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key not in ["content-length", "transfer-encoding"]:
                transformed_headers[key] = value

        # Set content length
        transformed_headers["Content-Length"] = str(content_length)

        # Add CORS headers
        transformed_headers["Access-Control-Allow-Origin"] = "*"
        transformed_headers["Access-Control-Allow-Headers"] = "*"
        transformed_headers["Access-Control-Allow-Methods"] = "*"

        return transformed_headers

    def _is_openai_request(self, path: str) -> bool:
        """Check if this is an OpenAI API request."""
        return "/openai/" in path or "/chat/completions" in path
