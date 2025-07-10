"""Claude SDK client wrapper for handling core Claude Code SDK interactions."""

from collections.abc import AsyncIterator
from typing import Any

from ccproxy.exceptions import ClaudeProxyError, ServiceUnavailableError
from ccproxy.utils.helper import patched_typing
from ccproxy.utils.logging import get_logger


with patched_typing():
    from claude_code_sdk import (
        AssistantMessage,
        ClaudeCodeOptions,
        CLIConnectionError,
        CLIJSONDecodeError,
        CLINotFoundError,
        ProcessError,
        ResultMessage,
        SystemMessage,
        UserMessage,
        query,
    )

logger = get_logger(__name__)


class ClaudeSDKError(Exception):
    """Base exception for Claude SDK errors."""


class ClaudeSDKConnectionError(ClaudeSDKError):
    """Raised when unable to connect to Claude Code."""


class ClaudeSDKProcessError(ClaudeSDKError):
    """Raised when Claude Code process fails."""


class ClaudeSDKClient:
    """
    Minimal Claude SDK client wrapper that handles core SDK interactions.

    This class provides a clean interface to the Claude Code SDK while handling
    error translation and basic query execution.
    """

    def __init__(self) -> None:
        """Initialize the Claude SDK client."""
        pass

    async def query_completion(
        self, prompt: str, options: ClaudeCodeOptions
    ) -> AsyncIterator[UserMessage | AssistantMessage | SystemMessage | ResultMessage]:
        """
        Execute a query using the Claude Code SDK.

        Args:
            prompt: The prompt string to send to Claude
            options: Claude Code options configuration

        Yields:
            Messages from the Claude Code SDK

        Raises:
            ClaudeSDKError: If the query fails
        """
        try:
            async for message in query(prompt=prompt, options=options):
                yield message
        except (CLINotFoundError, CLIConnectionError) as e:
            raise ServiceUnavailableError(f"Claude CLI not available: {str(e)}") from e
        except (ProcessError, CLIJSONDecodeError) as e:
            raise ClaudeProxyError(
                message=f"Claude process error: {str(e)}",
                error_type="service_unavailable_error",
                status_code=503,
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in query_completion: {e}")
            raise ClaudeProxyError(
                message=f"Unexpected error: {str(e)}",
                error_type="internal_server_error",
                status_code=500,
            ) from e

    async def validate_health(self) -> bool:
        """
        Validate that the Claude SDK is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check - the SDK is available if we can import it
            # More sophisticated checks could be added here
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        # Claude Code SDK doesn't require explicit cleanup
        pass

    async def __aenter__(self) -> "ClaudeSDKClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
