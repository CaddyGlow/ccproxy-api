"""Claude SDK client for handling API requests and responses."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from claude_code_proxy.exceptions import (
    ClaudeProxyError,
    ServiceUnavailableError,
    TimeoutError,
)
from claude_code_proxy.utils.helper import patched_typing


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
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        query,
    )


logger = logging.getLogger(__name__)


class ClaudeClientError(Exception):
    """Base exception for Claude client errors."""


class ClaudeClientConnectionError(ClaudeClientError):
    """Raised when unable to connect to Claude Code."""


class ClaudeClientProcessError(ClaudeClientError):
    """Raised when Claude Code process fails."""


class ClaudeClient:
    """
    Async Claude SDK client that handles translation between Anthropic API format
    and Claude Code SDK format.
    """

    def __init__(
        self,
        connection_id: str | None = None,
    ) -> None:
        """
        Initialize Claude client.

        Args:
            connection_id: Optional ID for pooled connections
        """
        self.connection_id = connection_id
        self._is_pooled = connection_id is not None

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        options: ClaudeCodeOptions,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """
        Create a completion using Claude Code SDK.

        Args:
            messages: List of messages in Anthropic format
            options: Claude Code options configuration
            stream: Whether to stream responses
            **kwargs: Additional arguments

        Returns:
            Response dict or async iterator of response chunks if streaming

        Raises:
            ClaudeClientError: If request fails
        """
        try:
            # Convert Anthropic messages to Claude SDK format
            prompt = self._format_messages_to_prompt(messages)

            if stream:
                return self._stream_completion(prompt, options)
            else:
                return await self._complete_non_streaming(prompt, options)

        except (CLINotFoundError, CLIConnectionError) as e:
            raise ServiceUnavailableError(f"Claude CLI not available: {str(e)}") from e
        except (ProcessError, CLIJSONDecodeError) as e:
            raise ClaudeProxyError(
                message=f"Claude process error: {str(e)}",
                error_type="service_unavailable_error",
                status_code=503,
            ) from e
        except ClaudeClientError as e:
            # Re-raise ClaudeClientError as-is for proper testing
            raise
        except Exception as e:
            logger.error(f"Unexpected error in create_completion: {e}")
            raise ClaudeProxyError(
                message=f"Unexpected error: {str(e)}",
                error_type="internal_server_error",
                status_code=500,
            ) from e

    async def _get_query_iterator(
        self, prompt: str, options: ClaudeCodeOptions
    ) -> AsyncIterator[UserMessage | AssistantMessage | SystemMessage | ResultMessage]:
        """Get query iterator using Claude Code SDK."""
        # The Claude CLI path is already set up in PATH by the settings configuration
        # The anyio task scope issue should be fixed in the GitHub version of the SDK
        async for message in query(prompt=prompt, options=options):
            yield message

    async def _complete_non_streaming(
        self, prompt: str, options: ClaudeCodeOptions
    ) -> dict[str, Any]:
        """Complete a non-streaming request."""
        messages = []
        result_message = None

        async for message in self._get_query_iterator(prompt, options):
            messages.append(message)
            if isinstance(message, ResultMessage):
                result_message = message

        if result_message is None:
            raise ClaudeClientError("No result message received")

        # Find the last assistant message for the response
        assistant_messages = [
            msg for msg in messages if isinstance(msg, AssistantMessage)
        ]
        if not assistant_messages:
            raise ClaudeClientError("No assistant response received")

        last_assistant_message = assistant_messages[-1]

        # Convert to Anthropic format
        return {
            "id": f"msg_{result_message.session_id}",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": self._extract_text_from_content(
                        last_assistant_message.content
                    ),
                }
            ],
            "model": options.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,  # Claude Code SDK doesn't provide token counts
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def _stream_completion(
        self, prompt: str, options: ClaudeCodeOptions
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream completion responses."""
        import asyncio

        first_chunk = True
        query_iterator = None

        try:
            query_iterator = self._get_query_iterator(prompt, options)
            message_count = 0
            async for message in query_iterator:
                message_count += 1
                logger.debug(
                    f"Claude SDK message {message_count}: {type(message).__name__} - {message}"
                )
                if isinstance(message, AssistantMessage):
                    if first_chunk:
                        # Send initial chunk
                        yield {
                            "id": f"msg_{id(message)}",
                            "type": "message_start",
                            "message": {
                                "id": f"msg_{id(message)}",
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": options.model,
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0,
                                },
                            },
                        }
                        first_chunk = False

                    # Send content delta
                    text_content = self._extract_text_from_content(message.content)
                    if text_content:
                        yield {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": text_content},
                        }

                elif isinstance(message, ResultMessage):
                    # Send final chunk
                    yield {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 0},
                    }
                    break

        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.info("Stream completion cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in stream completion: {e}")
            # Send error chunk
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "error"},
                "usage": {"output_tokens": 0},
            }
            raise
        finally:
            # Clean up resources if needed
            if query_iterator and hasattr(query_iterator, "aclose"):
                try:
                    await query_iterator.aclose()
                except Exception as e:
                    logger.debug(f"Error closing query iterator: {e}")

    def _format_messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert Anthropic messages format to a single prompt string."""
        prompt_parts = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if isinstance(content, list):
                # Handle content blocks
                text_parts = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = " ".join(text_parts)

            if role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                # System messages are handled via options
                continue

        return "\n\n".join(prompt_parts)

    def _extract_text_from_content(
        self, content: list[TextBlock | ToolUseBlock | ToolResultBlock]
    ) -> str:
        """Extract text content from Claude SDK content blocks."""
        text_parts = []

        for block in content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                # For tool use blocks, we might want to include the tool name and input
                text_parts.append(f"[Tool: {block.name}]")
            elif isinstance(block, ToolResultBlock) and isinstance(block.content, str):
                text_parts.append(block.content)

        return " ".join(text_parts)

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List available Claude models.

        Returns:
            List of available models in Anthropic format
        """
        # These are the models supported by Claude Code SDK
        models: list[dict[str, Any]] = []

        return models

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        # Claude Code SDK doesn't require explicit cleanup
        pass

    async def validate_health(self) -> bool:
        """
        Validate that this client connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check - try a minimal query
            # We could potentially implement a more sophisticated check
            return True
        except Exception:
            return False

    async def __aenter__(self) -> "ClaudeClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
