"""Claude SDK client for handling API requests and responses."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    InternalClient,
    ProcessError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from claude_code_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from claude_proxy.exceptions import (
    ClaudeProxyError,
    ServiceUnavailableError,
    TimeoutError,
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
        *,
        api_key: str | None = None,
        default_model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        claude_cli_path: str | None = None,
    ) -> None:
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (optional, can be set via environment)
            default_model: Default Claude model to use
            max_tokens: Maximum tokens for responses
            temperature: Temperature for response generation
            system_prompt: Default system prompt
            claude_cli_path: Path to Claude CLI executable
        """
        self.api_key = api_key
        self.default_model = default_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.claude_cli_path = claude_cli_path

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """
        Create a completion using Claude Code SDK.

        Args:
            messages: List of messages in Anthropic format
            model: Model to use (overrides default)
            max_tokens: Maximum tokens (overrides default)
            temperature: Temperature (overrides default)
            system: System prompt (overrides default)
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

            # Create options
            options = self._create_options(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                **kwargs,
            )

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
                status_code=503
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in create_completion: {e}")
            raise ClaudeProxyError(
                message=f"Unexpected error: {str(e)}",
                error_type="internal_server_error",
                status_code=500
            ) from e

    async def _get_query_iterator(
        self, prompt: str, options: ClaudeCodeOptions
    ) -> AsyncIterator[UserMessage | AssistantMessage | SystemMessage | ResultMessage]:
        """Get query iterator with custom CLI path if specified."""
        if self.claude_cli_path:
            # Use custom transport with specified CLI path
            transport = SubprocessCLITransport(
                prompt=prompt, options=options, cli_path=self.claude_cli_path
            )
            client = InternalClient()
            async for message in client.process_query(prompt=prompt, options=options):
                yield message
        else:
            # Use default query method
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
            "model": options.model or self.default_model,
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
        first_chunk = True

        async for message in self._get_query_iterator(prompt, options):
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
                            "model": options.model or self.default_model,
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

    def _create_options(
        self,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> ClaudeCodeOptions:
        """Create Claude Code SDK options from parameters."""
        effective_system = system or self.system_prompt

        return ClaudeCodeOptions(
            model=model or self.default_model,
            system_prompt=effective_system,
            max_turns=kwargs.get("max_turns", 1),
            permission_mode=kwargs.get("permission_mode", "default"),
            allowed_tools=kwargs.get("allowed_tools", []),
            disallowed_tools=kwargs.get("disallowed_tools", []),
        )

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
        models = [
            {
                "id": "claude-3-opus-20240229",
                "object": "model",
                "created": 1677610602,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-sonnet-20240229",
                "object": "model",
                "created": 1677610602,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-haiku-20240307",
                "object": "model",
                "created": 1677610602,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "object": "model",
                "created": 1677610602,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "object": "model",
                "created": 1677610602,
                "owned_by": "anthropic",
            },
        ]

        return models

    async def close(self) -> None:
        """Close the client and cleanup resources."""
        # Claude Code SDK doesn't require explicit cleanup
        pass

    async def __aenter__(self) -> "ClaudeClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
