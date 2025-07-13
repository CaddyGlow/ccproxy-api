"""Claude SDK service orchestration for business logic."""

from collections.abc import AsyncIterator
from typing import Any

from ccproxy.auth.manager import AuthManager
from ccproxy.claude_sdk.client import ClaudeSDKClient
from ccproxy.claude_sdk.converter import MessageConverter
from ccproxy.claude_sdk.options import OptionsHandler
from ccproxy.core.errors import (
    ClaudeProxyError,
    ServiceUnavailableError,
)
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class ClaudeSDKService:
    """
    Service layer for Claude SDK operations orchestration.

    This class handles business logic coordination between the pure SDK client,
    authentication, metrics, and format conversion while maintaining clean
    separation of concerns.
    """

    def __init__(
        self,
        sdk_client: ClaudeSDKClient | None = None,
        auth_manager: AuthManager | None = None,
    ) -> None:
        """
        Initialize Claude SDK service.

        Args:
            sdk_client: Claude SDK client instance
            auth_manager: Authentication manager (optional)
        """
        self.sdk_client = sdk_client or ClaudeSDKClient()
        self.auth_manager = auth_manager
        self.message_converter = MessageConverter()
        self.options_handler = OptionsHandler()

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """
        Create a completion using Claude SDK with business logic orchestration.

        Args:
            messages: List of messages in Anthropic format
            model: The model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            stream: Whether to stream responses
            user_id: User identifier for auth/metrics
            **kwargs: Additional arguments

        Returns:
            Response dict or async iterator of response chunks if streaming

        Raises:
            ClaudeProxyError: If request fails
            ServiceUnavailableError: If service is unavailable
        """
        # Validate authentication if auth manager is configured
        if self.auth_manager and user_id:
            try:
                await self._validate_user_auth(user_id)
            except Exception as e:
                logger.error(f"Authentication failed for user {user_id}: {e}")
                raise

        # Extract system message and create options
        system_message = self.options_handler.extract_system_message(messages)
        options = self.options_handler.create_options(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            **kwargs,
        )

        # Validate model
        if not self.options_handler.validate_model(model):
            raise ClaudeProxyError(
                message=f"Unsupported model: {model}",
                error_type="invalid_request_error",
                status_code=400,
            )

        # Convert messages to prompt format
        prompt = self.message_converter.format_messages_to_prompt(messages)

        # Generate request ID for correlation
        from uuid import uuid4

        request_id = str(uuid4())

        try:
            if stream:
                return self._stream_completion(prompt, options, model, request_id)
            else:
                return await self._complete_non_streaming(
                    prompt, options, model, request_id
                )

        except Exception as e:
            # Error handling and logging can be added here if needed
            raise

    async def _complete_non_streaming(
        self, prompt: str, options: Any, model: str, request_id: str | None = None
    ) -> dict[str, Any]:
        """
        Complete a non-streaming request with business logic.

        Args:
            prompt: The formatted prompt
            options: Claude SDK options
            model: The model being used
            request_id: The request ID for metrics correlation

        Returns:
            Response in Anthropic format

        Raises:
            ClaudeProxyError: If completion fails
        """
        messages = []
        result_message = None
        assistant_message = None

        async for message in self.sdk_client.query_completion(prompt, options):
            messages.append(message)
            # Import here to avoid circular imports
            from claude_code_sdk import AssistantMessage, ResultMessage

            if isinstance(message, AssistantMessage):
                assistant_message = message
            elif isinstance(message, ResultMessage):
                result_message = message

        # Get Claude API call timing
        claude_api_call_ms = self.sdk_client.get_last_api_call_time_ms()

        if result_message is None:
            raise ClaudeProxyError(
                message="No result message received from Claude SDK",
                error_type="internal_server_error",
                status_code=500,
            )

        if assistant_message is None:
            raise ClaudeProxyError(
                message="No assistant response received from Claude SDK",
                error_type="internal_server_error",
                status_code=500,
            )

        # Convert to Anthropic format
        response = self.message_converter.convert_to_anthropic_response(
            assistant_message, result_message, model
        )

        # Completion metrics can be recorded here if needed

        return response

    async def _stream_completion(
        self, prompt: str, options: Any, model: str, request_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream completion responses with business logic.

        Args:
            prompt: The formatted prompt
            options: Claude SDK options
            model: The model being used

        Yields:
            Response chunks in Anthropic format
        """
        import asyncio

        first_chunk = True
        message_count = 0
        assistant_messages = []

        try:
            async for message in self.sdk_client.query_completion(prompt, options):
                message_count += 1
                logger.debug(
                    f"Claude SDK message {message_count}: {type(message).__name__}"
                )

                # Import here to avoid circular imports
                from claude_code_sdk import AssistantMessage, ResultMessage

                if isinstance(message, AssistantMessage):
                    assistant_messages.append(message)

                    if first_chunk:
                        # Send initial chunk
                        yield self.message_converter.create_streaming_start_chunk(
                            f"msg_{id(message)}", model
                        )
                        first_chunk = False

                    # Send content delta
                    text_content = self.message_converter.extract_text_from_content(
                        message.content
                    )
                    if text_content:
                        yield self.message_converter.create_streaming_delta_chunk(
                            text_content
                        )

                elif isinstance(message, ResultMessage):
                    # Get Claude API call timing
                    claude_api_call_ms = self.sdk_client.get_last_api_call_time_ms()

                    # Send final chunk
                    yield self.message_converter.create_streaming_end_chunk()

                    # Streaming metrics can be recorded here if needed
                    break

        except asyncio.CancelledError:
            logger.info("Stream completion cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in stream completion: {e}")
            yield self.message_converter.create_streaming_end_chunk("error")
            raise

    async def _validate_user_auth(self, user_id: str) -> None:
        """
        Validate user authentication.

        Args:
            user_id: User identifier

        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.auth_manager:
            return

        # Implement authentication validation logic
        # This is a placeholder for future auth integration
        logger.debug(f"Validating auth for user: {user_id}")

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List available Claude models.

        Returns:
            List of available models in Anthropic format
        """
        supported_models = self.options_handler.get_supported_models()
        models = []

        for model_id in supported_models:
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 1677610602,  # Static timestamp
                    "owned_by": "anthropic",
                }
            )

        return models

    async def validate_health(self) -> bool:
        """
        Validate that the service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            return await self.sdk_client.validate_health()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the service and cleanup resources."""
        await self.sdk_client.close()

    async def __aenter__(self) -> "ClaudeSDKService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
