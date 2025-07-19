"""Options handling for Claude SDK interactions."""

from typing import Any

from ccproxy.core.async_utils import patched_typing


with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions


class OptionsHandler:
    """
    Handles creation and management of Claude SDK options.
    """

    @staticmethod
    def create_options(
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_message: str | None = None,
        **kwargs: Any,
    ) -> ClaudeCodeOptions:
        """
        Create Claude SDK options from API parameters.

        Args:
            model: The model name
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            system_message: System message to include
            **kwargs: Additional options

        Returns:
            Configured ClaudeCodeOptions instance
        """
        options = ClaudeCodeOptions(model=model)

        if temperature is not None:
            options.temperature = temperature  # type: ignore[attr-defined]

        if max_tokens is not None:
            options.max_tokens = max_tokens  # type: ignore[attr-defined]

        if system_message is not None:
            options.system_prompt = system_message

        # Handle other options as needed
        for key, value in kwargs.items():
            if hasattr(options, key):
                setattr(options, key, value)

        return options

    @staticmethod
    def extract_system_message(messages: list[dict[str, Any]]) -> str | None:
        """
        Extract system message from Anthropic messages format.

        Args:
            messages: List of messages in Anthropic format

        Returns:
            System message content if found, None otherwise
        """
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if isinstance(content, list):
                    # Handle content blocks
                    text_parts = []
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    return " ".join(text_parts)
                return str(content)
        return None

    @staticmethod
    def get_supported_models() -> list[str]:
        """
        Get list of supported Claude models.

        Returns:
            List of supported model names
        """
        # Import here to avoid circular imports
        from ccproxy.adapters.openai.adapter import OPENAI_TO_CLAUDE_MODEL_MAPPING

        # Extract unique Claude models from OpenAI mapping
        claude_models = list(set(OPENAI_TO_CLAUDE_MODEL_MAPPING.values()))
        return sorted(claude_models)

    @staticmethod
    def validate_model(model: str) -> bool:
        """
        Validate if a model is supported.

        Args:
            model: The model name to validate

        Returns:
            True if supported, False otherwise
        """
        return model in OptionsHandler.get_supported_models()

    @staticmethod
    def get_default_options() -> dict[str, Any]:
        """
        Get default options for Claude SDK.

        Returns:
            Dictionary of default options
        """
        return {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.7,
            "max_tokens": 4000,
        }
