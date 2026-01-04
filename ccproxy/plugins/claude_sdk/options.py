"""Options handling for Claude SDK interactions."""

from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from .config import ClaudeSDKSettings


class OptionsHandler:
    """
    Handles creation and management of Claude SDK options.
    """

    def __init__(self, config: ClaudeSDKSettings) -> None:
        """
        Initialize options handler.

        Args:
            config: Plugin-specific configuration for Claude SDK
        """
        self.config = config

    def create_options(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_message: str | None = None,
        **additional_options: Any,
    ) -> ClaudeAgentOptions:
        """
        Create Claude SDK options from API parameters.

        Args:
            model: The model name
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            system_message: System message to include
            **additional_options: Additional options to set on the ClaudeAgentOptions instance

        Returns:
            Configured ClaudeAgentOptions instance
        """
        # Start with configured defaults if available, otherwise create fresh instance
        if self.config and self.config.code_options:
            configured_opts = self.config.code_options
            options = ClaudeAgentOptions()

            # Copy all attributes from configured defaults
            for attr in dir(configured_opts):
                if not attr.startswith("_"):
                    configured_value = getattr(configured_opts, attr)
                    if configured_value is not None and hasattr(options, attr):
                        # Special handling for mcp_servers to ensure we copy the dict
                        if attr == "mcp_servers" and isinstance(configured_value, dict):
                            setattr(options, attr, configured_value.copy())
                        else:
                            setattr(options, attr, configured_value)
        else:
            options = ClaudeAgentOptions()

        # Override the model (API parameter takes precedence)
        options.model = model

        # Apply system message if provided (this is supported by ClaudeAgentOptions)
        if system_message is not None:
            options.system_prompt = system_message

        # If session_id is provided via additional_options, enable continue_conversation
        if additional_options.get("session_id"):
            options.continue_conversation = True

        # Automatically map additional_options to ClaudeAgentOptions attributes
        for key, value in additional_options.items():
            if hasattr(options, key):
                try:
                    # Attempt type conversion if the attribute already exists
                    attr_type = type(getattr(options, key))
                    # Only convert if the attribute is not None
                    if getattr(options, key) is not None:
                        setattr(options, key, attr_type(value))
                    else:
                        setattr(options, key, value)
                except Exception:
                    # Fallback to direct assignment if conversion fails
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
        from ccproxy.plugins.claude_shared.model_defaults import (
            DEFAULT_CLAUDE_MODEL_CARDS,
        )

        return [card.id for card in DEFAULT_CLAUDE_MODEL_CARDS]

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
        Get default options for API parameters.

        Returns:
            Dictionary of default API parameter values
        """
        return {
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.7,
            "max_tokens": 4000,
        }
