"""Options handling for Claude SDK interactions."""

from typing import Any

from claude_code_sdk import ClaudeCodeOptions

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
    ) -> ClaudeCodeOptions:
        """
        Create Claude SDK options from API parameters.

        Args:
            model: The model name
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            system_message: System message to include
            **additional_options: Additional options to set on the ClaudeCodeOptions instance

        Returns:
            Configured ClaudeCodeOptions instance
        """
        # Start with configured defaults if available, otherwise create fresh instance
        if self.config and self.config.code_options:
            # Use the configured options as base - this preserves all default settings
            # including complex objects like mcp_servers and permission_prompt_tool_name
            configured_opts = self.config.code_options

            # Create a new instance with the same configuration
            # We need to extract the configuration values properly with type safety

            # Extract configuration values with proper types
            mcp_servers = (
                configured_opts.mcp_servers.copy()
                if isinstance(configured_opts.mcp_servers, dict)
                else {}
            )
            permission_prompt_tool_name = configured_opts.permission_prompt_tool_name
            max_thinking_tokens = getattr(configured_opts, "max_thinking_tokens", None)
            allowed_tools = getattr(configured_opts, "allowed_tools", None)
            disallowed_tools = getattr(configured_opts, "disallowed_tools", None)
            cwd = getattr(configured_opts, "cwd", None)
            append_system_prompt = getattr(
                configured_opts, "append_system_prompt", None
            )
            max_turns = getattr(configured_opts, "max_turns", None)
            continue_conversation = getattr(
                configured_opts, "continue_conversation", None
            )
            permission_mode = getattr(configured_opts, "permission_mode", None)

            # Build ClaudeCodeOptions with proper type handling
            # Start with a basic instance and set attributes individually for type safety
            options = ClaudeCodeOptions(
                mcp_servers=mcp_servers,
                permission_prompt_tool_name=permission_prompt_tool_name,
            )

            # Set additional attributes if they exist and are not None
            if max_thinking_tokens is not None:
                options.max_thinking_tokens = int(max_thinking_tokens)
            if allowed_tools is not None:
                options.allowed_tools = list(allowed_tools)
            if disallowed_tools is not None:
                options.disallowed_tools = list(disallowed_tools)
            if cwd is not None:
                options.cwd = cwd
            if append_system_prompt is not None:
                options.append_system_prompt = append_system_prompt
            if max_turns is not None:
                options.max_turns = max_turns
            if continue_conversation is not None:
                options.continue_conversation = bool(continue_conversation)
            if permission_mode is not None:
                options.permission_mode = permission_mode
        else:
            options = ClaudeCodeOptions()

        # Override the model (API parameter takes precedence)
        options.model = model

        # Apply system message if provided (this is supported by ClaudeCodeOptions)
        if system_message is not None:
            options.system_prompt = system_message

        # If session_id is provided via additional_options, enable continue_conversation
        # This ensures conversation continuity when using session IDs
        if additional_options.get("session_id"):
            options.continue_conversation = True

        # Note: temperature and max_tokens are API-level parameters, not ClaudeCodeOptions parameters
        # These are handled at the API request level, not in the options object

        # Handle additional options as needed
        for key, value in additional_options.items():
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
        from ccproxy.utils.model_mapping import get_supported_claude_models

        # Get supported Claude models
        claude_models = get_supported_claude_models()
        return claude_models

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
