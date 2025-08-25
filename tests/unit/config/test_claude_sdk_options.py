"""Tests for Claude SDK options handling."""

from typing import Any, cast

from claude_code_sdk import ClaudeCodeOptions

from ccproxy.config.settings import Settings
from ccproxy.core.async_utils import patched_typing
from plugins.claude_sdk.config import ClaudeSDKSettings
from plugins.claude_sdk.options import OptionsHandler


with patched_typing():
    pass


class TestOptionsHandler:
    """Test the OptionsHandler class."""

    def test_create_options_without_settings(self):
        """Test creating options without any settings."""
        handler = OptionsHandler(settings=None)

        options = handler.create_options(model="claude-3-5-sonnet-20241022")

        assert options.model == "claude-3-5-sonnet-20241022"
        # Should not have any defaults - mcp_servers will be empty dict, permission_prompt_tool_name will be None
        assert options.mcp_servers == {}  # ClaudeCodeOptions defaults to empty dict
        assert options.permission_prompt_tool_name is None

    def test_create_options_with_default_mcp_configuration(self):
        """Test that default MCP configuration is applied from settings."""
        # Create settings with default Claude configuration (includes MCP defaults)
        claude_settings = (
            ClaudeSDKSettings()
        )  # Uses the default factory with MCP config
        settings = Settings(claude=claude_settings)

        handler = OptionsHandler(settings=settings)

        options = handler.create_options(model="claude-3-5-sonnet-20241022")

        assert options.model == "claude-3-5-sonnet-20241022"
        # Should have the default MCP server configuration
        assert hasattr(options, "mcp_servers")
        assert options.mcp_servers is not None
        mcp_servers = cast(dict[str, Any], options.mcp_servers)
        assert "confirmation" in mcp_servers
        assert mcp_servers["confirmation"].get("type") == "sse"
        assert mcp_servers["confirmation"].get("url") == "http://127.0.0.1:8000/mcp"
        # Should have the default permission tool name
        assert hasattr(options, "permission_prompt_tool_name")
        assert (
            options.permission_prompt_tool_name == "mcp__confirmation__check_permission"
        )

    def test_create_options_with_custom_configuration(self):
        """Test that custom configuration overrides defaults."""
        # Create custom code options object with different values
        custom_code_options = ClaudeCodeOptions(
            mcp_servers={"custom": {"type": "sse", "url": "http://localhost:9000/mcp"}},
            permission_prompt_tool_name="custom_permission_tool",
            max_thinking_tokens=15000,
        )

        claude_settings = ClaudeSDKSettings(code_options=custom_code_options)
        settings = Settings(claude=claude_settings)

        handler = OptionsHandler(settings=settings)

        options = handler.create_options(model="claude-3-5-sonnet-20241022")

        assert options.model == "claude-3-5-sonnet-20241022"
        # Should have the custom MCP server configuration
        mcp_servers = cast(dict[str, Any], options.mcp_servers)
        assert "custom" in mcp_servers
        assert mcp_servers["custom"].get("url") == "http://localhost:9000/mcp"
        # Should have the custom permission tool name
        assert options.permission_prompt_tool_name == "custom_permission_tool"
        # Should have the custom max thinking tokens
        assert options.max_thinking_tokens == 15000

    def test_create_options_api_parameters_override_settings(self):
        """Test that API parameters override settings."""
        claude_settings = ClaudeSDKSettings()  # Uses defaults
        settings = Settings(claude=claude_settings)

        handler = OptionsHandler(settings=settings)

        options = handler.create_options(
            model="claude-3-5-sonnet-20241022",
            temperature=0.8,
            max_tokens=2000,
            system_message="Custom system prompt",
            max_thinking_tokens=20000,  # Override default
        )

        assert options.model == "claude-3-5-sonnet-20241022"
        assert options.system_prompt == "Custom system prompt"
        assert options.max_thinking_tokens == 20000
        # Should still have the default MCP configuration
        mcp_servers = cast(dict[str, Any], options.mcp_servers)
        assert "confirmation" in mcp_servers
        assert (
            options.permission_prompt_tool_name == "mcp__confirmation__check_permission"
        )

    def test_create_options_with_kwargs_override(self):
        """Test that additional kwargs are applied correctly."""
        claude_settings = ClaudeSDKSettings()
        settings = Settings(claude=claude_settings)
        handler = OptionsHandler(settings=settings)

        options = handler.create_options(
            model="claude-3-5-sonnet-20241022",
            allowed_tools=["Read", "Write"],
            cwd="/custom/path",
            permission_prompt_tool_name="override_tool",  # Override default
        )

        assert options.model == "claude-3-5-sonnet-20241022"
        assert options.allowed_tools == ["Read", "Write"]
        assert options.cwd == "/custom/path"
        # Should override the default permission tool name
        assert options.permission_prompt_tool_name == "override_tool"
        # Should still have the default MCP configuration

        mcp_servers = cast(dict[str, Any], options.mcp_servers)
        assert "confirmation" in mcp_servers

    def test_create_options_preserves_all_configuration_attributes(self):
        """Test that all attributes from configuration are properly copied."""
        # Create comprehensive configuration
        custom_code_options = ClaudeCodeOptions(
            mcp_servers={
                "confirmation": {"type": "sse", "url": "http://127.0.0.1:8000/mcp"},
                "filesystem": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["@modelcontextprotocol/server-filesystem"],
                },
            },
            permission_prompt_tool_name="mcp__confirmation__check_permission",
            max_thinking_tokens=10000,
            allowed_tools=["Read", "Write", "Bash"],
            cwd="/project/root",
            append_system_prompt="Additional context",
        )

        claude_settings = ClaudeSDKSettings(code_options=custom_code_options)
        settings = Settings(claude=claude_settings)
        handler = OptionsHandler(settings=settings)

        options = handler.create_options(model="claude-3-5-sonnet-20241022")

        # Verify all attributes are preserved
        mcp_servers = cast(dict[str, Any], options.mcp_servers)
        assert len(mcp_servers) == 2
        assert "confirmation" in mcp_servers
        assert "filesystem" in mcp_servers
        assert (
            options.permission_prompt_tool_name == "mcp__confirmation__check_permission"
        )
        assert options.max_thinking_tokens == 10000
        assert options.allowed_tools == ["Read", "Write", "Bash"]
        assert options.cwd == "/project/root"
        assert options.append_system_prompt == "Additional context"

    def test_model_parameter_always_overrides_settings(self):
        """Test that the model parameter always takes precedence over settings."""
        custom_code_options = ClaudeCodeOptions(
            model="claude-3-opus-20240229"  # Different model in settings
        )

        claude_settings = ClaudeSDKSettings(code_options=custom_code_options)
        settings = Settings(claude=claude_settings)
        handler = OptionsHandler(settings=settings)

        options = handler.create_options(model="claude-3-5-sonnet-20241022")

        # API model parameter should override settings model
        assert options.model == "claude-3-5-sonnet-20241022"

    def test_get_supported_models(self):
        """Test getting supported models list."""
        models = OptionsHandler.get_supported_models()

        assert isinstance(models, list)
        assert len(models) > 0
        # Should include common Claude models
        assert any("claude-3" in model for model in models)

    def test_validate_model(self):
        """Test model validation."""
        # Should work with supported models
        assert OptionsHandler.validate_model("claude-3-5-sonnet-20241022")

        # Should fail with unsupported models
        assert not OptionsHandler.validate_model("invalid-model")

    def test_get_default_options(self):
        """Test getting default options."""
        defaults = OptionsHandler.get_default_options()

        assert isinstance(defaults, dict)
        assert "model" in defaults
        assert "temperature" in defaults
        assert "max_tokens" in defaults

        # Verify expected defaults
        assert defaults["temperature"] == 0.7
        assert defaults["max_tokens"] == 4000
