"""Command Replay plugin implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from ccproxy.core.plugins.hooks import HookRegistry

from .config import CommandReplayConfig
from .formatter import CommandFileFormatter
from .hook import CommandReplayHook


logger = get_plugin_logger()


class CommandReplayRuntime(SystemPluginRuntime):
    """Runtime for the command replay plugin.

    Generates curl and xh commands for provider requests to enable
    easy replay and debugging of API calls.
    """

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: CommandReplayConfig | None = None
        self.hook: CommandReplayHook | None = None
        self.file_formatter: CommandFileFormatter | None = None

    async def _on_initialize(self) -> None:
        """Initialize the command replay plugin."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, CommandReplayConfig):
            logger.debug("plugin_no_config")
            config = CommandReplayConfig()
            logger.debug("plugin_using_default_config")
        self.config = config

        # Debug log the configuration being used
        logger.debug(
            "plugin_configuration_loaded",
            enabled=config.enabled,
            generate_curl=config.generate_curl,
            generate_xh=config.generate_xh,
            include_patterns=config.include_url_patterns,
            exclude_patterns=config.exclude_url_patterns,
            log_to_console=config.log_to_console,
            log_level=config.log_level,
            only_provider_requests=config.only_provider_requests,
        )

        if self.config.enabled:
            # Initialize file formatter if file writing is enabled
            if self.config.write_to_files:
                self.file_formatter = CommandFileFormatter(
                    log_dir=self.config.log_dir,
                    enabled=True,
                    separate_files_per_command=self.config.separate_files_per_command,
                )
                logger.debug(
                    "command_replay_file_formatter_initialized",
                    log_dir=self.config.log_dir,
                    separate_files=self.config.separate_files_per_command,
                )

            # Register hook for provider request events
            self.hook = CommandReplayHook(
                config=self.config,
                file_formatter=self.file_formatter,
            )

            # Try to get hook registry from context
            hook_registry = self.context.get("hook_registry")

            # If not found, try app state
            if not hook_registry:
                app = self.context.get("app")
                if app and hasattr(app.state, "hook_registry"):
                    hook_registry = app.state.hook_registry

            if hook_registry and isinstance(hook_registry, HookRegistry):
                hook_registry.register(self.hook)
                logger.debug(
                    "command_replay_hook_registered",
                    events=self.hook.events,
                    priority=self.hook.priority,
                    generate_curl=self.config.generate_curl,
                    generate_xh=self.config.generate_xh,
                    write_to_files=self.config.write_to_files,
                    log_dir=self.config.log_dir if self.config.write_to_files else None,
                )
            else:
                logger.warning(
                    "hook_registry_not_available",
                    fallback="disabled",
                )
        else:
            logger.debug("command_replay_plugin_disabled")

    async def _on_shutdown(self) -> None:
        """Clean up plugin resources."""
        if self.hook:
            logger.info("command_replay_plugin_shutdown")
            self.hook = None

        if self.file_formatter:
            self.file_formatter.cleanup()
            self.file_formatter = None

    def get_health_info(self) -> dict[str, Any]:
        """Get plugin health information."""
        return {
            "enabled": self.config.enabled if self.config else False,
            "hook_registered": self.hook is not None,
            "generate_curl": self.config.generate_curl if self.config else False,
            "generate_xh": self.config.generate_xh if self.config else False,
            "write_to_files": self.config.write_to_files if self.config else False,
            "file_formatter_enabled": self.file_formatter is not None,
            "log_dir": self.config.log_dir if self.config else None,
        }


class CommandReplayFactory(SystemPluginFactory):
    """Factory for creating command replay plugin instances."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations
        manifest = PluginManifest(
            name="command_replay",
            version="0.1.0",
            description="Generates curl and xh commands for provider requests",
            is_provider=False,
            config_class=CommandReplayConfig,
        )

        # Initialize with manifest
        super().__init__(manifest)

    def create_runtime(self) -> CommandReplayRuntime:
        """Create runtime instance."""
        return CommandReplayRuntime(self.manifest)

    def create_context(self, core_services: Any) -> Any:
        """Create context for the plugin."""
        # Get base context from parent
        context = super().create_context(core_services)
        return context


# Export the factory for plugin discovery
factory = CommandReplayFactory()
