"""Claude SDK plugin v2 implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins import (
    PluginContext,
    PluginManifest,
    ProviderPluginFactory,
    ProviderPluginRuntime,
    RouteSpec,
    TaskSpec,
)
from plugins.claude_sdk.adapter import ClaudeSDKAdapter
from plugins.claude_sdk.config import ClaudeSDKSettings
from plugins.claude_sdk.detection_service import ClaudeSDKDetectionService
from plugins.claude_sdk.routes import router
from plugins.claude_sdk.tasks import ClaudeSDKDetectionRefreshTask


logger = get_plugin_logger()


class ClaudeSDKRuntime(ProviderPluginRuntime):
    """Runtime for Claude SDK plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.session_manager: Any | None = None

    async def _on_initialize(self) -> None:
        """Initialize the Claude SDK plugin."""
        # Call parent initialization to set up adapter, detection_service, etc.
        await super()._on_initialize()

        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, ClaudeSDKSettings):
            logger.warning("plugin_no_config")
            # Use default config if none provided
            config = ClaudeSDKSettings()
            logger.info("plugin_using_default_config")

        # Initialize adapter with session manager if enabled
        if self.adapter and hasattr(self.adapter, "session_manager"):
            self.session_manager = self.adapter.session_manager
            if self.session_manager:
                await self.session_manager.start()
                logger.info("session_manager_started")

        # Initialize detection service if present
        if self.detection_service and hasattr(
            self.detection_service, "initialize_detection"
        ):
            await self.detection_service.initialize_detection()

            # Check CLI status
            version = self.detection_service.get_version()
            cli_path = self.detection_service.get_cli_path()

            if cli_path:
                # Single consolidated log message with both CLI detection and plugin initialization status
                logger.debug(
                    "plugin_initialized",
                    status="initialized",
                    cli_available=True,
                    cli_version=version,
                    cli_path=cli_path,
                    cli_source="package_manager",
                    has_credentials=True,  # SDK handles its own auth
                    has_adapter=self.adapter is not None,
                    has_session_manager=self.session_manager is not None,
                    category="plugin",
                )
            else:
                error_msg = "Claude CLI not found in PATH or common locations - SDK plugin requires installed CLI"
                logger.error(
                    "plugin_initialization_failed",
                    status="failed",
                    error=error_msg,
                )
                raise RuntimeError(error_msg)

    async def _on_shutdown(self) -> None:
        """Cleanup on shutdown."""
        # Shutdown session manager first
        if self.session_manager:
            await self.session_manager.shutdown()
            logger.debug("session_manager_shutdown")

        # Call parent shutdown which handles adapter cleanup
        await super()._on_shutdown()

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        details = await super()._get_health_details()

        # Add SDK-specific health info
        details.update(
            {
                "has_session_manager": self.session_manager is not None,
            }
        )

        # Add CLI information if available
        if self.detection_service:
            details.update(
                {
                    "cli_available": self.detection_service.is_claude_available(),
                    "cli_version": self.detection_service.get_version(),
                    "cli_path": self.detection_service.get_cli_path(),
                }
            )

        return details


class ClaudeSDKFactory(ProviderPluginFactory):
    """Factory for Claude SDK plugin."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations
        manifest = PluginManifest(
            name="claude_sdk",
            version="1.0.0",
            description="Claude SDK plugin providing access to Claude through the Claude Code SDK",
            is_provider=True,
            config_class=ClaudeSDKSettings,
            routes=[
                RouteSpec(
                    router=router,
                    prefix="/claude",
                    tags=["plugin-claude_sdk"],
                )
            ],
            tasks=[
                TaskSpec(
                    task_name="claude_sdk_detection_refresh",
                    task_type="claude_sdk_detection_refresh",
                    task_class=ClaudeSDKDetectionRefreshTask,
                    interval_seconds=3600,  # Refresh every hour
                    enabled=True,
                    kwargs={"skip_initial_run": True},
                )
            ],
        )

        # Initialize with manifest
        super().__init__(manifest)

    def create_runtime(self) -> ClaudeSDKRuntime:
        """Create runtime instance."""
        return ClaudeSDKRuntime(self.manifest)

    def create_adapter(self, context: PluginContext) -> ClaudeSDKAdapter:
        """Create the Claude SDK adapter.

        Args:
            context: Plugin context

        Returns:
            ClaudeSDKAdapter instance
        """
        config = context.get("config")
        if not isinstance(config, ClaudeSDKSettings):
            raise RuntimeError("No configuration provided for Claude SDK adapter")

        proxy_service = context.get("proxy_service")

        # Create adapter with config and proxy service
        adapter = ClaudeSDKAdapter(config=config, proxy_service=proxy_service)

        return adapter

    def create_detection_service(
        self, context: PluginContext
    ) -> ClaudeSDKDetectionService:
        """Create the Claude SDK detection service.

        Args:
            context: Plugin context

        Returns:
            ClaudeSDKDetectionService instance
        """
        settings = context.get("settings")
        if not settings:
            raise RuntimeError("No settings provided for Claude SDK detection service")

        cli_service = context.get("cli_detection_service")
        return ClaudeSDKDetectionService(settings, cli_service)

    def create_credentials_manager(self, context: PluginContext) -> None:
        """Create the credentials manager for Claude SDK.

        Args:
            context: Plugin context

        Returns:
            None - Claude SDK uses its own authentication mechanism
        """
        # Claude SDK doesn't use a traditional credentials manager
        # It uses the built-in CLI authentication
        return None

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context and set up detection service in tasks."""
        # Get base context
        context = super().create_context(core_services)

        # Create detection service early so it can be passed to tasks
        detection_service = self.create_detection_service(context)

        # Update task kwargs with detection service
        for task_spec in self.manifest.tasks:
            if task_spec.task_name == "claude_sdk_detection_refresh":
                task_spec.kwargs["detection_service"] = detection_service

        return context


# Export the factory instance
factory = ClaudeSDKFactory()
