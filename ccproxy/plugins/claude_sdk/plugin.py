"""Claude SDK plugin v2 implementation."""

from typing import Any

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    BaseProviderPluginFactory,
    FormatAdapterSpec,
    FormatPair,
    PluginContext,
    PluginManifest,
    ProviderPluginRuntime,
    TaskSpec,
)
from ccproxy.core.plugins.declaration import RouterSpec
from ccproxy.llms.formatters.formatter_adapter import create_formatter_adapter_factory
from ccproxy.services.adapters.base import BaseAdapter

from .adapter import ClaudeSDKAdapter
from .config import ClaudeSDKSettings
from .detection_service import ClaudeSDKDetectionService
from .routes import router
from .tasks import ClaudeSDKDetectionRefreshTask


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

        await self._setup_format_registry()

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
                logger.info(
                    "plugin_initialized",
                    plugin="claude_sdk",
                    version="1.0.0",
                    status="initialized",
                    has_credentials=True,  # SDK handles its own auth
                    cli_available=True,
                    cli_version=version,
                    cli_path=cli_path,
                    cli_source="package_manager",
                    has_adapter=self.adapter is not None,
                    has_session_manager=self.session_manager is not None,
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

    async def _setup_format_registry(self) -> None:
        """No-op; manifest-based format adapters are always used."""
        logger.debug(
            "claude_sdk_format_registry_setup_skipped_using_manifest",
            category="format",
        )


class ClaudeSDKFactory(BaseProviderPluginFactory):
    """Factory for Claude SDK plugin."""

    # Plugin configuration via class attributes
    plugin_name = "claude_sdk"
    plugin_description = (
        "Claude SDK plugin providing access to Claude through the Claude Code SDK"
    )
    runtime_class = ClaudeSDKRuntime
    adapter_class = ClaudeSDKAdapter
    detection_service_class = ClaudeSDKDetectionService
    config_class = ClaudeSDKSettings
    routers = [
        RouterSpec(router=router, prefix="/claude"),
    ]
    optional_requires = ["pricing"]

    format_adapters = [
        FormatAdapterSpec(
            from_format=FORMAT_OPENAI_CHAT,
            to_format=FORMAT_ANTHROPIC_MESSAGES,
            adapter_factory=create_formatter_adapter_factory(
                "openai.chat_completions", "anthropic.messages"
            ),
            priority=40,  # Higher priority than API plugin
            description="OpenAI to Anthropic format conversion for Claude SDK (FormatterRegistry)",
        )
    ]

    # Dependencies: anthropic.messages -> openai.responses handled by core
    requires_format_adapters: list[FormatPair] = [
        (FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_RESPONSES),
    ]

    tasks = [
        TaskSpec(
            task_name="claude_sdk_detection_refresh",
            task_type="claude_sdk_detection_refresh",
            task_class=ClaudeSDKDetectionRefreshTask,
            interval_seconds=3600,
            enabled=True,
            kwargs={"skip_initial_run": True},
        )
    ]

    async def create_adapter(self, context: PluginContext) -> BaseAdapter:
        """Create the Claude SDK adapter.

        This method overrides the base implementation because Claude SDK
        has different dependencies than HTTP-based adapters.

        Args:
            context: Plugin context

        Returns:
            ClaudeSDKAdapter instance
        """
        config = context.get("config")
        if not isinstance(config, ClaudeSDKSettings):
            raise RuntimeError("No configuration provided for Claude SDK adapter")

        # Get optional dependencies
        metrics = context.get("metrics")

        # Try to get hook_manager from context (provided by core services)
        hook_manager = context.get("hook_manager")
        if not hook_manager:
            # Try to get from app state as fallback
            app = context.get("app")
            if app and hasattr(app, "state") and hasattr(app.state, "hook_manager"):
                hook_manager = app.state.hook_manager

        if hook_manager:
            logger.debug("claude_sdk_hook_manager_found", source="context_or_app")

        # Create adapter with config and optional dependencies
        # Note: ClaudeSDKAdapter doesn't need http_client as it uses SDK
        adapter = ClaudeSDKAdapter(
            config=config, metrics=metrics, hook_manager=hook_manager
        )

        return adapter

    def create_detection_service(
        self, context: PluginContext
    ) -> ClaudeSDKDetectionService:
        """Create the Claude SDK detection service with validation.

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
