"""Claude API plugin v2 implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins import (
    PluginContext,
    PluginManifest,
    ProviderPluginRuntime,
    TaskSpec,
)
from ccproxy.plugins.base_factory import BaseProviderPluginFactory
from plugins.claude_api.adapter import ClaudeAPIAdapter
from plugins.claude_api.config import ClaudeAPISettings
from plugins.claude_api.detection_service import ClaudeAPIDetectionService
from plugins.claude_api.health import claude_api_health_check
from plugins.claude_api.routes import router as claude_api_router
from plugins.claude_api.tasks import ClaudeAPIDetectionRefreshTask


logger = get_plugin_logger()


class ClaudeAPIRuntime(ProviderPluginRuntime):
    """Runtime for Claude API plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: ClaudeAPISettings | None = None

    async def _on_initialize(self) -> None:
        """Initialize the Claude API plugin."""
        # Debug: Log what we receive in context
        logger.debug(
            "claude_api_initializing",
            context_keys=list(self.context.keys()) if self.context else [],
            has_config="config" in (self.context or {}),
            config_type=type(self.context.get("config")).__name__
            if self.context
            else None,
        )

        await super()._on_initialize()

        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, ClaudeAPISettings):
            logger.warning(
                "plugin_no_config",
                config_type=type(config).__name__ if config else None,
                config_value=config,
            )
            # Use default config if none provided
            config = ClaudeAPISettings()
            logger.info("plugin_using_default_config")
        self.config = config

        # Initialize detection service to populate cached data
        if self.detection_service:
            try:
                # This will detect headers and system prompt
                await self.detection_service.initialize_detection()
                version = self.detection_service.get_version()
                cli_path = self.detection_service.get_cli_path()

                if not cli_path:
                    logger.warning(
                        "cli_detection_completed",
                        cli_available=False,
                        version=None,
                        cli_path=None,
                        source="unknown",
                    )
            except Exception as e:
                logger.error(
                    "claude_detection_initialization_failed",
                    error=str(e),
                    exc_info=e,
                )

        # Get CLI info for consolidated logging (only for successful detection)
        cli_info = {}
        if self.detection_service and self.detection_service.get_cli_path():
            cli_info.update(
                {
                    "cli_available": True,
                    "cli_version": self.detection_service.get_version(),
                    "cli_path": self.detection_service.get_cli_path(),
                    "cli_source": "package_manager",
                }
            )

        logger.info(
            "plugin_initialized",
            plugin="claude_api",
            version="1.0.0",
            status="initialized",
            has_credentials=self.credentials_manager is not None,
            base_url=self.config.base_url,
            models_count=len(self.config.models) if self.config.models else 0,
            has_adapter=self.adapter is not None,
            **cli_info,
        )

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        details = await super()._get_health_details()

        # Add claude-api specific health check
        if self.config and self.detection_service and self.credentials_manager:
            try:
                health_result = await claude_api_health_check(
                    self.config, self.detection_service, self.credentials_manager
                )
                details.update(
                    {
                        "health_check_status": health_result.status,
                        "health_check_detail": health_result.details,
                    }
                )
            except Exception as e:
                details["health_check_error"] = str(e)

        return details

    async def get_profile_info(self) -> dict[str, Any] | None:
        """Get Claude-specific profile information from stored credentials."""
        try:
            if not self.credentials_manager:
                return None

            # Get profile using credentials manager
            profile = await self.credentials_manager.get_account_profile()
            if not profile:
                # Try to fetch fresh profile
                profile = await self.credentials_manager.fetch_user_profile()

            if profile:
                profile_info = {}

                if profile.organization:
                    profile_info.update(
                        {
                            "organization_name": profile.organization.name,
                            "organization_type": profile.organization.organization_type,
                            "billing_type": profile.organization.billing_type,
                            "rate_limit_tier": profile.organization.rate_limit_tier,
                        }
                    )

                if profile.account:
                    profile_info.update(
                        {
                            "email": profile.account.email,
                            "full_name": profile.account.full_name,
                            "display_name": profile.account.display_name,
                            "has_claude_pro": profile.account.has_claude_pro,
                            "has_claude_max": profile.account.has_claude_max,
                        }
                    )

                return profile_info

        except Exception as e:
            logger.debug(
                "claude_api_profile_error",
                error=str(e),
                exc_info=e,
            )

        return None


class ClaudeAPIFactory(BaseProviderPluginFactory):
    """Factory for Claude API plugin."""

    # Plugin configuration via class attributes
    plugin_name = "claude_api"
    plugin_description = "Claude API provider plugin with support for both native Anthropic format and OpenAI-compatible format"
    runtime_class = ClaudeAPIRuntime
    adapter_class = ClaudeAPIAdapter
    detection_service_class = ClaudeAPIDetectionService
    config_class = ClaudeAPISettings
    router = claude_api_router
    route_prefix = "/api"
    dependencies = ["oauth_claude"]
    optional_requires = ["pricing"]
    tasks = [
        TaskSpec(
            task_name="claude_api_detection_refresh",
            task_type="claude_api_detection_refresh",
            task_class=ClaudeAPIDetectionRefreshTask,
            interval_seconds=3600,
            enabled=True,
            kwargs={"skip_initial_run": True},
        )
    ]

    def create_credentials_manager(self, context: PluginContext) -> Any:
        """Create the credentials manager for Claude API.

        Args:
            context: Plugin context

        Returns:
            ClaudeApiTokenManager instance
        """
        from plugins.claude_api.auth.manager import ClaudeApiTokenManager

        return ClaudeApiTokenManager()

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context with additional components.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with Claude API components
        """
        # Get base context
        context = super().create_context(core_services)

        # Add detection service to context for task creation
        detection_service = self.create_detection_service(context)
        context["detection_service"] = detection_service

        # Update task spec with detection service
        if self.manifest.tasks:
            for task_spec in self.manifest.tasks:
                if task_spec.task_name == "claude_api_detection_refresh":
                    # Add detection service to task kwargs
                    task_spec.kwargs["detection_service"] = detection_service

        return context


# Export the factory instance
factory = ClaudeAPIFactory()
