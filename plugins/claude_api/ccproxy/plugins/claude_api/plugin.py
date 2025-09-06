"""Claude API plugin v2 implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginContext,
    PluginManifest,
    ProviderPluginRuntime,
    TaskSpec,
)
from ccproxy.core.plugins.base_factory import BaseProviderPluginFactory
from ccproxy.core.plugins.declaration import FormatAdapterSpec, FormatPair
from ccproxy.plugins.oauth_claude.manager import ClaudeApiTokenManager

from .adapter import ClaudeAPIAdapter
from .config import ClaudeAPISettings
from .detection_service import ClaudeAPIDetectionService
from .health import claude_api_health_check
from .routes import router as claude_api_router
from .tasks import ClaudeAPIDetectionRefreshTask


# if TYPE_CHECKING:
#     from ccproxy.config.settings import Settings
#     from ccproxy.hooks.registry import HookRegistry
#     from ccproxy.services.cli_detection import CLIDetectionService
#     from ccproxy.services.container import ServiceContainer


logger = get_plugin_logger()


class ClaudeAPIRuntime(ProviderPluginRuntime):
    """Runtime for Claude API plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        self.credential_manager: ClaudeApiTokenManager | None = None
        super().__init__(manifest)
        self.config: ClaudeAPISettings | None = None

    async def _on_initialize(self) -> None:
        """Initialize the Claude API plugin."""
        # Call parent initialization first
        await super()._on_initialize()

        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        try:
            config = self.context.get(ClaudeAPISettings)
        except ValueError:
            logger.warning("plugin_no_config")
            # Use default config if none provided
            config = ClaudeAPISettings()
            logger.info("plugin_using_default_config")
        self.config = config

        # Setup format registry
        await self._setup_format_registry()

        # Register streaming metrics hook
        await self._register_streaming_metrics_hook()

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

    async def _setup_format_registry(self) -> None:
        """Format registry setup with feature flag control."""
        settings = self.context.get("settings")
        if settings is None:
            from ccproxy.config.settings import Settings

            settings = Settings()

        # Skip manual setup if manifest system is enabled
        if settings.features.manifest_format_adapters:
            logger.debug(
                "claude_api_format_registry_setup_skipped_using_manifest",
                category="format",
            )
            return

        # Deprecation warning for double registration
        if settings.features.deprecate_manual_format_setup:
            logger.warning(
                "deprecated_claude_api_manual_format_registry_setup",
                message="Manual format registry setup is deprecated. Use manifest format_adapters instead.",
                migration_guide="Update ClaudeAPIFactory.format_adapters list",
                category="format",
            )

        # Existing manual registration logic (or lack thereof)
        try:
            if not self.context:
                raise RuntimeError("Context not available for format registry setup")

            # Get format registry from service container
            from ccproxy.services.container import ServiceContainer

            service_container = self.context.get(ServiceContainer)
            registry = service_container.get_format_registry()

            # Claude API plugin now registers its own adapters manually
            from ccproxy.adapters.anthropic.response_adapter import (
                AnthropicResponseAPIAdapter,
            )
            from ccproxy.adapters.openai.adapter import OpenAIAdapter

            # Register the format adapters manually
            registry.register("openai", "anthropic", OpenAIAdapter(), "claude_api")
            registry.register(
                "response_api", "anthropic", AnthropicResponseAPIAdapter(), "claude_api"
            )

            logger.info(
                "claude_api_format_adapters_registered_manually",
                formats=registry.list_formats(),
                message="Registered claude_api format adapters manually",
                category="format",
            )

        except Exception as e:
            logger.error(
                "claude_api_format_registry_setup_failed",
                error=str(e),
                category="format",
            )
            raise ValueError("Failed to register Claude API format adapters") from e

    async def _register_streaming_metrics_hook(self) -> None:
        """Register the streaming metrics extraction hook."""
        try:
            if not self.context:
                logger.warning(
                    "streaming_metrics_hook_not_registered",
                    reason="no_context",
                    plugin="claude_api",
                )
                return
            # Debug: Log context details
            logger.debug(
                "streaming_metrics_hook_context_check",
                plugin="claude_api",
                has_context=self.context is not None,
                context_type=type(self.context).__name__ if self.context else None,
                context_keys=list(self.context.keys()) if self.context else [],
                has_hook_registry="hook_registry" in (self.context or {}),
                has_plugin_registry="plugin_registry" in (self.context or {}),
            )

            # Get hook registry from context
            from ccproxy.hooks.registry import HookRegistry

            try:
                hook_registry = self.context.get(HookRegistry)
            except ValueError:
                logger.warning(
                    "streaming_metrics_hook_not_registered",
                    reason="no_hook_registry",
                    plugin="claude_api",
                    context_keys=list(self.context.keys()) if self.context else [],
                )
                return

            # Get pricing service from plugin registry if available
            pricing_service = None
            if "plugin_registry" in self.context:
                try:
                    from ccproxy.plugins.pricing.service import PricingService

                    plugin_registry = self.context["plugin_registry"]
                    logger.debug(
                        "getting_pricing_service",
                        plugin="claude_api",
                        registry_type=type(plugin_registry).__name__,
                    )
                    pricing_service = plugin_registry.get_service(
                        "pricing", PricingService
                    )
                    logger.debug(
                        "pricing_service_obtained",
                        plugin="claude_api",
                        service_type=type(pricing_service).__name__
                        if pricing_service
                        else None,
                        is_none=pricing_service is None,
                    )
                except Exception as e:
                    logger.debug(
                        "pricing_service_not_available_for_hook",
                        plugin="claude_api",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
            else:
                logger.debug(
                    "plugin_registry_not_in_context",
                    plugin="claude_api",
                    context_keys=list(self.context.keys()) if self.context else [],
                )

            # Create and register the hook
            from .hooks import ClaudeAPIStreamingMetricsHook

            # Pass both pricing_service (if available now) and plugin_registry (for lazy loading)
            metrics_hook = ClaudeAPIStreamingMetricsHook(
                pricing_service=pricing_service,
                plugin_registry=self.context.get("plugin_registry"),
            )
            hook_registry.register(metrics_hook)

            logger.info(
                "streaming_metrics_hook_registered",
                plugin="claude_api",
                hook_name=metrics_hook.name,
                priority=metrics_hook.priority,
                has_pricing=pricing_service is not None,
                pricing_service_type=type(pricing_service).__name__
                if pricing_service
                else "None",
            )

        except Exception as e:
            logger.error(
                "streaming_metrics_hook_registration_failed",
                plugin="claude_api",
                error=str(e),
                exc_info=e,
            )


class ClaudeAPIFactory(BaseProviderPluginFactory):
    """Factory for Claude API plugin."""

    cli_safe = False  # Heavy provider plugin - not safe for CLI

    # Plugin configuration via class attributes
    plugin_name = "claude_api"
    plugin_description = "Claude API provider plugin with support for both native Anthropic format and OpenAI-compatible format"
    runtime_class = ClaudeAPIRuntime
    adapter_class = ClaudeAPIAdapter
    detection_service_class = ClaudeAPIDetectionService
    config_class = ClaudeAPISettings
    # Provide credentials manager so HTTP adapter receives an auth manager
    credentials_manager_class = ClaudeApiTokenManager
    router = claude_api_router
    route_prefix = "/api"
    # OAuth provider is optional because the token manager can operate
    # without a globally-registered auth provider. When present, it enables
    # first-class OAuth flows in the UI.
    dependencies = ["oauth_claude"]
    optional_requires = ["pricing"]

    # NEW: Declarative format adapter specification
    format_adapters = [
        FormatAdapterSpec(
            from_format="openai",
            to_format="anthropic",
            adapter_factory=lambda: __import__(
                "ccproxy.adapters.openai.adapter", fromlist=["OpenAIAdapter"]
            ).OpenAIAdapter(),
            priority=60,  # Lower priority than SDK plugin
            description="OpenAI to Anthropic format conversion for Claude API",
        ),
        FormatAdapterSpec(
            from_format="response_api",
            to_format="anthropic",
            adapter_factory=lambda: __import__(
                "ccproxy.adapters.anthropic.response_adapter",
                fromlist=["AnthropicResponseAPIAdapter"],
            ).AnthropicResponseAPIAdapter(),
            priority=50,  # Medium priority
            description="Response API to Anthropic format conversion for Claude API",
        ),
    ]

    # Define requirements for adapters this plugin needs
    requires_format_adapters: list[FormatPair] = [
        ("anthropic", "response_api"),  # Provided by core
    ]
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

    def create_detection_service(self, context: PluginContext) -> Any:
        """Create detection service and inject it into task kwargs.

        Ensures the scheduled detection-refresh task uses the same instance
        that the runtime receives via context.
        """
        detection_service = super().create_detection_service(context)

        if self.manifest.tasks and detection_service is not None:
            for task_spec in self.manifest.tasks:
                if task_spec.task_name == "claude_api_detection_refresh":
                    task_spec.kwargs["detection_service"] = detection_service

        return detection_service


# Create factory instance for plugin discovery
# Note: This follows the existing pattern but creates a singleton
factory = ClaudeAPIFactory()

__all__ = ["ClaudeAPIFactory", "ClaudeAPIRuntime", "factory"]
