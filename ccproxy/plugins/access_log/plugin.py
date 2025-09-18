from typing import Any

from ccproxy.core.log_events import ACCESS_LOG_READY, HOOK_REGISTERED
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from ccproxy.core.plugins.hooks import HookRegistry

from .config import AccessLogConfig
from .hook import AccessLogHook


logger = get_plugin_logger()


class AccessLogRuntime(SystemPluginRuntime):
    """Runtime for access log plugin.

    Integrates with the Hook system to receive and log events.
    """

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.hook: AccessLogHook | None = None
        self.config: AccessLogConfig | None = None

    async def _on_initialize(self) -> None:
        """Initialize the access logger."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, AccessLogConfig):
            logger.info("plugin_no_config")
            config = AccessLogConfig()
            logger.debug("plugin_using_default_config")
        self.config = config

        if not config.enabled:
            logger.info("access_log_disabled")
            return

        # Create hook instance
        self.hook = AccessLogHook(config)

        # Get hook registry from context
        hook_registry = None

        # Try direct from context first (provided by CoreServicesAdapter)
        hook_registry = self.context.get("hook_registry")
        logger.debug(
            "hook_registry_from_context",
            found=hook_registry is not None,
            context_keys=list(self.context.keys()) if self.context else [],
        )

        # If not found, try app state
        if not hook_registry:
            app = self.context.get("app")
            if app and hasattr(app, "state") and hasattr(app.state, "hook_registry"):
                hook_registry = app.state.hook_registry
                logger.debug("hook_registry_from_app_state", found=True)

        if hook_registry and isinstance(hook_registry, HookRegistry):
            hook_registry.register(self.hook)
            logger.debug(
                HOOK_REGISTERED,
                mode="hooks",
                client_enabled=config.client_enabled,
                client_format=config.client_format,
                client_log_file=config.client_log_file,
                provider_enabled=config.provider_enabled,
                provider_log_file=config.provider_log_file,
            )
            # Consolidated ready summary at INFO
            logger.info(
                ACCESS_LOG_READY,
                client_enabled=config.client_enabled,
                provider_enabled=config.provider_enabled,
                client_format=config.client_format,
                client_log_file=config.client_log_file,
                provider_log_file=config.provider_log_file,
            )
        else:
            logger.warning(
                "hook_registry_not_available",
                mode="hooks",
                fallback="No fallback - access logging disabled",
            )

        # Try to wire analytics ingest service if available
        try:
            if self.context and self.hook:
                registry = self.context.get("plugin_registry")
                ingest_service = None
                if registry:
                    from ccproxy.plugins.analytics.ingest import AnalyticsIngestService

                    ingest_service = registry.get_service(
                        "analytics_ingest", AnalyticsIngestService
                    )
                if not ingest_service and self.context.get("app"):
                    # Not registered in registry; skip silently
                    pass
                if ingest_service:
                    self.hook.ingest_service = ingest_service
                    logger.debug("access_log_ingest_service_connected")
        except Exception as e:
            logger.debug("access_log_ingest_service_connect_failed", error=str(e))

    async def _on_shutdown(self) -> None:
        """Cleanup on shutdown."""
        # Unregister hook from registry
        if self.hook:
            # Try to get hook registry
            hook_registry = None
            if self.context:
                hook_registry = self.context.get("hook_registry")
                if not hook_registry:
                    app = self.context.get("app")
                    if (
                        app
                        and hasattr(app, "state")
                        and hasattr(app.state, "hook_registry")
                    ):
                        hook_registry = app.state.hook_registry

            if hook_registry and isinstance(hook_registry, HookRegistry):
                hook_registry.unregister(self.hook)
                logger.debug("access_log_hook_unregistered")

            # Close hook (flushes writers)
            await self.hook.close()
            logger.debug("access_log_shutdown")

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        config = self.config

        return {
            "type": "system",
            "initialized": self.initialized,
            "enabled": config.enabled if config else False,
            "client_enabled": config.client_enabled if config else False,
            "provider_enabled": config.provider_enabled if config else False,
            "mode": "hooks",  # Now integrated with Hook system
        }

    def get_hook(self) -> AccessLogHook | None:
        """Get the hook instance (for testing or manual integration)."""
        return self.hook


class AccessLogFactory(SystemPluginFactory):
    """Factory for access log plugin."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="access_log",
            version="1.0.0",
            description="Simple access logging with Common, Combined, and Structured formats",
            is_provider=False,
            config_class=AccessLogConfig,
            dependencies=["analytics"],
        )
        super().__init__(manifest)

    def create_runtime(self) -> AccessLogRuntime:
        return AccessLogRuntime(self.manifest)


# Export the factory instance
factory = AccessLogFactory()
