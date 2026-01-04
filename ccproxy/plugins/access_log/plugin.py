from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from ccproxy.core.plugins.hooks import HookRegistry
from ccproxy.plugins.analytics.ingest import AnalyticsIngestService
from ccproxy.services.container import ServiceContainer

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
        config: AccessLogConfig | None = self.context.get("config")
        if config is None or not isinstance(config, AccessLogConfig):
            config = AccessLogConfig()
        self.config = config

        if not config.enabled:
            return

        self.hook = AccessLogHook(config)

        hook_registry = self.context.get(HookRegistry)

        if hook_registry is None or not isinstance(hook_registry, HookRegistry):
            raise RuntimeError("Hook registry not found in context")

        hook_registry.register(self.hook)

        # Try to wire analytics ingest service if available
        try:
            registry = self.context.get(ServiceContainer)
            self.hook.ingest_service = registry.get_service(AnalyticsIngestService)
            if not self.hook.ingest_service:
                # optional service
                logger.debug("access_log_analytics_service_not_found")
        except Exception as e:
            logger.warning(
                "access_log_ingest_service_connect_failed", error=str(e), exc_info=e
            )
        #
        # Consolidated ready summary at INFO
        logger.trace(
            "access_log_ready",
            client_enabled=config.client_enabled,
            provider_enabled=config.provider_enabled,
            client_format=config.client_format,
            client_log_file=config.client_log_file,
            provider_log_file=config.provider_log_file,
        )

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
                logger.trace("access_log_hook_unregistered")

            # Close hook (flushes writers)
            await self.hook.close()
            logger.trace("access_log_shutdown")

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        config = self.config

        return {
            "type": "system",
            "initialized": self.initialized,
            "enabled": config.enabled if config else False,
            "client_enabled": config.client_enabled if config else False,
            "provider_enabled": config.provider_enabled if config else False,
        }

    def get_hook(self) -> AccessLogHook | None:
        """Get the hook instance (for testing or manual integration)."""
        return self.hook


class AccessLogFactory(SystemPluginFactory):
    """Factory for access log plugin."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="access_log",
            version="0.1.0",
            description="Simple access logging with Common, Combined, and Structured formats",
            is_provider=False,
            config_class=AccessLogConfig,
            # dependencies=["analytics"], # optional, handled at runtime
        )
        super().__init__(manifest)

    def create_runtime(self) -> AccessLogRuntime:
        return AccessLogRuntime(self.manifest)


# Export the factory instance
factory = AccessLogFactory()
