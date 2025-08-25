"""Permissions plugin v2 implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins import (
    PluginContext,
    PluginManifest,
    RouteSpec,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from plugins.permissions.config import PermissionsConfig
from plugins.permissions.mcp import mcp_router
from plugins.permissions.routes import router
from plugins.permissions.service import get_permission_service


logger = get_plugin_logger()


class PermissionsRuntime(SystemPluginRuntime):
    """Runtime for permissions plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: PermissionsConfig | None = None
        self.service = get_permission_service()

    async def _on_initialize(self) -> None:
        """Initialize the permissions plugin."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, PermissionsConfig):
            logger.warning("plugin_no_config")
            # Use default config if none provided
            self.config = PermissionsConfig()
        else:
            self.config = config

        logger.info("initializing_permissions_plugin")

        # Start the permission service if enabled
        if self.config.enabled:
            # Update service timeout from config
            self.service._timeout_seconds = self.config.timeout_seconds
            await self.service.start()
            logger.info(
                "permission_service_started",
                timeout_seconds=self.config.timeout_seconds,
                terminal_ui=self.config.enable_terminal_ui,
                sse_stream=self.config.enable_sse_stream,
            )
        else:
            logger.info("permission_service_disabled")

    async def _on_shutdown(self) -> None:
        """Shutdown the plugin and cleanup resources."""
        logger.info("shutting_down_permissions_plugin")

        # Stop the permission service
        await self.service.stop()

        logger.info("permissions_plugin_shutdown_complete")

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        try:
            # Check if service is running
            pending_count = len(await self.service.get_pending_requests())
            return {
                "type": "system",
                "initialized": self.initialized,
                "pending_requests": pending_count,
                "enabled": self.config.enabled if self.config else False,
                "service_running": self.service is not None,
            }
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return {
                "type": "system",
                "initialized": self.initialized,
                "enabled": self.config.enabled if self.config else False,
                "error": str(e),
            }


class PermissionsFactory(SystemPluginFactory):
    """Factory for permissions plugin."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations
        manifest = PluginManifest(
            name="permissions",
            version="1.0.0",
            description="Permissions plugin providing authorization services for tool calls",
            is_provider=False,
            config_class=PermissionsConfig,
        )

        # Initialize with manifest
        super().__init__(manifest)

    def create_runtime(self) -> PermissionsRuntime:
        """Create runtime instance."""
        return PermissionsRuntime(self.manifest)

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context and update manifest with routes if enabled."""
        # Get base context
        context = super().create_context(core_services)

        # Check if plugin is enabled
        config = context.get("config")
        if isinstance(config, PermissionsConfig) and config.enabled:
            # Add routes to manifest
            # This is safe because it happens during app creation phase
            if not self.manifest.routes:
                self.manifest.routes = []

            # Always add MCP routes at /mcp root (they're essential for Claude Code)
            mcp_route_spec = RouteSpec(
                router=mcp_router,
                prefix="/mcp",
                tags=["mcp"],
            )
            self.manifest.routes.append(mcp_route_spec)

            # Add SSE streaming routes at /permissions if enabled
            if config.enable_sse_stream:
                permissions_route_spec = RouteSpec(
                    router=router,
                    prefix="/permissions",
                    tags=["permissions"],
                )
                self.manifest.routes.append(permissions_route_spec)

            logger.debug(
                "permissions_routes_added_to_manifest",
                sse_enabled=config.enable_sse_stream,
            )

        return context


# Export the factory instance
factory = PermissionsFactory()
