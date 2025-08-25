"""Pricing plugin implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins import (
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)

from .config import PricingConfig
from .service import PricingService
from .tasks import PricingCacheUpdateTask


logger = get_plugin_logger()


class PricingRuntime(SystemPluginRuntime):
    """Runtime for pricing plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: PricingConfig | None = None
        self.service: PricingService | None = None
        self.update_task: PricingCacheUpdateTask | None = None

    async def _on_initialize(self) -> None:
        """Initialize the pricing plugin."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, PricingConfig):
            logger.warning("plugin_no_config_using_defaults")
            # Use default config if none provided
            self.config = PricingConfig()
        else:
            self.config = config

        logger.info("initializing_pricing_plugin", enabled=self.config.enabled)

        # Create pricing service
        self.service = PricingService(self.config)

        if self.config.enabled:
            # Initialize the service
            await self.service.initialize()

            # Create and start pricing update task
            interval_seconds = self.config.update_interval_hours * 3600
            self.update_task = PricingCacheUpdateTask(
                name="pricing_cache_update",
                interval_seconds=interval_seconds,
                pricing_service=self.service,
                enabled=self.config.auto_update,
                force_refresh_on_startup=self.config.force_refresh_on_startup,
            )

            await self.update_task.start()
            logger.info(
                "pricing_plugin_initialized",
                update_interval_hours=self.config.update_interval_hours,
                auto_update=self.config.auto_update,
                force_refresh_on_startup=self.config.force_refresh_on_startup,
            )
        else:
            logger.info("pricing_plugin_disabled")

    async def _on_shutdown(self) -> None:
        """Shutdown the plugin and cleanup resources."""
        logger.info("shutting_down_pricing_plugin")

        # Stop the update task
        if self.update_task:
            await self.update_task.stop()

        logger.info("pricing_plugin_shutdown_complete")

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        try:
            base_health = {
                "type": "system",
                "initialized": self.initialized,
                "enabled": self.config.enabled if self.config else False,
            }

            if not self.config or not self.config.enabled:
                return base_health

            # Add service-specific health info
            health_details = base_health.copy()

            if self.service:
                cache_info = self.service.get_cache_info()
                health_details.update(
                    {
                        "cache_valid": cache_info.get("valid", False),
                        "cache_age_hours": cache_info.get("age_hours"),
                        "cache_exists": cache_info.get("exists", False),
                    }
                )

            if self.update_task:
                task_status = self.update_task.get_status()
                health_details.update(
                    {
                        "update_task_running": task_status["running"],
                        "consecutive_failures": task_status["consecutive_failures"],
                        "last_success_ago_seconds": task_status[
                            "last_success_ago_seconds"
                        ],
                        "next_run_in_seconds": task_status["next_run_in_seconds"],
                    }
                )

            return health_details

        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return {
                "type": "system",
                "initialized": self.initialized,
                "enabled": self.config.enabled if self.config else False,
                "error": str(e),
            }

    def get_pricing_service(self) -> PricingService | None:
        """Get the pricing service instance."""
        return self.service


class PricingFactory(SystemPluginFactory):
    """Factory for pricing plugin."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations
        manifest = PluginManifest(
            name="pricing",
            version="1.0.0",
            description="Dynamic pricing plugin for AI model cost calculation",
            is_provider=False,
            config_class=PricingConfig,
        )

        # Initialize with manifest
        super().__init__(manifest)

    def create_runtime(self) -> PricingRuntime:
        """Create runtime instance."""
        return PricingRuntime(self.manifest)


# Export the factory instance
factory = PricingFactory()
