"""Metrics plugin implementation."""

import asyncio
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.hooks import HookRegistry
from ccproxy.plugins import (
    PluginContext,
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)

from .config import MetricsConfig
from .hook import MetricsHook
from .routes import create_metrics_router


logger = get_plugin_logger()


class MetricsRuntime(SystemPluginRuntime):
    """Runtime for metrics plugin."""
    
    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: MetricsConfig | None = None
        self.hook: MetricsHook | None = None
        self.pushgateway_task: asyncio.Task[None] | None = None
    
    async def _on_initialize(self) -> None:
        """Initialize the metrics plugin."""
        if not self.context:
            raise RuntimeError("Context not set")
        
        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, MetricsConfig):
            logger.warning("metrics_no_config")
            # Use default config if none provided
            config = MetricsConfig()
            logger.info("metrics_using_default_config")
        self.config = config
        
        if self.config.enabled:
            # Create metrics hook
            self.hook = MetricsHook(self.config)
            
            # Register hook with registry
            hook_registry = None
            
            # Try direct from context first
            hook_registry = self.context.get("hook_registry")
            logger.debug(
                "hook_registry_from_context",
                found=hook_registry is not None,
                context_keys=list(self.context.keys()) if self.context else [],
            )
            
            # If not found, try app state
            if not hook_registry:
                app = self.context.get("app")
                if app and hasattr(app.state, "hook_registry"):
                    hook_registry = app.state.hook_registry
                    logger.debug("hook_registry_from_app_state", found=True)
            
            if hook_registry and isinstance(hook_registry, HookRegistry):
                hook_registry.register(self.hook)
                logger.info(
                    "metrics_hook_registered",
                    namespace=self.config.namespace,
                    pushgateway_enabled=self.config.pushgateway_enabled,
                    metrics_endpoint_enabled=self.config.metrics_endpoint_enabled,
                )
            else:
                logger.warning(
                    "hook_registry_not_available",
                    message="Metrics plugin will not collect metrics via hooks"
                )
            
            # Register metrics endpoint if enabled
            if self.config.metrics_endpoint_enabled and self.hook:
                app = self.context.get("app")
                if app:
                    # Create and register metrics router
                    metrics_router = create_metrics_router(self.hook.get_collector())
                    app.include_router(metrics_router, prefix="")
                    logger.info(
                        "metrics_endpoint_registered",
                        endpoint="/metrics",
                    )
            
            # Start pushgateway push task if enabled
            if self.config.pushgateway_enabled and self.hook:
                self.pushgateway_task = asyncio.create_task(
                    self._pushgateway_loop()
                )
                logger.info(
                    "pushgateway_task_started",
                    url=self.config.pushgateway_url,
                    job=self.config.pushgateway_job,
                    interval=self.config.pushgateway_push_interval,
                )
            
            logger.info(
                "metrics_plugin_enabled",
                namespace=self.config.namespace,
                collect_request_metrics=self.config.collect_request_metrics,
                collect_token_metrics=self.config.collect_token_metrics,
                collect_cost_metrics=self.config.collect_cost_metrics,
                collect_error_metrics=self.config.collect_error_metrics,
                collect_pool_metrics=self.config.collect_pool_metrics,
            )
        else:
            logger.info("metrics_plugin_disabled")
    
    async def _pushgateway_loop(self) -> None:
        """Background task to periodically push metrics to Pushgateway."""
        if not self.config or not self.hook:
            return
        
        while True:
            try:
                await asyncio.sleep(self.config.pushgateway_push_interval)
                
                # Push metrics
                success = await self.hook.push_metrics()
                if success:
                    logger.debug(
                        "pushgateway_push_success",
                        url=self.config.pushgateway_url,
                        job=self.config.pushgateway_job,
                    )
                else:
                    logger.warning(
                        "pushgateway_push_failed",
                        url=self.config.pushgateway_url,
                        job=self.config.pushgateway_job,
                    )
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                logger.debug("pushgateway_task_cancelled")
                break
            except Exception as e:
                logger.error(
                    "pushgateway_loop_error",
                    error=str(e),
                    exc_info=e,
                )
                # Continue looping despite errors
    
    async def _on_shutdown(self) -> None:
        """Cleanup on shutdown."""
        # Cancel pushgateway task if running
        if self.pushgateway_task and not self.pushgateway_task.done():
            self.pushgateway_task.cancel()
            try:
                await self.pushgateway_task
            except asyncio.CancelledError:
                pass
            logger.debug("pushgateway_task_stopped")
        
        # Unregister hook from registry
        if self.hook:
            hook_registry = None
            if self.context:
                app = self.context.get("app")
                if app and hasattr(app.state, "hook_registry"):
                    hook_registry = app.state.hook_registry
                if not hook_registry:
                    hook_registry = self.context.get("hook_registry")
            
            if hook_registry and isinstance(hook_registry, HookRegistry):
                hook_registry.unregister(self.hook)
                logger.debug("metrics_hook_unregistered")
        
        # Push final metrics if pushgateway is enabled
        if self.config and self.config.pushgateway_enabled and self.hook:
            try:
                await self.hook.push_metrics()
                logger.info("final_metrics_pushed_to_pushgateway")
            except Exception as e:
                logger.error(
                    "final_metrics_push_failed",
                    error=str(e),
                    exc_info=e,
                )
    
    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        details = {
            "type": "system",
            "initialized": self.initialized,
            "enabled": self.config.enabled if self.config else False,
        }
        
        if self.config and self.config.enabled:
            details.update({
                "namespace": self.config.namespace,
                "metrics_endpoint_enabled": self.config.metrics_endpoint_enabled,
                "pushgateway_enabled": self.config.pushgateway_enabled,
                "pushgateway_url": self.config.pushgateway_url,
                "collector_enabled": self.hook.get_collector().is_enabled() if self.hook else False,
            })
        
        return details


class MetricsFactory(SystemPluginFactory):
    """Factory for metrics plugin."""
    
    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest
        manifest = PluginManifest(
            name="metrics",
            version="1.0.0",
            description="Prometheus metrics collection and export plugin",
            is_provider=False,
            config_class=MetricsConfig,
        )
        
        # Initialize with manifest
        super().__init__(manifest)
    
    def create_runtime(self) -> MetricsRuntime:
        """Create runtime instance."""
        return MetricsRuntime(self.manifest)
    
    def create_context(self, core_services: Any) -> PluginContext:
        """Create context for the plugin.
        
        Args:
            core_services: Core services from the application
            
        Returns:
            Plugin context with required services
        """
        # Get base context
        context = super().create_context(core_services)
        
        # The metrics plugin doesn't need special context setup
        # It will get hook_registry and app from the base context
        
        return context


# Export the factory instance
factory = MetricsFactory()