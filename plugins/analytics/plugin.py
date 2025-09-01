from __future__ import annotations

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins import (
    PluginManifest,
    RouteSpec,
    SystemPluginFactory,
    SystemPluginRuntime,
)

from .config import AnalyticsPluginConfig


logger = get_plugin_logger()


class AnalyticsRuntime(SystemPluginRuntime):
    async def _on_initialize(self) -> None:
        logger.info("analytics_plugin_initialized")


class AnalyticsFactory(SystemPluginFactory):
    def __init__(self) -> None:
        from .routes import router as analytics_router

        manifest = PluginManifest(
            name="analytics",
            version="1.0.0",
            description="Logs query, analytics, and streaming endpoints",
            is_provider=False,
            config_class=AnalyticsPluginConfig,
            routes=[RouteSpec(router=analytics_router, prefix="/logs", tags=["logs"])],
        )
        super().__init__(manifest)

    def create_runtime(self) -> AnalyticsRuntime:
        return AnalyticsRuntime(self.manifest)


factory = AnalyticsFactory()

