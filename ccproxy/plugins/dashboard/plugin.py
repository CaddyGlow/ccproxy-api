from __future__ import annotations

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    RouteSpec,
    SystemPluginFactory,
    SystemPluginRuntime,
)

from .config import DashboardPluginConfig


logger = get_plugin_logger()


class DashboardRuntime(SystemPluginRuntime):
    async def _on_initialize(self) -> None:
        if not self.context:
            raise RuntimeError("Context not set")
        from typing import cast

        cfg = cast(DashboardPluginConfig | None, self.context.get("config"))
        app = self.context.get("app")
        if not app or not hasattr(app, "mount"):
            return

        # Optionally mount static assets for the SPA
        cfg = cfg or DashboardPluginConfig()
        if cfg.mount_static:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            dashboard_static_path = project_root / "ccproxy" / "static" / "dashboard"
            if dashboard_static_path.exists():
                try:
                    app.mount(
                        "/dashboard/assets",
                        StaticFiles(directory=str(dashboard_static_path)),
                        name="dashboard-static",
                    )
                    logger.debug(
                        "dashboard_static_files_mounted",
                        path=str(dashboard_static_path),
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning("dashboard_static_mount_failed", error=str(e))


class DashboardFactory(SystemPluginFactory):
    def __init__(self) -> None:
        from .routes import router as dashboard_router

        manifest = PluginManifest(
            name="dashboard",
            version="1.0.0",
            description="Dashboard SPA routes and static asset mounting",
            is_provider=False,
            config_class=DashboardPluginConfig,
            routes=[RouteSpec(router=dashboard_router, prefix="", tags=["dashboard"])],
        )
        super().__init__(manifest)

    def create_runtime(self) -> DashboardRuntime:
        return DashboardRuntime(self.manifest)


factory = DashboardFactory()
