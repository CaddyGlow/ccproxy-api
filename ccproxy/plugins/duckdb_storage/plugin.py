from __future__ import annotations

from pathlib import Path
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    RouteSpec,
    SystemPluginFactory,
    SystemPluginRuntime,
)

from .config import DuckDBStorageConfig
from .storage import SimpleDuckDBStorage


logger = get_plugin_logger()


def _default_db_path() -> str:
    # Mirrors previous default: XDG_DATA_HOME/ccproxy/metrics.duckdb
    import os

    return str(
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        / "ccproxy"
        / "metrics.duckdb"
    )


class DuckDBStorageRuntime(SystemPluginRuntime):
    """Runtime for DuckDB storage plugin."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.config: DuckDBStorageConfig | None = None
        self.storage: SimpleDuckDBStorage | None = None

    async def _on_initialize(self) -> None:
        if not self.context:
            raise RuntimeError("Context not set")

        # Resolve config
        cfg = self.context.get("config")
        if not isinstance(cfg, DuckDBStorageConfig):
            logger.warning("plugin_no_config_using_defaults")
            cfg = DuckDBStorageConfig()
        self.config = cfg

        # Determine if storage should be enabled: respect plugin flag and any
        # app-wide observability needs (logs endpoints/collection) if present.
        # Enable only if plugin config enables it
        enabled = bool(cfg.enabled)
        if not enabled:
            from ccproxy.core.logging import reduce_startup

            if reduce_startup(
                self.context.get("app") if hasattr(self, "context") else None
            ):
                logger.debug("duckdb_plugin_disabled", category="plugin")
            else:
                logger.info("duckdb_plugin_disabled", category="plugin")
            return

        # Resolve DB path
        db_path = cfg.database_path or _default_db_path()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        self.storage = SimpleDuckDBStorage(database_path=db_path)
        await self.storage.initialize()

        # Expose storage via plugin registry and app.state
        registry = self.context.get("plugin_registry")
        if registry:
            registry.register_service("log_storage", self.storage, self.manifest.name)
            from ccproxy.core.logging import reduce_startup

            if reduce_startup(
                self.context.get("app") if hasattr(self, "context") else None
            ):
                logger.debug(
                    "duckdb_storage_service_registered", path=db_path, category="plugin"
                )
            else:
                logger.info(
                    "duckdb_storage_service_registered", path=db_path, category="plugin"
                )

        app = self.context.get("app")
        if app and hasattr(app, "state"):
            app.state.log_storage = self.storage
            if cfg.register_app_state_alias:
                # Backward compat alias
                app.state.duckdb_storage = self.storage
            logger.debug("duckdb_storage_attached_to_app_state")

        logger.info("duckdb_storage_initialized", path=db_path, category="plugin")

    async def _on_shutdown(self) -> None:
        if self.storage:
            # Optional optimize on shutdown
            if self.config and self.config.optimize_on_shutdown:
                try:
                    self.storage.optimize()
                except Exception as e:  # pragma: no cover - best-effort
                    logger.warning("duckdb_optimize_on_shutdown_failed", error=str(e))
            try:
                await self.storage.close()
            except Exception as e:
                logger.warning("duckdb_storage_close_error", error=str(e))
            self.storage = None

    async def _get_health_details(self) -> dict[str, Any]:
        has_service = False
        if self.context:
            reg = self.context.get("plugin_registry")
            if reg is not None:
                try:
                    has_service = reg.has_service("log_storage")
                except Exception:
                    has_service = False
        return {
            "type": "system",
            "initialized": self.initialized,
            "enabled": bool(self.storage),
            "has_service": has_service,
        }


class DuckDBStorageFactory(SystemPluginFactory):
    def __init__(self) -> None:
        from .routes import router as duckdb_router

        manifest = PluginManifest(
            name="duckdb_storage",
            version="1.0.0",
            description="Provides DuckDB-backed request log storage",
            is_provider=False,
            provides=["log_storage"],
            config_class=DuckDBStorageConfig,
            routes=[RouteSpec(router=duckdb_router, prefix="/duckdb", tags=["duckdb"])],
        )
        super().__init__(manifest)

    def create_runtime(self) -> DuckDBStorageRuntime:
        return DuckDBStorageRuntime(self.manifest)


# Export the factory instance for entry points
factory = DuckDBStorageFactory()
