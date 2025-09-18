from __future__ import annotations

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    RouteSpec,
    SystemPluginFactory,
    SystemPluginRuntime,
)

from .config import AnalyticsPluginConfig


logger = get_plugin_logger()


class AnalyticsRuntime(SystemPluginRuntime):
    async def _on_initialize(self) -> None:
        # Ensure AccessLog model is registered and table exists on the engine.
        from sqlmodel import SQLModel

        # Import models to register with SQLModel metadata
        try:
            from . import models as _models  # noqa: F401
        except Exception as e:  # pragma: no cover - defensive
            logger.error("analytics_models_import_failed", error=str(e))
            raise

        # Assert model registration in metadata
        table = SQLModel.metadata.tables.get("access_logs")
        if table is None:
            logger.error("access_logs_table_not_in_metadata")
            raise RuntimeError("AccessLog model not registered in SQLModel metadata")

        # Try to get storage engine via plugin registry service
        engine = None
        try:
            registry = self.context.get("plugin_registry") if self.context else None
            if registry:
                storage = registry.get_service("log_storage")
                engine = getattr(storage, "_engine", None)

            # Fallback to app.state if needed
            if (engine is None) and self.context and self.context.get("app"):
                app = self.context["app"]
                storage = getattr(app.state, "log_storage", None)
                engine = getattr(storage, "_engine", None)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("analytics_engine_lookup_failed", error=str(e))

        # If we have an engine, assert table is created (idempotent create_all)
        if engine is not None:
            try:
                SQLModel.metadata.create_all(engine)
                logger.debug("analytics_table_ready", table="access_logs")
            except Exception as e:
                logger.error("analytics_table_create_failed", error=str(e))
                raise
        else:
            logger.warning(
                "analytics_no_engine_available",
                message="Storage engine not available during analytics init; table creation skipped",
            )

        # Register ingest service for access_log hook to call
        try:
            if self.context:
                registry = self.context.get("plugin_registry")
                storage = None
                if registry:
                    # Get storage service without importing DuckDB-specific classes
                    storage = registry.get_service("log_storage")
                if not storage and self.context.get("app"):
                    storage = getattr(self.context["app"].state, "log_storage", None)

                if storage:
                    engine = getattr(storage, "_engine", None)
                else:
                    engine = None

                if engine is not None:
                    from .ingest import AnalyticsIngestService

                    ingest_service = AnalyticsIngestService(engine)
                    if registry:
                        registry.register_service(
                            "analytics_ingest", ingest_service, self.manifest.name
                        )
                        logger.debug("analytics_ingest_service_registered")
                else:
                    logger.warning(
                        "analytics_ingest_registration_skipped",
                        reason="no_engine_available",
                    )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("analytics_ingest_registration_failed", error=str(e))

        logger.debug("analytics_plugin_initialized")


class AnalyticsFactory(SystemPluginFactory):
    def __init__(self) -> None:
        from .routes import router as analytics_router

        manifest = PluginManifest(
            name="analytics",
            version="1.0.0",
            description="Logs query, analytics, and streaming endpoints",
            is_provider=False,
            config_class=AnalyticsPluginConfig,
            provides=["analytics_ingest"],
            dependencies=["duckdb_storage"],
            routes=[RouteSpec(router=analytics_router, prefix="/logs", tags=["logs"])],
        )
        super().__init__(manifest)

    def create_runtime(self) -> AnalyticsRuntime:
        return AnalyticsRuntime(self.manifest)


factory = AnalyticsFactory()
