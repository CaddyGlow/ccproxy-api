from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request


router = APIRouter()


def _get_storage(request: Request) -> Any:
    storage = getattr(request.app.state, "log_storage", None)
    if not storage:
        # Backward-compat alias
        storage = getattr(request.app.state, "duckdb_storage", None)
    return storage


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    storage = _get_storage(request)
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    return cast(dict[str, Any], await storage.health_check())


@router.get("/status")
async def status(request: Request) -> dict[str, Any]:
    storage = _get_storage(request)
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    health = cast(dict[str, Any], await storage.health_check())

    # Include basic plugin/service context when available
    plugin_info: dict[str, Any] = {
        "plugin": "duckdb_storage",
        "service_registered": False,
    }

    try:
        if hasattr(request.app.state, "plugin_registry"):
            registry = request.app.state.plugin_registry
            plugin_info["service_registered"] = registry.has_service("log_storage")
    except Exception:
        pass

    return {
        "health": health,
        **plugin_info,
    }
