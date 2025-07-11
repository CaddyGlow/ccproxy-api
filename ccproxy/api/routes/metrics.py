"""Metrics endpoints for Claude Code Proxy API Server."""

from fastapi import APIRouter


# Create the router for metrics endpoints
router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/status")
async def metrics_status() -> dict[str, str]:
    """Get metrics status."""
    return {"status": "metrics endpoint available"}
