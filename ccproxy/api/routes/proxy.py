"""Proxy endpoints for Claude Code Proxy API Server."""

from fastapi import APIRouter


# Create the router for proxy endpoints
router = APIRouter(prefix="/proxy", tags=["proxy"])


@router.get("/status")
async def proxy_status() -> dict[str, str]:
    """Get proxy status."""
    return {"status": "proxy endpoint available"}
