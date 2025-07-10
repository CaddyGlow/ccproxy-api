"""Authentication endpoints for the API."""

from fastapi import APIRouter

# Create the router for auth endpoints
router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/status")
async def auth_status():
    """Get authentication status."""
    return {"status": "auth endpoint available"}