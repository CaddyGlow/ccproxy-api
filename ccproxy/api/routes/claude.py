"""Claude SDK endpoints for Claude Code Proxy API Server."""

from fastapi import APIRouter


# Create the router for Claude SDK endpoints
router = APIRouter(prefix="/claude", tags=["claude-sdk"])


@router.get("/status")
async def claude_status():
    """Get Claude SDK status."""
    return {"status": "claude sdk endpoint available"}
