"""Health check endpoints for Claude Code Proxy API Server."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ccproxy.api.dependencies import get_claude_service, get_proxy_service
from ccproxy.core.logging import get_logger
from ccproxy.services.claude_sdk_service import ClaudeSDKService
from ccproxy.services.proxy_service import ProxyService


router = APIRouter()
logger = get_logger(__name__)


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint.

    Returns:
        Health status information
    """
    logger.debug("Health check request")
    return {
        "status": "healthy",
        "service": "claude-code-proxy",
        "version": "0.1.0",
    }


@router.get("/health/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness check endpoint.

    Returns:
        Readiness status information
    """
    logger.debug("Readiness check request")
    return {
        "status": "ready",
        "service": "claude-code-proxy",
        "timestamp": "2024-01-01T00:00:00Z",
    }


@router.get("/health/live")
async def liveness_check() -> dict[str, Any]:
    """Liveness check endpoint.

    Returns:
        Liveness status information
    """
    logger.debug("Liveness check request")
    return {
        "status": "alive",
        "service": "claude-code-proxy",
        "uptime": "0s",
    }


@router.get("/health/claude")
async def claude_health_check(
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> dict[str, Any]:
    """Health check for Claude SDK service.

    Args:
        claude_service: Injected Claude SDK service dependency

    Returns:
        Claude SDK health status

    Raises:
        HTTPException: If Claude SDK is not healthy
    """
    try:
        logger.debug("Claude SDK health check request")
        status = await claude_service.health_check()
        return {
            "status": "healthy",
            "service": "claude-sdk",
            "details": status,
        }

    except Exception as e:
        logger.error(f"Claude SDK health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Claude SDK unhealthy: {e}") from e


@router.get("/health/proxy")
async def proxy_health_check(
    proxy_service: ProxyService = Depends(get_proxy_service),
) -> dict[str, Any]:
    """Health check for proxy service.

    Args:
        proxy_service: Injected proxy service dependency

    Returns:
        Proxy service health status

    Raises:
        HTTPException: If proxy service is not healthy
    """
    try:
        logger.debug("Proxy service health check request")
        status = await proxy_service.health_check()
        return {
            "status": "healthy",
            "service": "proxy",
            "details": status,
        }

    except Exception as e:
        logger.error(f"Proxy service health check failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Proxy service unhealthy: {e}"
        ) from e


@router.get("/health/detailed")
async def detailed_health_check(
    claude_service: ClaudeSDKService = Depends(get_claude_service),
    proxy_service: ProxyService = Depends(get_proxy_service),
) -> dict[str, Any]:
    """Detailed health check for all services.

    Args:
        claude_service: Injected Claude SDK service dependency
        proxy_service: Injected proxy service dependency

    Returns:
        Detailed health status for all services
    """
    logger.debug("Detailed health check request")

    health_status = {
        "status": "healthy",
        "service": "claude-code-proxy",
        "version": "0.1.0",
        "checks": {},
    }

    # Check Claude SDK
    try:
        claude_status = await claude_service.health_check()
        health_status["checks"]["claude"] = {
            "status": "healthy",
            "details": claude_status,
        }
    except Exception as e:
        logger.warning(f"Claude SDK health check failed: {e}")
        health_status["checks"]["claude"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check proxy service
    try:
        proxy_status = await proxy_service.health_check()
        health_status["checks"]["proxy"] = {
            "status": "healthy",
            "details": proxy_status,
        }
    except Exception as e:
        logger.warning(f"Proxy service health check failed: {e}")
        health_status["checks"]["proxy"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    return health_status
