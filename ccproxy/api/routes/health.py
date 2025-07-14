"""Health check endpoints for Claude Code Proxy API Server."""

import asyncio
from typing import Any

from fastapi import APIRouter

from ccproxy import __version__
from ccproxy.core.logging import get_structlog_logger
from ccproxy.services.credentials import CredentialsManager


router = APIRouter()
logger = get_structlog_logger(__name__)


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Basic health check endpoint.

    Returns:
        Health status information
    """
    logger.debug(
        "Health check request",
        service="claude-code-proxy",
        check_type="basic",
    )
    return {
        "status": "healthy",
        "service": "claude-code-proxy",
        "version": __version__,
    }


@router.get("/health/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness check endpoint.

    Returns:
        Readiness status information
    """
    logger.debug(
        "Readiness check request",
        service="claude-code-proxy",
        check_type="readiness",
    )
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
    logger.debug(
        "Liveness check request",
        service="claude-code-proxy",
        check_type="liveness",
    )
    return {
        "status": "alive",
        "service": "claude-code-proxy",
        "uptime": "0s",
    }


@router.get("/health/claude")
async def claude_health_check() -> dict[str, Any]:
    """Health check for Claude SDK service.

    Returns:
        Claude SDK health status including auth status
    """
    try:
        logger.debug(
            "Claude SDK health check request",
            service="claude-sdk",
            check_type="individual",
        )

        # Check credentials status
        manager = CredentialsManager()
        validation = await manager.validate()

        auth_status = (
            "valid" if validation.valid and not validation.expired else "invalid"
        )

        return {
            "status": "healthy",
            "service": "claude-sdk",
            "auth_status": auth_status,
            "credentials_path": str(validation.path) if validation.path else None,
        }

    except Exception as e:
        logger.error(
            "Claude SDK health check failed",
            error=str(e),
            error_type=type(e).__name__,
            service="claude-sdk",
        )
        return {
            "status": "unhealthy",
            "service": "claude-sdk",
            "auth_status": "error",
            "error": str(e),
        }


@router.get("/health/proxy")
async def proxy_health_check() -> dict[str, Any]:
    """Health check for proxy service.

    Returns:
        Proxy service health status
    """
    logger.debug(
        "Proxy service health check request",
        service="proxy",
        check_type="individual",
    )
    return {
        "status": "healthy",
        "service": "proxy",
        "version": __version__,
    }


@router.get("/health/detailed")
async def detailed_health_check() -> dict[str, Any]:
    """Detailed health check for all services.

    Returns:
        Detailed health status for all services
    """
    logger.debug(
        "Detailed health check request",
        service="claude-code-proxy",
        check_type="detailed",
    )

    health_status: dict[str, Any] = {
        "status": "healthy",
        "service": "claude-code-proxy",
        "version": __version__,
        "checks": {},
    }

    # Check Claude SDK and auth
    try:
        manager = CredentialsManager()
        validation = await manager.validate()

        auth_status = (
            "valid" if validation.valid and not validation.expired else "invalid"
        )

        health_status["checks"]["claude"] = {
            "status": "healthy" if auth_status == "valid" else "degraded",
            "auth_status": auth_status,
            "credentials_path": str(validation.path) if validation.path else None,
        }

        if auth_status != "valid":
            health_status["status"] = "degraded"

    except Exception as e:
        logger.warning(
            "Claude SDK health check failed in detailed check",
            error=str(e),
            error_type=type(e).__name__,
            service="claude-sdk",
            check_type="detailed",
        )
        health_status["checks"]["claude"] = {
            "status": "unhealthy",
            "auth_status": "error",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check proxy service
    health_status["checks"]["proxy"] = {
        "status": "healthy",
        "version": __version__,
    }

    return health_status
