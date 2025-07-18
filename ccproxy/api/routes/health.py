"""Health check endpoints for Claude Code Proxy API Server.

Implements modern health check patterns following 2024 best practices:
- /health/live: Liveness probe for Kubernetes (minimal, fast)
- /health/ready: Readiness probe for Kubernetes (critical dependencies)
- /health: Detailed diagnostics (comprehensive status)

Follows IETF Health Check Response Format draft standard.
"""

from datetime import UTC, datetime, timezone
from typing import Any

from fastapi import APIRouter, Response, status
from structlog import get_logger

from ccproxy import __version__
from ccproxy.services.credentials import CredentialsManager


router = APIRouter()
logger = get_logger(__name__)


async def _check_claude_sdk() -> tuple[str, dict[str, Any]]:
    """Check Claude SDK health status.

    Returns:
        Tuple of (status, details) where status is 'pass'/'fail'/'warn'
    """
    try:
        manager = CredentialsManager()
        validation = await manager.validate()

        if validation.valid and not validation.expired:
            return "pass", {
                "auth_status": "valid",
                "credentials_path": str(validation.path) if validation.path else None,
            }
        else:
            return "warn", {
                "auth_status": "invalid_or_expired",
                "credentials_path": str(validation.path) if validation.path else None,
            }
    except Exception as e:
        return "fail", {
            "auth_status": "error",
            "error": str(e),
        }


@router.get("/health/live")
async def liveness_probe(response: Response) -> dict[str, Any]:
    """Liveness probe for Kubernetes.

    Minimal health check that only verifies the application process is running.
    Used by Kubernetes to determine if the pod should be restarted.

    Returns:
        Simple health status following IETF health check format
    """
    # Add cache control headers as per best practices
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Content-Type"] = "application/health+json"

    logger.debug("Liveness probe request")

    return {
        "status": "pass",
        "version": __version__,
        "output": "Application process is running",
    }


@router.get("/health/ready")
async def readiness_probe(response: Response) -> dict[str, Any]:
    """Readiness probe for Kubernetes.

    Checks critical dependencies to determine if the service is ready to accept traffic.
    Used by Kubernetes to determine if the pod should receive traffic.

    Returns:
        Readiness status with critical dependency checks
    """
    # Add cache control headers
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Content-Type"] = "application/health+json"

    logger.debug("Readiness probe request")

    # Check critical dependencies only
    claude_status, claude_details = await _check_claude_sdk()

    # Service is ready if Claude SDK is accessible (pass or warn)
    # Only fail readiness if Claude SDK is completely unavailable
    if claude_status == "fail":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "fail",
            "version": __version__,
            "output": "Critical dependency unavailable",
            "checks": {
                "claude_sdk": [
                    {
                        "status": claude_status,
                        "output": claude_details.get("error", "Claude SDK unavailable"),
                    }
                ]
            },
        }

    return {
        "status": "pass",
        "version": __version__,
        "output": "Service is ready to accept traffic",
        "checks": {
            "claude_sdk": [
                {
                    "status": claude_status,
                    "output": "Claude SDK accessible",
                }
            ]
        },
    }


@router.get("/health")
async def detailed_health_check(response: Response) -> dict[str, Any]:
    """Comprehensive health check for diagnostics and monitoring.

    Provides detailed status of all services and dependencies.
    Used by monitoring dashboards, debugging, and operations teams.

    Returns:
        Detailed health status following IETF health check format
    """
    # Add cache control headers
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Content-Type"] = "application/health+json"

    logger.debug("Detailed health check request")

    # Perform all health checks
    claude_status, claude_details = await _check_claude_sdk()

    # Determine overall status
    overall_status = "pass"
    if claude_status == "fail":
        overall_status = "fail"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif claude_status == "warn":
        overall_status = "warn"
        response.status_code = status.HTTP_200_OK

    return {
        "status": overall_status,
        "version": __version__,
        "serviceId": "claude-code-proxy",
        "description": "Claude Code Proxy API Server",
        "time": datetime.now(UTC).isoformat(),
        "checks": {
            "claude_sdk": [
                {
                    "componentId": "claude-sdk",
                    "componentType": "service",
                    "status": claude_status,
                    "time": datetime.now(UTC).isoformat(),
                    "output": f"Claude SDK status: {claude_details.get('auth_status', 'unknown')}",
                    **claude_details,
                }
            ],
            "proxy_service": [
                {
                    "componentId": "proxy-service",
                    "componentType": "service",
                    "status": "pass",
                    "time": datetime.now(UTC).isoformat(),
                    "output": "Proxy service operational",
                    "version": __version__,
                }
            ],
        },
    }
