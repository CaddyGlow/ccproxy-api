"""Health check endpoints for CCProxy API Server.

Implements modern health check patterns following 2024 best practices:
- /health/live: Liveness probe for Kubernetes (minimal, fast)
- /health/ready: Readiness probe for Kubernetes (critical dependencies)
- /health: Detailed diagnostics (comprehensive status)

Follows IETF Health Check Response Format draft standard.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse

from ccproxy.core import __version__
from ccproxy.core.logging import get_logger


class HealthJSONResponse(JSONResponse):
    media_type = "application/health+json"


def _health_responses(description: str) -> dict[int | str, dict[str, Any]]:
    return {
        200: {
            "description": description,
            "content": {"application/health+json": {"schema": {"type": "object"}}},
        }
    }


router = APIRouter(default_response_class=HealthJSONResponse)
logger = get_logger(__name__)

# Authentication and CLI health are managed by provider plugins; no core CLI checks


@router.get(
    "/health/live",
    response_class=HealthJSONResponse,
    responses=_health_responses("Liveness probe result"),
)
async def liveness_probe(response: Response) -> dict[str, Any]:
    """Liveness probe for Kubernetes.

    Minimal health check that only verifies the application process is running.
    Used by Kubernetes to determine if the pod should be restarted.

    Returns:
        Simple health status following IETF health check format
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Content-Type"] = "application/health+json"

    logger.debug("liveness_probe_request")

    return {
        "status": "pass",
        "version": __version__,
        "output": "Application process is running",
    }


@router.get(
    "/health/ready",
    response_class=HealthJSONResponse,
    responses=_health_responses("Readiness probe result"),
)
async def readiness_probe(response: Response) -> dict[str, Any]:
    """Readiness probe for Kubernetes.

    Checks critical dependencies to determine if the service is ready to accept traffic.
    Used by Kubernetes to determine if the pod should receive traffic.

    Returns:
        Readiness status with critical dependency checks
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Content-Type"] = "application/health+json"

    logger.debug("readiness_probe_request")

    # Core readiness only checks application availability; plugins provide their own health
    return {
        "status": "pass",
        "version": __version__,
        "output": "Service is ready to accept traffic",
    }


@router.get(
    "/health",
    response_class=HealthJSONResponse,
    responses=_health_responses("Detailed health diagnostics"),
)
async def detailed_health_check(response: Response) -> dict[str, Any]:
    """Comprehensive health check for diagnostics and monitoring.

    Provides detailed status of core service only. Provider/plugin-specific
    health, including CLI availability, is reported by each plugin's health endpoint.

    Returns:
        Detailed health status following IETF health check format
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Content-Type"] = "application/health+json"

    logger.debug("detailed_health_check_request")

    overall_status = "pass"
    response.status_code = status.HTTP_200_OK

    current_time = datetime.now(UTC).isoformat()

    return {
        "status": overall_status,
        "version": __version__,
        "serviceId": "claude-code-proxy",
        "description": "CCProxy API Server",
        "time": current_time,
        "checks": {
            "service_container": [
                {
                    "componentId": "service-container",
                    "componentType": "service",
                    "status": "pass",
                    "time": current_time,
                    "output": "Service container operational",
                    "version": __version__,
                }
            ],
        },
    }
