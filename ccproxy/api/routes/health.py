"""Health check endpoints for Claude Code Proxy API Server.

Implements modern health check patterns following 2024 best practices:
- /health/live: Liveness probe for Kubernetes (minimal, fast)
- /health/ready: Readiness probe for Kubernetes (critical dependencies)
- /health: Detailed diagnostics (comprehensive status)

Follows IETF Health Check Response Format draft standard.
TODO: health endpoint Content-Type header to only return application/health+json per IETF spec
"""

import asyncio
import functools
import shutil
import time
from datetime import UTC, datetime, timezone
from typing import Any

from fastapi import APIRouter, Response, status
from structlog import get_logger

from ccproxy import __version__
from ccproxy.auth.exceptions import CredentialsExpiredError, CredentialsNotFoundError
from ccproxy.core.async_utils import patched_typing
from ccproxy.services.credentials import CredentialsManager


router = APIRouter()
logger = get_logger(__name__)

# Cache for Claude CLI check results
_claude_cli_cache: tuple[float, tuple[str, dict[str, Any]]] | None = None
_cache_ttl_seconds = 300  # Cache for 5 minutes


async def _check_oauth2_credentials() -> tuple[str, dict[str, Any]]:
    """Check OAuth2 credentials health status.

    Returns:
        Tuple of (status, details) where status is 'pass'/'fail'/'warn'
        Details include token metadata without exposing sensitive data
    """
    try:
        manager = CredentialsManager()
        validation = await manager.validate()

        if validation.valid and not validation.expired:
            # Get token metadata without exposing sensitive information
            credentials = validation.credentials
            oauth_token = credentials.claude_ai_oauth if credentials else None

            details = {
                "auth_status": "valid",
                "credentials_path": str(validation.path) if validation.path else None,
            }

            if oauth_token:
                details.update(
                    {
                        "expiration": oauth_token.expires_at_datetime.isoformat()
                        if oauth_token.expires_at_datetime
                        else None,
                        "subscription_type": oauth_token.subscription_type,
                        "expires_in_hours": str(
                            int(
                                (
                                    oauth_token.expires_at_datetime - datetime.now(UTC)
                                ).total_seconds()
                                / 3600
                            )
                        )
                        if oauth_token.expires_at_datetime
                        else None,
                    }
                )

            return "pass", details
        else:
            # Handle expired credentials
            credentials = validation.credentials
            oauth_token = credentials.claude_ai_oauth if credentials else None

            details = {
                "auth_status": "expired" if validation.expired else "invalid",
                "credentials_path": str(validation.path) if validation.path else None,
            }

            if oauth_token and oauth_token.expires_at_datetime:
                details.update(
                    {
                        "expiration": oauth_token.expires_at_datetime.isoformat(),
                        "subscription_type": oauth_token.subscription_type,
                        "expired_hours_ago": str(
                            int(
                                (
                                    datetime.now(UTC) - oauth_token.expires_at_datetime
                                ).total_seconds()
                                / 3600
                            )
                        )
                        if validation.expired
                        else None,
                    }
                )

            return "warn", details

    except CredentialsNotFoundError:
        return "warn", {
            "auth_status": "not_configured",
            "error": "Claude credentials file not found",
            "credentials_path": None,
        }
    except CredentialsExpiredError:
        return "warn", {
            "auth_status": "expired",
            "error": "Claude credentials have expired",
        }
    except Exception as e:
        return "fail", {
            "auth_status": "error",
            "error": f"Unexpected error: {str(e)}",
        }


@functools.lru_cache(maxsize=1)
def _get_claude_cli_path() -> str | None:
    """Get Claude CLI path with caching. Returns None if not found."""
    return shutil.which("claude")


async def _check_claude_code() -> tuple[str, dict[str, Any]]:
    """Check Claude Code CLI installation and version by running 'claude --version'.

    Results are cached for 5 minutes to avoid repeated subprocess calls.

    Returns:
        Tuple of (status, details) where status is 'pass'/'fail'/'warn'
        Details include CLI version and binary path
    """
    global _claude_cli_cache

    # Check if we have a valid cached result
    current_time = time.time()
    if _claude_cli_cache is not None:
        cache_time, cached_result = _claude_cli_cache
        if current_time - cache_time < _cache_ttl_seconds:
            logger.debug("claude_cli_check_cache_hit")
            return cached_result

    logger.debug("claude_cli_check_cache_miss")

    # First check if claude binary exists in PATH (cached)
    claude_path = _get_claude_cli_path()

    if not claude_path:
        result = (
            "warn",
            {
                "installation_status": "not_found",
                "cli_status": "not_installed",
                "error": "Claude CLI binary not found in PATH",
                "version": None,
                "binary_path": None,
            },
        )
        # Cache the result
        _claude_cli_cache = (current_time, result)
        return result

    try:
        # Run 'claude --version' to get actual version
        process = await asyncio.create_subprocess_exec(
            "claude",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            version_output = stdout.decode().strip()
            # Extract version from output (e.g., "1.0.48 (Claude Code)" -> "1.0.48")
            if version_output:
                import re

                # Try to find a version pattern (e.g., "1.0.48", "v2.1.0")
                version_match = re.search(
                    r"\b(?:v)?(\d+\.\d+(?:\.\d+)?)\b", version_output
                )
                if version_match:
                    version = version_match.group(1)
                else:
                    # Fallback: take the first part if no version pattern found
                    parts = version_output.split()
                    version = parts[0] if parts else "unknown"
            else:
                version = "unknown"

            result = (
                "pass",
                {
                    "installation_status": "found",
                    "cli_status": "available",
                    "version": version,
                    "binary_path": claude_path,
                    "version_output": version_output,
                },
            )
            # Cache the result
            _claude_cli_cache = (current_time, result)
            return result
        else:
            # Binary exists but --version failed
            error_output = stderr.decode().strip() if stderr else "Unknown error"
            result = (
                "warn",
                {
                    "installation_status": "found_with_issues",
                    "cli_status": "binary_found_but_errors",
                    "error": f"'claude --version' failed: {error_output}",
                    "version": None,
                    "binary_path": claude_path,
                    "return_code": str(process.returncode),
                },
            )
            # Cache the result
            _claude_cli_cache = (current_time, result)
            return result

    except TimeoutError:
        result = (
            "warn",
            {
                "installation_status": "found_with_issues",
                "cli_status": "timeout",
                "error": "Timeout running 'claude --version'",
                "version": None,
                "binary_path": claude_path,
            },
        )
        # Cache the result
        _claude_cli_cache = (current_time, result)
        return result
    except Exception as e:
        result = (
            "fail",
            {
                "installation_status": "error",
                "cli_status": "error",
                "error": f"Unexpected error running 'claude --version': {str(e)}",
                "version": None,
                "binary_path": claude_path,
            },
        )
        # Cache the result
        _claude_cli_cache = (current_time, result)
        return result


async def _check_claude_sdk() -> tuple[str, dict[str, Any]]:
    """Check Claude SDK installation and version.

    Returns:
        Tuple of (status, details) where status is 'pass'/'fail'/'warn'
        Details include SDK version and availability
    """
    try:
        # Try to import Claude Code SDK
        with patched_typing():
            from claude_code_sdk import __version__ as sdk_version

        return "pass", {
            "installation_status": "found",
            "sdk_status": "available",
            "version": sdk_version,
            "import_successful": True,
        }

    except ImportError as e:
        return "warn", {
            "installation_status": "not_found",
            "sdk_status": "not_installed",
            "error": f"Claude SDK not available: {str(e)}",
            "version": None,
            "import_successful": False,
        }
    except Exception as e:
        return "fail", {
            "installation_status": "error",
            "sdk_status": "error",
            "error": f"Unexpected error checking SDK: {str(e)}",
            "version": None,
            "import_successful": False,
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

    # Check OAuth credentials, CLI, and SDK separately
    oauth_status, oauth_details = await _check_oauth2_credentials()
    cli_status, cli_details = await _check_claude_code()
    sdk_status, sdk_details = await _check_claude_sdk()

    # Service is ready if no check returns "fail"
    # "warn" statuses (missing credentials/CLI/SDK) don't prevent readiness
    if oauth_status == "fail" or cli_status == "fail" or sdk_status == "fail":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        failed_components = []

        if oauth_status == "fail":
            failed_components.append("oauth2_credentials")
        if cli_status == "fail":
            failed_components.append("claude_cli")
        if sdk_status == "fail":
            failed_components.append("claude_sdk")

        return {
            "status": "fail",
            "version": __version__,
            "output": f"Critical dependency error: {', '.join(failed_components)}",
            "checks": {
                "oauth2_credentials": [
                    {
                        "status": oauth_status,
                        "output": oauth_details.get("error", "OAuth credentials error"),
                    }
                ],
                "claude_cli": [
                    {
                        "status": cli_status,
                        "output": cli_details.get("error", "Claude CLI error"),
                    }
                ],
                "claude_sdk": [
                    {
                        "status": sdk_status,
                        "output": sdk_details.get("error", "Claude SDK error"),
                    }
                ],
            },
        }

    return {
        "status": "pass",
        "version": __version__,
        "output": "Service is ready to accept traffic",
        "checks": {
            "oauth2_credentials": [
                {
                    "status": oauth_status,
                    "output": f"OAuth credentials: {oauth_details.get('auth_status', 'unknown')}",
                }
            ],
            "claude_cli": [
                {
                    "status": cli_status,
                    "output": f"Claude CLI: {cli_details.get('cli_status', 'unknown')}",
                }
            ],
            "claude_sdk": [
                {
                    "status": sdk_status,
                    "output": f"Claude SDK: {sdk_details.get('sdk_status', 'unknown')}",
                }
            ],
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
    oauth_status, oauth_details = await _check_oauth2_credentials()
    cli_status, cli_details = await _check_claude_code()
    sdk_status, sdk_details = await _check_claude_sdk()

    # Determine overall status - prioritize failures, then warnings
    overall_status = "pass"
    if oauth_status == "fail" or cli_status == "fail" or sdk_status == "fail":
        overall_status = "fail"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif oauth_status == "warn" or cli_status == "warn" or sdk_status == "warn":
        overall_status = "warn"
        response.status_code = status.HTTP_200_OK

    current_time = datetime.now(UTC).isoformat()

    return {
        "status": overall_status,
        "version": __version__,
        "serviceId": "claude-code-proxy",
        "description": "Claude Code Proxy API Server",
        "time": current_time,
        "checks": {
            "oauth2_credentials": [
                {
                    "componentId": "oauth2-credentials",
                    "componentType": "authentication",
                    "status": oauth_status,
                    "time": current_time,
                    "output": f"OAuth2 credentials: {oauth_details.get('auth_status', 'unknown')}",
                    **oauth_details,
                }
            ],
            "claude_cli": [
                {
                    "componentId": "claude-cli",
                    "componentType": "external_dependency",
                    "status": cli_status,
                    "time": current_time,
                    "output": f"Claude CLI: {cli_details.get('cli_status', 'unknown')}",
                    **cli_details,
                }
            ],
            "claude_sdk": [
                {
                    "componentId": "claude-sdk",
                    "componentType": "python_package",
                    "status": sdk_status,
                    "time": current_time,
                    "output": f"Claude SDK: {sdk_details.get('sdk_status', 'unknown')}",
                    **sdk_details,
                }
            ],
            "proxy_service": [
                {
                    "componentId": "proxy-service",
                    "componentType": "service",
                    "status": "pass",
                    "time": current_time,
                    "output": "Proxy service operational",
                    "version": __version__,
                }
            ],
        },
    }
