"""Health check endpoints for CCProxy API Server.

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
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from structlog import get_logger

from ccproxy import __version__


router = APIRouter()
logger = get_logger(__name__)


class ClaudeCliStatus(str, Enum):
    """Claude CLI status enumeration."""

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    BINARY_FOUND_BUT_ERRORS = "binary_found_but_errors"
    TIMEOUT = "timeout"
    ERROR = "error"


class CodexCliStatus(str, Enum):
    """Codex CLI status enumeration."""

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    BINARY_FOUND_BUT_ERRORS = "binary_found_but_errors"
    TIMEOUT = "timeout"
    ERROR = "error"


class ClaudeCliInfo(BaseModel):
    """Claude CLI information with structured data."""

    status: ClaudeCliStatus
    version: str | None = None
    binary_path: str | None = None
    version_output: str | None = None
    error: str | None = None
    return_code: str | None = None


class CodexCliInfo(BaseModel):
    """Codex CLI information with structured data."""

    status: CodexCliStatus
    version: str | None = None
    binary_path: str | None = None
    version_output: str | None = None
    error: str | None = None
    return_code: str | None = None


# Cache for Claude CLI check results
_claude_cli_cache: tuple[float, tuple[str, dict[str, Any]]] | None = None
# Cache for Codex CLI check results
_codex_cli_cache: tuple[float, tuple[str, dict[str, Any]]] | None = None
_cache_ttl_seconds = 300  # Cache for 5 minutes


async def _check_oauth2_credentials() -> tuple[str, dict[str, Any]]:
    """Check OAuth2 credentials health status.

    This function is deprecated as authentication is now handled by individual plugins.
    Returns a pass status for backward compatibility.

    Returns:
        Tuple of (status, details) where status is 'pass'
        Details indicate auth is plugin-managed
    """
    # Authentication is now handled by individual plugins
    return "pass", {
        "auth_status": "plugin_managed",
        "message": "Authentication handled by plugins",
    }


@functools.lru_cache(maxsize=1)
def _get_claude_cli_path() -> str | None:
    """Get Claude CLI path with caching. Returns None if not found."""
    return shutil.which("claude")


def _get_codex_cli_path() -> str | None:
    """Get Codex CLI path with caching. Returns None if not found."""
    return shutil.which("codex")


async def check_claude_code() -> tuple[str, dict[str, Any]]:
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
            logger.debug("claude_cli_check_cache_hit", category="cache")
            return cached_result

    logger.debug("claude_cli_check_cache_miss", category="cache")

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
    except (OSError, PermissionError) as e:
        logger.error("claude_cli_check_io_failed", error=str(e), exc_info=e)
        result = (
            "fail",
            {
                "installation_status": "error",
                "cli_status": "error",
                "error": f"Process execution error: {str(e)}",
                "version": None,
                "binary_path": claude_path,
            },
        )
        # Cache the result
        _claude_cli_cache = (current_time, result)
        return result
    except Exception as e:
        logger.error("claude_cli_check_failed", error=str(e), exc_info=e)
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


async def get_claude_cli_info() -> ClaudeCliInfo:
    """Get Claude CLI information as a structured Pydantic model.

    Returns:
        ClaudeCliInfo: Structured information about Claude CLI installation and status
    """
    cli_status, cli_details = await check_claude_code()

    # Map the status to our enum values
    if cli_status == "pass":
        status_value = ClaudeCliStatus.AVAILABLE
    elif cli_details.get("cli_status") == "not_installed":
        status_value = ClaudeCliStatus.NOT_INSTALLED
    elif cli_details.get("cli_status") == "binary_found_but_errors":
        status_value = ClaudeCliStatus.BINARY_FOUND_BUT_ERRORS
    elif cli_details.get("cli_status") == "timeout":
        status_value = ClaudeCliStatus.TIMEOUT
    else:
        status_value = ClaudeCliStatus.ERROR

    return ClaudeCliInfo(
        status=status_value,
        version=cli_details.get("version"),
        binary_path=cli_details.get("binary_path"),
        version_output=cli_details.get("version_output"),
        error=cli_details.get("error"),
        return_code=cli_details.get("return_code"),
    )


async def check_codex_cli() -> tuple[str, dict[str, Any]]:
    """Check Codex CLI installation and version by running 'codex --version'.
    Results are cached for 5 minutes to avoid repeated subprocess calls.
    Returns:
        Tuple of (status, details) where status is 'pass'/'fail'/'warn'
        Details include CLI version and binary path
    """
    global _codex_cli_cache
    # Check if we have a valid cached result
    current_time = time.time()
    if _codex_cli_cache is not None:
        cache_time, cached_result = _codex_cli_cache
        if current_time - cache_time < _cache_ttl_seconds:
            logger.debug("codex_cli_check_cache_hit", category="cache")
            return cached_result

    logger.debug("codex_cli_check_cache_miss", category="cache")

    # First check if codex binary exists in PATH (cached)
    codex_path = _get_codex_cli_path()
    if not codex_path:
        result = (
            "warn",
            {
                "installation_status": "not_found",
                "cli_status": "not_installed",
                "error": "Codex CLI binary not found in PATH",
                "version": None,
                "binary_path": None,
            },
        )
        # Cache the result
        _codex_cli_cache = (current_time, result)
        return result

    try:
        # Run 'codex --version' to get actual version
        process = await asyncio.create_subprocess_exec(
            "codex",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            version_output = stdout.decode().strip()
            # Extract version from output (e.g., "codex 0.21.0" -> "0.21.0")
            if version_output:
                import re

                # Try to find a version pattern (e.g., "0.21.0", "v1.0.0")
                version_match = re.search(
                    r"\b(?:v)?(\d+\.\d+(?:\.\d+)?)\b", version_output
                )
                if version_match:
                    version = version_match.group(1)
                else:
                    # Fallback: take the last part if no version pattern found
                    parts = version_output.split()
                    version = parts[-1] if parts else "unknown"
            else:
                version = "unknown"

            result = (
                "pass",
                {
                    "installation_status": "found",
                    "cli_status": "available",
                    "version": version,
                    "binary_path": codex_path,
                    "version_output": version_output,
                },
            )
            # Cache the result
            _codex_cli_cache = (current_time, result)
            return result
        else:
            # Binary exists but --version failed
            error_output = stderr.decode().strip() if stderr else "Unknown error"
            result = (
                "warn",
                {
                    "installation_status": "found_with_issues",
                    "cli_status": "binary_found_but_errors",
                    "error": f"'codex --version' failed: {error_output}",
                    "version": None,
                    "binary_path": codex_path,
                    "return_code": str(process.returncode),
                },
            )
            # Cache the result
            _codex_cli_cache = (current_time, result)
            return result

    except TimeoutError:
        result = (
            "warn",
            {
                "installation_status": "found_with_issues",
                "cli_status": "timeout",
                "error": "Timeout running 'codex --version'",
                "version": None,
                "binary_path": codex_path,
            },
        )
        # Cache the result
        _codex_cli_cache = (current_time, result)
        return result

    except (OSError, PermissionError) as e:
        logger.error("codex_cli_check_io_failed", error=str(e), exc_info=e)
        result = (
            "fail",
            {
                "installation_status": "error",
                "cli_status": "error",
                "error": f"Process execution error: {str(e)}",
                "version": None,
                "binary_path": codex_path,
            },
        )
        # Cache the result
        _codex_cli_cache = (current_time, result)
        return result
    except Exception as e:
        logger.error("codex_cli_check_failed", error=str(e), exc_info=e)
        result = (
            "fail",
            {
                "installation_status": "error",
                "cli_status": "error",
                "error": f"Unexpected error running 'codex --version': {str(e)}",
                "version": None,
                "binary_path": codex_path,
            },
        )
        # Cache the result
        _codex_cli_cache = (current_time, result)
        return result


async def get_codex_cli_info() -> CodexCliInfo:
    """Get Codex CLI information as a structured Pydantic model.
    Returns:
        CodexCliInfo: Structured information about Codex CLI installation and status
    """
    cli_status, cli_details = await check_codex_cli()

    # Map the status to our enum values
    if cli_status == "pass":
        status_value = CodexCliStatus.AVAILABLE
    elif cli_details.get("cli_status") == "not_installed":
        status_value = CodexCliStatus.NOT_INSTALLED
    elif cli_details.get("cli_status") == "binary_found_but_errors":
        status_value = CodexCliStatus.BINARY_FOUND_BUT_ERRORS
    elif cli_details.get("cli_status") == "timeout":
        status_value = CodexCliStatus.TIMEOUT
    else:
        status_value = CodexCliStatus.ERROR

    return CodexCliInfo(
        status=status_value,
        version=cli_details.get("version"),
        binary_path=cli_details.get("binary_path"),
        version_output=cli_details.get("version_output"),
        error=cli_details.get("error"),
        return_code=cli_details.get("return_code"),
    )


async def _check_claude_sdk() -> tuple[str, dict[str, Any]]:
    """Check Claude SDK installation and version.

    Note: Claude SDK is now only available as a plugin and not part of core.
    This check always returns 'warn' status to indicate SDK should be accessed
    via the plugin system.

    Returns:
        Tuple of (status, details) where status is 'pass'/'fail'/'warn'
        Details include SDK version and availability
    """
    # Claude SDK has been moved to plugin system and is not available in core
    return "warn", {
        "installation_status": "plugin_only",
        "sdk_status": "moved_to_plugin",
        "error": "Claude SDK is now available via the claude_sdk plugin only",
        "version": None,
        "import_successful": False,
        "plugin_name": "claude_sdk",
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

    logger.debug("liveness_probe_request")

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

    logger.debug("readiness_probe_request")

    # Check OAuth credentials, CLI, and SDK separately
    oauth_status, oauth_details = await _check_oauth2_credentials()
    cli_status, cli_details = await check_claude_code()
    codex_cli_status, codex_cli_details = await check_codex_cli()
    sdk_status, sdk_details = await _check_claude_sdk()

    # Service is ready if no check returns "fail"
    # "warn" statuses (missing credentials/CLI/SDK) don't prevent readiness
    if (
        oauth_status == "fail"
        or cli_status == "fail"
        or codex_cli_status == "fail"
        or sdk_status == "fail"
    ):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        failed_components = []

        if oauth_status == "fail":
            failed_components.append("oauth2_credentials")
        if cli_status == "fail":
            failed_components.append("claude_cli")
        if codex_cli_status == "fail":
            failed_components.append("codex_cli")
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
                "codex_cli": [
                    {
                        "status": codex_cli_status,
                        "output": codex_cli_details.get("error", "Codex CLI error"),
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
            "codex_cli": [
                {
                    "status": codex_cli_status,
                    "output": f"Codex CLI: {codex_cli_details.get('cli_status', 'unknown')}",
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

    logger.debug("detailed_health_check_request")

    # Perform all health checks
    oauth_status, oauth_details = await _check_oauth2_credentials()
    cli_status, cli_details = await check_claude_code()
    codex_cli_status, codex_cli_details = await check_codex_cli()
    sdk_status, sdk_details = await _check_claude_sdk()

    # Determine overall status - prioritize failures, then warnings
    overall_status = "pass"
    if (
        oauth_status == "fail"
        or cli_status == "fail"
        or codex_cli_status == "fail"
        or sdk_status == "fail"
    ):
        overall_status = "fail"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif (
        oauth_status == "warn"
        or cli_status == "warn"
        or codex_cli_status == "warn"
        or sdk_status == "warn"
    ):
        overall_status = "warn"
        response.status_code = status.HTTP_200_OK

    current_time = datetime.now(UTC).isoformat()

    return {
        "status": overall_status,
        "version": __version__,
        "serviceId": "claude-code-proxy",
        "description": "CCProxy API Server",
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
            "codex_cli": [
                {
                    "componentId": "codex-cli",
                    "componentType": "external_dependency",
                    "status": codex_cli_status,
                    "time": current_time,
                    "output": f"Codex CLI: {codex_cli_details.get('cli_status', 'unknown')}",
                    **codex_cli_details,
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
