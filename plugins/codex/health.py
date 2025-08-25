"""Codex health check implementation."""

from typing import Any, Literal

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins.protocol import HealthCheckResult

from .config import CodexSettings
from .detection_service import CodexDetectionService


logger = get_plugin_logger()


async def codex_health_check(
    config: CodexSettings | None,
    detection_service: CodexDetectionService | None = None,
    auth_manager: Any | None = None,
) -> HealthCheckResult:
    """Perform health check for Codex plugin."""
    try:
        if not config:
            return HealthCheckResult(
                status="fail",
                componentId="plugin-codex",
                output="Codex plugin configuration not available",
                version="1.0.0",
            )

        # Check basic configuration validity
        if not config.base_url:
            return HealthCheckResult(
                status="fail",
                componentId="plugin-codex",
                output="Codex base URL not configured",
                version="1.0.0",
            )

        # Check OAuth configuration
        if not config.oauth.base_url or not config.oauth.client_id:
            return HealthCheckResult(
                status="warn",
                componentId="plugin-codex",
                output="Codex OAuth configuration incomplete",
                version="1.0.0",
            )

        # Get CLI status if detection service is available
        cli_details = {}
        if detection_service:
            cli_version = detection_service.get_version()
            cli_path = detection_service.get_binary_path()

            cli_details = {
                "cli_available": cli_path is not None,
                "cli_version": cli_version,
                "cli_path": cli_path,
            }

        # Get authentication status if auth manager is available
        auth_details: dict[str, Any] = {}
        if auth_manager:
            try:
                # Use the new helper method to get auth status
                auth_details = await auth_manager.get_auth_status()
            except Exception as e:
                logger.debug(
                    "Failed to check auth status", error=str(e), category="auth"
                )
                auth_details = {
                    "auth_configured": False,
                    "auth_error": str(e),
                }

        # Determine overall status
        status: Literal["pass", "warn", "fail"]
        if cli_details.get("cli_available") and auth_details.get("token_available"):
            output = f"Codex plugin is healthy (CLI v{cli_details.get('cli_version')} available, authenticated)"
            status = "pass"
        elif cli_details.get("cli_available"):
            output = f"Codex plugin is functional (CLI v{cli_details.get('cli_version')} available, auth missing)"
            status = "warn"
        elif auth_details.get("token_available"):
            output = "Codex plugin is functional (authenticated, CLI not found)"
            status = "warn"
        else:
            output = "Codex plugin is functional but CLI and auth missing"
            status = "warn"

        # Basic health check passes
        return HealthCheckResult(
            status=status,
            componentId="plugin-codex",
            output=output,
            version="1.0.0",
            details={
                "base_url": config.base_url,
                "oauth_configured": bool(
                    config.oauth.base_url and config.oauth.client_id
                ),
                "verbose_logging": config.verbose_logging,
                **cli_details,  # Include CLI details if available
                **auth_details,  # Include auth details if available
            },
        )

    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return HealthCheckResult(
            status="fail",
            componentId="plugin-codex",
            output=f"Codex health check failed: {str(e)}",
            version="1.0.0",
        )
