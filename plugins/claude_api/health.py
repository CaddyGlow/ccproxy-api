"""Claude API plugin health check implementation."""

from typing import Any, Literal

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins.protocol import HealthCheckResult

from .auth.manager import ClaudeApiTokenManager
from .config import ClaudeAPISettings
from .detection_service import ClaudeAPIDetectionService


logger = get_plugin_logger()


async def claude_api_health_check(
    config: ClaudeAPISettings | None,
    detection_service: ClaudeAPIDetectionService | None = None,
    credentials_manager: ClaudeApiTokenManager | None = None,
) -> HealthCheckResult:
    """Perform health check for Claude API plugin.

    Args:
        config: Plugin configuration
        credentials_manager: Token manager for OAuth token status

    Returns:
        HealthCheckResult with plugin status including OAuth token details
    """
    try:
        if not config:
            return HealthCheckResult(
                status="fail",
                componentId="plugin-claude-api",
                componentType="provider_plugin",
                output="Claude API plugin configuration not available",
                version="1.0.0",
            )

        # Check if plugin is enabled
        if not config.enabled:
            return HealthCheckResult(
                status="warn",
                componentId="plugin-claude-api",
                componentType="provider_plugin",
                output="Claude API plugin is disabled",
                version="1.0.0",
                details={"enabled": False},
            )

        # Check basic configuration
        if not config.base_url:
            return HealthCheckResult(
                status="fail",
                componentId="plugin-claude-api",
                componentType="provider_plugin",
                output="Claude API base URL not configured",
                version="1.0.0",
            )

        # Get CLI status if detection service is available
        cli_details = {}
        cli_status_msg = None
        if detection_service:
            cli_version = detection_service.get_version()
            cli_path = detection_service.get_cli_path()

            cli_details = {
                "cli_available": cli_path is not None,
                "cli_version": cli_version,
                "cli_path": cli_path,
            }

            if cli_path:
                cli_status_msg = (
                    f"CLI v{cli_version}" if cli_version else "CLI available"
                )
            else:
                cli_status_msg = "CLI not found"

        # Get authentication status from credentials manager
        auth_details: dict[str, Any] = {}
        if credentials_manager:
            try:
                # Use the new helper method to get auth status
                auth_details = await credentials_manager.get_auth_status()
            except Exception as e:
                logger.debug(
                    "Failed to check auth status", error=str(e), category="auth"
                )
                auth_details = {
                    "auth_configured": False,
                    "auth_error": str(e),
                }
        else:
            auth_details = {
                "auth_configured": False,
                "auth_status": "Credentials manager not available",
            }

        # Determine overall status and build output message
        status: Literal["pass", "warn", "fail"]
        output_parts = []

        if auth_details.get("token_available") and not auth_details.get(
            "token_expired"
        ):
            output_parts.append("Authenticated")
            if auth_details.get("subscription_type"):
                output_parts.append(
                    f"Subscription: {auth_details['subscription_type']}"
                )
            if auth_details.get("has_claude_max"):
                output_parts.append("Claude Max")
            elif auth_details.get("has_claude_pro"):
                output_parts.append("Claude Pro")
            status = "pass"
        elif auth_details.get("token_expired"):
            output_parts.append("Token expired")
            status = "warn"
        elif auth_details.get("auth_configured"):
            output_parts.append("Auth configured but token unavailable")
            status = "warn"
        else:
            output_parts.append("Authentication not configured")
            status = "warn"

        # Add CLI status
        if cli_status_msg:
            output_parts.append(cli_status_msg)

        # Add model info
        if config.models:
            output_parts.append(f"{len(config.models)} models available")

        output = "Claude API: " + ", ".join(output_parts)

        # Build details dict with non-sensitive information
        details = {
            "base_url": config.base_url,
            "enabled": config.enabled,
            "model_count": len(config.models) if config.models else 0,
            "support_openai_format": config.support_openai_format,
            **cli_details,  # Include CLI detection details
            **auth_details,  # Include all auth details from helper
        }

        return HealthCheckResult(
            status=status,
            componentId="plugin-claude-api",
            componentType="provider_plugin",
            output=output,
            version="1.0.0",
            details=details,
        )

    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return HealthCheckResult(
            status="fail",
            componentId="plugin-claude-api",
            componentType="provider_plugin",
            output=f"Claude API health check failed: {str(e)}",
            version="1.0.0",
        )
