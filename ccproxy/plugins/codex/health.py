"""Codex health check implementation."""

from typing import Any, Literal

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.protocol import HealthCheckResult

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

        # Standardized details models
        from ccproxy.core.plugins.models import (
            AuthHealth,
            CLIHealth,
            ConfigHealth,
            ProviderHealthDetails,
        )

        cli_info = (
            detection_service.get_cli_health_info() if detection_service else None
        )
        status_val = (
            cli_info.status.value
            if (cli_info and hasattr(cli_info, "status"))
            else "unknown"
        )
        available = bool(status_val == "available")
        cli_health = (
            CLIHealth(
                available=available,
                status=status_val,
                version=(cli_info.version if cli_info else None),
                path=(cli_info.binary_path if cli_info else None),
            )
            if cli_info
            else None
        )

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
                    "authenticated": False,
                    "reason": str(e),
                }

        # Determine overall status
        status: Literal["pass", "warn", "fail"]
        provider_auth = (
            AuthHealth(
                configured=bool(auth_manager),
                token_available=auth_details.get("authenticated"),
                token_expired=(
                    not auth_details.get("authenticated")
                    and auth_details.get("reason") == "Token expired"
                ),
                account_id=auth_details.get("account_id"),
                expires_at=auth_details.get("expires_at"),
                error=(
                    None
                    if auth_details.get("authenticated")
                    else auth_details.get("reason")
                ),
            )
            if auth_manager
            else AuthHealth(configured=False)
        )

        if (cli_health and cli_health.available) and provider_auth.token_available:
            output = f"Codex plugin is healthy (CLI v{cli_health.version} available, authenticated)"
            status = "pass"
        elif cli_health and cli_health.available:
            output = f"Codex plugin is functional (CLI v{cli_health.version} available, auth missing)"
            status = "warn"
        elif provider_auth.token_available:
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
                **ProviderHealthDetails(
                    provider="codex",
                    enabled=True,
                    base_url=config.base_url,
                    cli=cli_health,
                    auth=provider_auth,
                    config=ConfigHealth(
                        model_count=None,
                        supports_openai_format=None,
                        verbose_logging=config.verbose_logging,
                        extra={
                            "oauth_configured": bool(
                                config.oauth.base_url and config.oauth.client_id
                            )
                        },
                    ),
                ).model_dump(),
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
