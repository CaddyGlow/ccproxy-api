"""Claude API plugin health check implementation."""

from typing import Any, Literal

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.protocol import HealthCheckResult
from ccproxy.plugins.oauth_claude.manager import ClaudeApiTokenManager

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

        # Standardized details
        from ccproxy.core.plugins.models import (
            AuthHealth,
            CLIHealth,
            ConfigHealth,
            ProviderHealthDetails,
        )

        cli_info = (
            detection_service.get_cli_health_info() if detection_service else None
        )
        cli_health = (
            CLIHealth(
                available=bool(
                    cli_info
                    and getattr(cli_info, "status", None)
                    == getattr(cli_info.__class__, "__members__", {}).get("AVAILABLE")
                ),
                status=(cli_info.status.value if cli_info else "unknown"),
                version=(cli_info.version if cli_info else None),
                path=(cli_info.binary_path if cli_info else None),
            )
            if cli_info
            else None
        )

        auth_raw: dict[str, Any] = {}
        if credentials_manager:
            try:
                auth_raw = await credentials_manager.get_auth_status()
            except Exception as e:
                logger.debug("auth_status_failed", error=str(e), category="auth")
                auth_raw = {"authenticated": False, "reason": str(e)}

        auth_health = (
            AuthHealth(
                configured=bool(credentials_manager),
                token_available=auth_raw.get("authenticated"),
                token_expired=(
                    not auth_raw.get("authenticated")
                    and auth_raw.get("reason") == "Token expired"
                ),
                account_id=auth_raw.get("account_id"),
                expires_at=auth_raw.get("expires_at"),
                error=(
                    None if auth_raw.get("authenticated") else auth_raw.get("reason")
                ),
            )
            if credentials_manager
            else AuthHealth(configured=False)
        )

        config_health = ConfigHealth(
            model_count=len(config.models) if config.models else 0,
            supports_openai_format=config.support_openai_format,
            extra=None,
        )

        # Compose output message
        status: Literal["pass", "warn", "fail"]
        output_parts: list[str] = []
        if auth_health.token_available and not auth_health.token_expired:
            output_parts.append("Authenticated")
            status = "pass"
        elif auth_health.token_expired:
            output_parts.append("Token expired")
            status = "warn"
        elif auth_health.configured:
            output_parts.append("Auth configured but token unavailable")
            status = "warn"
        else:
            output_parts.append("Authentication not configured")
            status = "warn"

        if cli_health and cli_health.available:
            output_parts.append(
                f"CLI v{cli_health.version}" if cli_health.version else "CLI available"
            )
        else:
            output_parts.append("CLI not found")

        if config.models:
            output_parts.append(f"{len(config.models)} models available")

        output = "Claude API: " + ", ".join(output_parts)

        details_model = ProviderHealthDetails(
            provider="claude_api",
            enabled=config.enabled,
            base_url=config.base_url,
            cli=cli_health,
            auth=auth_health,
            config=config_health,
        )

        return HealthCheckResult(
            status=status,
            componentId="plugin-claude-api",
            componentType="provider_plugin",
            output=output,
            version="1.0.0",
            details=details_model.model_dump(),
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
