"""Health check implementation for Claude SDK plugin."""

from typing import TYPE_CHECKING, Literal, cast

from ccproxy.core.plugins.protocol import HealthCheckResult


if TYPE_CHECKING:
    from .config import ClaudeSDKSettings
    from .detection_service import ClaudeSDKDetectionService


async def claude_sdk_health_check(
    config: "ClaudeSDKSettings | None",
    detection_service: "ClaudeSDKDetectionService | None",
) -> HealthCheckResult:
    """Perform health check for Claude SDK plugin.

    Args:
        config: Claude SDK plugin configuration
        detection_service: Claude CLI detection service

    Returns:
        HealthCheckResult with plugin status
    """
    checks = []
    status: str = "pass"

    # Check if plugin is enabled
    if not config or not config.enabled:
        return HealthCheckResult(
            status="fail",
            componentId="plugin-claude_sdk",
            output="Plugin is disabled",
            version="1.0.0",
            details={"enabled": False},
        )

    # Check Claude CLI detection
    if detection_service:
        cli_version = detection_service.get_version()
        cli_path = detection_service.get_cli_path()
        is_available = detection_service.is_claude_available()
        cli_info = detection_service.get_cli_health_info()

        if is_available and cli_path:
            checks.append(f"CLI: {cli_version or 'detected'} at {cli_path}")
        else:
            checks.append("CLI: not found")
            status = "warn"  # CLI not found is a warning, not a failure
    else:
        checks.append("CLI: detection service not initialized")
        status = "warn"

    # Check configuration
    if config:
        checks.append(f"Models: {len(config.models)} configured")
        checks.append(
            f"Session pool: {'enabled' if config.session_pool_enabled else 'disabled'}"
        )
        checks.append(
            f"Streaming: {'enabled' if config.supports_streaming else 'disabled'}"
        )
    else:
        checks.append("Config: not loaded")
        status = "fail"

    # Standardized details
    from ccproxy.core.plugins.models import (
        CLIHealth,
        ConfigHealth,
        ProviderHealthDetails,
    )

    cli_health = None
    if detection_service:
        path_list = detection_service.get_cli_path()
        cli_status = cli_info.status.value if cli_info else "unknown"
        cli_health = CLIHealth(
            available=bool(detection_service.is_claude_available()),
            status=cli_status,
            version=detection_service.get_version(),
            path=(path_list[0] if path_list else None),
        )

    details = ProviderHealthDetails(
        provider="claude_sdk",
        enabled=bool(config and config.enabled),
        base_url=None,
        cli=cli_health,
        auth=None,
        config=ConfigHealth(
            model_count=len(config.models) if config and config.models else 0,
            supports_openai_format=config.supports_streaming if config else None,
            extra={
                "session_pool_enabled": bool(config.session_pool_enabled)
                if config
                else None
            },
        ),
    ).model_dump()

    return HealthCheckResult(
        status=cast(Literal["pass", "warn", "fail"], status),
        componentId="plugin-claude_sdk",
        output="; ".join(checks),
        version="1.0.0",
        details=details,
    )
