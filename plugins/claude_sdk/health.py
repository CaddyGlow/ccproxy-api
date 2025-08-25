"""Health check implementation for Claude SDK plugin."""

from typing import TYPE_CHECKING, Literal, cast

from ccproxy.plugins.protocol import HealthCheckResult


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

    return HealthCheckResult(
        status=cast(Literal["pass", "warn", "fail"], status),
        componentId="plugin-claude_sdk",
        output="; ".join(checks),
        version="1.0.0",
        details={
            "cli_available": detection_service.is_claude_available()
            if detection_service
            else False,
            "cli_version": detection_service.get_version()
            if detection_service
            else None,
            "cli_path": detection_service.get_cli_path() if detection_service else None,
            "config_loaded": config is not None,
            "enabled": config.enabled if config else False,
            "checks": checks,
        },
    )
