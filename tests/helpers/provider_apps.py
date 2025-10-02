"""Provider-specific FastAPI app factories for endpoint runner tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI

from ccproxy.api.app import create_app, initialize_plugins_startup, shutdown_plugins
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.core import LoggingSettings
from ccproxy.config.settings import Settings
from ccproxy.config.utils import SchedulerSettings
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.core.logging import setup_logging


@asynccontextmanager
async def copilot_app() -> AsyncIterator[FastAPI]:
    """Create a Copilot-enabled FastAPI application for testing."""

    from ccproxy.plugins.copilot.models import CopilotCacheData

    setup_logging(
        json_logs=False, log_level_name="ERROR"
    )  # Changed from DEBUG to ERROR for faster tests
    settings = Settings(
        enable_plugins=True,
        enabled_plugins=["copilot", "oauth_copilot"],
        logging=LoggingSettings(
            **{
                "level": "ERROR",  # Changed from DEBUG to ERROR
                "verbose_api": False,  # Changed from True to False
            }
        ),
    )

    service_container = create_service_container(settings)
    app = create_app(service_container)

    await start_task_manager(container=service_container)

    detection_patch = patch(
        "ccproxy.plugins.copilot.detection_service.CopilotDetectionService.initialize_detection",
        new=AsyncMock(
            return_value=CopilotCacheData(
                cli_available=False,
                cli_version=None,
                auth_status=None,
                username=None,
            )
        ),
    )
    ensure_copilot_patch = patch(
        "ccproxy.plugins.copilot.manager.CopilotTokenManager.ensure_copilot_token",
        new=AsyncMock(return_value="copilot_test_service_token"),
    )
    ensure_oauth_patch = patch(
        "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider.ensure_oauth_token",
        new=AsyncMock(return_value="gh_oauth_access_token"),
    )
    profile_patch = patch(
        "ccproxy.plugins.copilot.manager.CopilotTokenManager.get_profile_quick",
        new=AsyncMock(return_value=None),
    )

    with detection_patch, ensure_copilot_patch, ensure_oauth_patch, profile_patch:
        await initialize_plugins_startup(app, settings)
        try:
            yield app
        finally:
            await shutdown_plugins(app)
            await stop_task_manager(container=service_container)
            if hasattr(app.state, "service_container"):
                await app.state.service_container.close()


@asynccontextmanager
async def codex_app() -> AsyncIterator[FastAPI]:
    """Create a Codex-enabled FastAPI application for testing."""

    from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
    from ccproxy.plugins.codex.models import CodexCacheData

    setup_logging(
        json_logs=False, log_level_name="ERROR"
    )  # Changed from DEBUG to ERROR for faster tests
    settings = Settings(
        scheduler=SchedulerSettings(enabled=False),
        enable_plugins=True,
        plugins_disable_local_discovery=False,
        enabled_plugins=["codex", "oauth_codex"],
    )

    service_container = create_service_container(settings)
    app = create_app(service_container)

    credentials_stub = SimpleNamespace(
        access_token="test-codex-access-token",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    profile_stub = SimpleNamespace(chatgpt_account_id="test-account-id")

    load_patch = patch(
        "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.load_credentials",
        new=AsyncMock(return_value=credentials_stub),
    )
    profile_patch = patch(
        "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_profile_quick",
        new=AsyncMock(return_value=profile_stub),
    )

    prompts = DetectedPrompts.from_body(
        {"instructions": "You are a helpful coding assistant."}
    )
    detection_data = CodexCacheData(
        codex_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url="https://chatgpt.com/backend-codex/responses",
        path="/api/backend-codex/responses",
        query_params={},
    )

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cached_data = detection_data
        return detection_data

    detection_patch = patch(
        "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
        new=init_detection_stub,
    )

    with load_patch, profile_patch, detection_patch:
        await initialize_plugins_startup(app, settings)
        try:
            yield app
        finally:
            await shutdown_plugins(app)
            await stop_task_manager(container=service_container)
            if hasattr(app.state, "service_container"):
                await app.state.service_container.close()


@asynccontextmanager
async def claude_app() -> AsyncIterator[FastAPI]:
    """Create a Claude API-enabled FastAPI application for testing."""

    from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
    from ccproxy.plugins.claude_api.models import ClaudeCacheData

    setup_logging(
        json_logs=False, log_level_name="ERROR"
    )  # Changed from DEBUG to ERROR for faster tests
    settings = Settings(
        enable_plugins=True,
        enabled_plugins=["claude_api", "oauth_claude"],
    )

    service_container = create_service_container(settings)
    app = create_app(service_container)

    await start_task_manager(container=service_container)

    token_patch = patch(
        "ccproxy.plugins.oauth_claude.manager.ClaudeApiTokenManager.get_access_token",
        new=AsyncMock(return_value="test-claude-access-token"),
    )
    load_patch = patch(
        "ccproxy.plugins.oauth_claude.manager.ClaudeApiTokenManager.load_credentials",
        new=AsyncMock(return_value=None),
    )

    prompts = DetectedPrompts.from_body(
        {"system": [{"type": "text", "text": "Hello from tests."}]}
    )
    detection_data = ClaudeCacheData(
        claude_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url=None,
        path=None,
        query_params=None,
    )

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cached_data = detection_data
        return detection_data

    detection_patch = patch(
        "ccproxy.plugins.claude_api.detection_service.ClaudeAPIDetectionService.initialize_detection",
        new=init_detection_stub,
    )

    with token_patch, load_patch, detection_patch:
        await initialize_plugins_startup(app, settings)
        try:
            yield app
        finally:
            await shutdown_plugins(app)
            await stop_task_manager(container=service_container)
            if hasattr(app.state, "service_container"):
                await app.state.service_container.close()


PROVIDER_APP_BUILDERS = {
    "copilot": copilot_app,
    "codex": codex_app,
    "claude": claude_app,
}

PROVIDER_FIXTURES = {
    "copilot": "mock_external_copilot_api",
    "claude": "mock_external_anthropic_api_samples",
    "codex": "mock_external_codex_api",
}
