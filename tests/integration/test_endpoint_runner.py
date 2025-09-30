"""Integration tests for endpoint runner using recorded upstream samples."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app, initialize_plugins_startup, shutdown_plugins
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.core import LoggingSettings
from ccproxy.config.settings import Settings
from ccproxy.config.utils import SchedulerSettings
from ccproxy.core.logging import setup_logging
from ccproxy.testing.endpoints import (
    ENDPOINT_TESTS,
    EndpointRequestResult,
    EndpointTestResult,
    TestEndpoint,
)
from ccproxy.testing.endpoints.config import REQUEST_DATA
from tests.conftest import ENDPOINT_TEST_SELECTION_ENV, get_selected_endpoint_indices
from tests.helpers.sample_loader import load_sample_registry


pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.asyncio(loop_scope="function"),
]


BASE_URL = "http://test"
SAMPLE_REGISTRY = load_sample_registry()
AVAILABLE_CASES = {case.name: case for case in ENDPOINT_TESTS}
SAMPLE_NAMES = [name for name in sorted(SAMPLE_REGISTRY) if name in AVAILABLE_CASES]
CASE_INDEX_LOOKUP = {case.name: index for index, case in enumerate(ENDPOINT_TESTS)}

TestEndpoint.__test__ = False

PROVIDER_FIXTURES = {
    "copilot": "mock_external_copilot_api",
    "claude": "mock_external_anthropic_api_samples",
    "codex": "mock_external_codex_api",
}


def _resolve_sample_names() -> list[str]:
    selection = os.getenv(ENDPOINT_TEST_SELECTION_ENV)
    if not selection:
        return SAMPLE_NAMES

    indices = get_selected_endpoint_indices(selection)
    return [SAMPLE_NAMES[idx] for idx in indices if 0 <= idx < len(SAMPLE_NAMES)]


SELECTED_SAMPLE_NAMES = _resolve_sample_names()


@asynccontextmanager
async def _copilot_app() -> AsyncIterator[FastAPI]:
    from ccproxy.plugins.copilot.models import CopilotCacheData

    setup_logging(json_logs=False, log_level_name="DEBUG")
    settings = Settings(
        enable_plugins=True,
        enabled_plugins=["copilot", "oauth_copilot"],
        logging=LoggingSettings(
            **{
                "level": "DEBUG",
                "verbose_api": True,
            }
        ),
    )

    service_container = create_service_container(settings)
    app = create_app(service_container)

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
        "ccproxy.plugins.copilot.manager.CopilotTokenManager.ensure_oauth_token",
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
            if hasattr(app.state, "service_container"):
                await app.state.service_container.close()


@asynccontextmanager
async def _codex_app() -> AsyncIterator[FastAPI]:
    from types import SimpleNamespace

    from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
    from ccproxy.plugins.codex.models import CodexCacheData

    setup_logging(json_logs=False, log_level_name="DEBUG")
    settings = Settings(
        scheduler=SchedulerSettings(enabled=False),
        enable_plugins=True,
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
            if hasattr(app.state, "service_container"):
                await app.state.service_container.close()


@asynccontextmanager
async def _claude_app() -> AsyncIterator[FastAPI]:
    from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
    from ccproxy.plugins.claude_api.models import ClaudeCacheData

    setup_logging(json_logs=False, log_level_name="DEBUG")
    settings = Settings(
        enable_plugins=True,
        enabled_plugins=["claude_api", "oauth_claude"],
    )

    service_container = create_service_container(settings)
    app = create_app(service_container)

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
            if hasattr(app.state, "service_container"):
                await app.state.service_container.close()


PROVIDER_APP_BUILDERS = {
    "copilot": _copilot_app,
    "codex": _codex_app,
    "claude": _claude_app,
}


@pytest.mark.parametrize(
    "sample_name", SELECTED_SAMPLE_NAMES, ids=SELECTED_SAMPLE_NAMES
)
async def test_endpoint_case_passes(
    sample_name: str,
    request: pytest.FixtureRequest,
    httpx_mock,
) -> None:
    sample = SAMPLE_REGISTRY[sample_name]
    endpoint_case = AVAILABLE_CASES[sample_name]
    provider = sample_name.split("_", 1)[0]

    fixture_name = PROVIDER_FIXTURES.get(provider)
    if fixture_name:
        request.getfixturevalue(fixture_name)

    app_builder = PROVIDER_APP_BUILDERS.get(provider)
    if app_builder is None:
        pytest.skip(f"No provider app builder registered for provider '{provider}'")

    async with app_builder() as app:
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url=BASE_URL) as client,
            TestEndpoint(base_url=BASE_URL, client=client) as tester,
        ):
            result = await tester.run_endpoint_test(
                endpoint_case,
                CASE_INDEX_LOOKUP[endpoint_case.name],
            )

    _assert_initial_request(result, endpoint_case.stream, endpoint_case.request)
    _assert_follow_up_requests(result)

    assert result.success, (
        result.error or f"Endpoint test '{endpoint_case.name}' failed"
    )


def _assert_initial_request(
    result: EndpointTestResult, expected_stream: bool, request_key: str
) -> EndpointRequestResult:
    assert result.request_results, "Test run should record at least one request"
    initial_request = result.request_results[0]

    assert initial_request.status_code == 200, (
        f"Expected HTTP 200 for initial request, got {initial_request.status_code}"
    )
    assert initial_request.stream == expected_stream, (
        "Recorded request type does not match test configuration"
    )

    if expected_stream:
        if initial_request.details.get("fallback_applied", False):
            pytest.fail(
                "Streaming response fell back to JSON; expected native streamed output"
            )
        assert initial_request.details.get("event_count", 0) > 0, (
            "Streaming test completed without emitting events"
        )
    else:
        payload = initial_request.details.get("response")
        assert isinstance(payload, dict), "Expected JSON response payload"
        assert not payload.get("error"), payload.get("error")

        api_format = REQUEST_DATA[request_key]["api_format"]
        required_field = REQUEST_DATA[request_key].get(
            "validation_field", DEFAULT_VALIDATION_FIELDS.get(api_format)
        )
        if required_field:
            assert payload.get(required_field), (
                f"{api_format} response missing '{required_field}' field"
            )

    return initial_request


def _assert_follow_up_requests(result: EndpointTestResult) -> None:
    for extra in result.request_results[1:]:
        if extra.status_code is None:
            continue
        error_detail = extra.details.get("error_detail")
        suffix = f": {error_detail}" if error_detail else ""
        assert extra.status_code == 200, (
            f"Expected HTTP 200 for follow-up request, got {extra.status_code}{suffix}"
        )


DEFAULT_VALIDATION_FIELDS = {
    "openai": "choices",
    "responses": "output",
    "anthropic": "content",
}
