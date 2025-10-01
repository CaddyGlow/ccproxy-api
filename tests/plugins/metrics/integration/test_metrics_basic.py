"""Integration coverage for the metrics plugin HTTP endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from httpx import AsyncClient
from starlette.requests import Request

from ccproxy.api.routes.plugins import plugin_health
from ccproxy.api.routes.plugins import router as plugins_router
from ccproxy.auth.dependencies import get_conditional_auth_manager


class _StubRuntime:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self._result = result or {"status": "pass"}

    async def health_check(self) -> dict[str, object]:
        return self._result


class _StubRegistry:
    def __init__(self, plugin: str, runtime: _StubRuntime | None = None) -> None:
        self._plugin = plugin
        self._runtime = runtime or _StubRuntime()

    def list_plugins(self) -> list[str]:
        return [self._plugin]

    def get_runtime(self, name: str) -> _StubRuntime | None:
        return self._runtime if name == self._plugin else None

    def get_factory(self, name: str):  # pragma: no cover - unused by tests
        return None


async def _call_plugin_health(
    registry: _StubRegistry, plugin_name: str
) -> dict[str, object]:
    app = FastAPI()
    app.include_router(plugins_router)
    app.state.plugin_registry = registry
    app.dependency_overrides[get_conditional_auth_manager] = lambda: None

    scope = {
        "type": "http",
        "app": app,
        "headers": [],
        "method": "GET",
        "path": f"/plugins/{plugin_name}/health",
        "query_string": b"",
    }

    async def receive() -> dict[str, str]:  # pragma: no cover - no payload expected
        return {"type": "http.request"}

    request = Request(scope, receive)
    result = await plugin_health(plugin_name=plugin_name, request=request)
    return result.model_dump()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.metrics
async def test_metrics_endpoint_available_when_enabled(
    metrics_integration_client: AsyncClient,
) -> None:
    """When metrics plugin is enabled the /metrics endpoint is exposed."""
    response = await metrics_integration_client.get("/metrics")
    assert response.status_code == 200
    assert b"# HELP" in response.content or b"# TYPE" in response.content


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.metrics
async def test_metrics_endpoint_absent_when_plugins_disabled(
    disabled_plugins_client: AsyncClient,
) -> None:
    """Core app should not expose /metrics when plugins are disabled."""
    response = await disabled_plugins_client.get("/metrics")
    assert response.status_code == 404


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.metrics
async def test_metrics_endpoint_returns_prometheus_format(
    metrics_integration_client: AsyncClient,
) -> None:
    """Endpoint emits Prometheus exposition headers and body."""
    response = await metrics_integration_client.get("/metrics")
    assert response.status_code == 200
    assert (
        response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    )
    assert response.content.strip()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.metrics
async def test_metrics_endpoint_supports_custom_config(
    metrics_custom_integration_client: AsyncClient,
) -> None:
    """Custom configured integration client still exposes metrics."""
    response = await metrics_custom_integration_client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.metrics
async def test_metrics_health_endpoint_reports_status(
    metrics_integration_client: AsyncClient,
) -> None:
    """Health probe surfaces success payload when plugin stays enabled."""
    response = await metrics_integration_client.get("/metrics/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") in {"healthy", "disabled"}


@pytest.mark.integration
@pytest.mark.metrics
@pytest.mark.asyncio
async def test_metrics_plugin_health_endpoint() -> None:
    """Plugin health endpoint surfaces healthy state when runtime passes."""
    payload = await _call_plugin_health(_StubRegistry("metrics"), "metrics")
    assert payload["plugin"] == "metrics"
    assert payload["status"] == "healthy"
    assert payload["adapter_loaded"] is True


@pytest.mark.integration
@pytest.mark.metrics
@pytest.mark.asyncio
async def test_unknown_plugin_health_returns_404() -> None:
    """Requesting health for an unknown plugin returns 404."""
    registry = _StubRegistry("metrics")
    app = FastAPI()
    app.include_router(plugins_router)
    app.state.plugin_registry = registry
    app.dependency_overrides[get_conditional_auth_manager] = lambda: None

    scope = {
        "type": "http",
        "app": app,
        "headers": [],
        "method": "GET",
        "path": "/plugins/does-not-exist/health",
        "query_string": b"",
    }

    async def receive() -> dict[str, str]:  # pragma: no cover - no payload expected
        return {"type": "http.request"}

    request = Request(scope, receive)
    with pytest.raises(HTTPException) as exc:
        await plugin_health("does-not-exist", request=request)
    assert exc.value.status_code == 404
