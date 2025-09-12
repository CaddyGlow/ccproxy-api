import pytest
from httpx import AsyncClient


pytestmark = [pytest.mark.integration, pytest.mark.metrics]


@pytest.mark.asyncio(loop_scope="session")
async def test_metrics_route_available_when_metrics_plugin_enabled(
    metrics_integration_client: AsyncClient,
) -> None:
    """Test that metrics route is available when metrics plugin is enabled."""
    resp = await metrics_integration_client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus exposition format usually starts with HELP/TYPE lines
    assert b"# HELP" in resp.content or b"# TYPE" in resp.content


@pytest.mark.asyncio(loop_scope="session")
async def test_metrics_route_absent_when_plugins_disabled(
    disabled_plugins_client: AsyncClient,
) -> None:
    """Test that metrics route is absent when plugins are disabled."""
    resp = await disabled_plugins_client.get("/metrics")
    # With plugins disabled, core does not mount /metrics
    assert resp.status_code == 404


@pytest.mark.asyncio(loop_scope="session")
async def test_metrics_endpoint_with_custom_config(
    metrics_custom_integration_client: AsyncClient,
) -> None:
    """Test metrics endpoint with custom configuration."""
    resp = await metrics_custom_integration_client.get("/metrics")
    assert resp.status_code == 200


@pytest.mark.asyncio(loop_scope="session")
async def test_metrics_health_when_plugin_enabled(
    metrics_integration_client: AsyncClient,
) -> None:
    """Test metrics health endpoint when plugin is enabled."""
    resp = await metrics_integration_client.get("/metrics/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") in {"healthy", "disabled"}
