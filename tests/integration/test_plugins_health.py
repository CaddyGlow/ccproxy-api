import pytest


pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.mark.asyncio
async def test_metrics_plugin_health_endpoint(metrics_integration_client) -> None:
    """Metrics plugin exposes health via /plugins/metrics/health."""
    resp = await metrics_integration_client.get("/plugins/metrics/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["plugin"] == "metrics"
    assert data["status"] in {"healthy", "unknown"}
    assert data["adapter_loaded"] is True


@pytest.mark.asyncio
async def test_unknown_plugin_health_returns_404(disabled_plugins_client) -> None:
    resp = await disabled_plugins_client.get("/plugins/does-not-exist/health")
    assert resp.status_code == 404
