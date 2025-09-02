import pytest
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app, initialize_plugins_startup
from ccproxy.config.settings import Settings


pytestmark = [pytest.mark.integration, pytest.mark.api]


@pytest.mark.asyncio
async def test_metrics_plugin_health_endpoint() -> None:
    """Metrics plugin exposes health via /plugins/metrics/health."""
    settings = Settings(
        enable_plugins=True,
        plugins={
            "metrics": {
                "enabled": True,
                "metrics_endpoint_enabled": True,
            }
        },
    )

    app = create_app(settings)
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/plugins/metrics/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["plugin"] == "metrics"
        assert data["status"] in {"healthy", "unknown"}
        assert data["adapter_loaded"] is True


@pytest.mark.asyncio
async def test_unknown_plugin_health_returns_404() -> None:
    settings = Settings(enable_plugins=True)
    app = create_app(settings)
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/plugins/does-not-exist/health")
        assert resp.status_code == 404
