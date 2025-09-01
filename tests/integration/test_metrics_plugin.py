import pytest
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app
from ccproxy.config.settings import Settings


pytestmark = [pytest.mark.integration, pytest.mark.metrics]


@pytest.mark.asyncio
async def test_metrics_route_available_when_metrics_plugin_enabled() -> None:
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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        # Prometheus exposition format usually starts with HELP/TYPE lines
        assert b"# HELP" in resp.content or b"# TYPE" in resp.content


@pytest.mark.asyncio
async def test_metrics_route_absent_when_plugins_disabled() -> None:
    settings = Settings(enable_plugins=False)
    app = create_app(settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
        # With plugins disabled, core does not mount /metrics
        assert resp.status_code == 404
