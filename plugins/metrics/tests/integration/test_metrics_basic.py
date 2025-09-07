import pytest
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app, initialize_plugins_startup
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.settings import Settings


@pytest.mark.asyncio
@pytest.mark.integration
async def test_metrics_endpoint_available_when_enabled():
    settings = Settings(
        enable_plugins=True,
        plugins={
            "metrics": {
                "enabled": True,
                "metrics_endpoint_enabled": True,
            }
        },
    )

    service_container = create_service_container(settings)
    app = create_app(service_container)
    await initialize_plugins_startup(app, settings)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
