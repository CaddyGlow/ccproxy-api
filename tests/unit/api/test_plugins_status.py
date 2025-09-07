import pytest
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app, initialize_plugins_startup
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.settings import Settings


pytestmark = [pytest.mark.unit, pytest.mark.api]


@pytest.mark.asyncio
async def test_plugins_status_types() -> None:
    settings = Settings(
        enable_plugins=True,
        plugins={
            # Enable metrics to ensure a system plugin is present
            "metrics": {"enabled": True, "metrics_endpoint_enabled": True},
        },
    )
    # create_app expects a ServiceContainer; build it from settings
    container = create_service_container(settings)
    app = create_app(container)
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/plugins/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "plugins" in data
        names_to_types = {p["name"]: p["type"] for p in data["plugins"]}

        # Expect at least one provider plugin and one system plugin
        assert "claude_api" in names_to_types or "codex" in names_to_types
        assert "metrics" in names_to_types

        # Type assertions (best-effort; plugins may vary by config)
        if "metrics" in names_to_types:
            assert names_to_types["metrics"] == "system"
        # Provider plugins
        for candidate in ("claude_api", "codex"):
            if candidate in names_to_types:
                assert names_to_types[candidate] in {"provider", "auth_provider"}
