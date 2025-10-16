from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ccproxy.api.app import create_app
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config import Settings


@pytest.fixture()
def health_client() -> Iterator[TestClient]:
    settings = Settings(enable_plugins=False)
    container = create_service_container(settings)
    app = create_app(container)

    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("path", ["/health", "/health/live", "/health/ready"])
def test_health_endpoints_expose_health_media_type(
    health_client: TestClient, path: str
) -> None:
    response = health_client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/health+json"


def test_health_openapi_declares_health_content_type(health_client: TestClient) -> None:
    schema = health_client.app.openapi()  # type: ignore[attr-defined]

    for path in ["/health", "/health/live", "/health/ready"]:
        content = schema["paths"][path]["get"]["responses"]["200"]["content"]
        assert "application/health+json" in content
