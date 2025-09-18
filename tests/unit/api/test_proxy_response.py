from typing import Any, Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ccproxy.api.responses import ProxyResponse


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()

    @app.get("/simple")
    def simple() -> ProxyResponse:
        headers = {"Server": "UpstreamX", "X-Custom": "abc"}
        return ProxyResponse(content=b"ok", status_code=200, headers=headers, media_type="text/plain")

    @app.get("/override-length")
    def override_length() -> ProxyResponse:
        # Simulate upstream setting Content-Length and Transfer-Encoding; ProxyResponse should override
        headers = {"Content-Length": "999", "Transfer-Encoding": "chunked", "X-Other": "1"}
        return ProxyResponse(content=b"data", status_code=200, headers=headers, media_type="application/octet-stream")

    @app.get("/no-type")
    def no_type() -> ProxyResponse:
        # No content-type provided; should not add one if media_type is None
        return ProxyResponse(content=b"", status_code=204, headers={})

    return app


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


def test_preserves_header_case_and_values(client: TestClient) -> None:
    res = client.get("/simple")
    assert res.status_code == 200
    # Header case preserved as sent
    assert res.headers.get("Server") == "UpstreamX"
    assert res.headers.get("X-Custom") == "abc"
    # Computed content-length must match body length (2)
    assert res.headers.get("Content-Length") == str(len(b"ok"))
    assert res.text == "ok"


def test_filters_unsafe_and_computes_length(client: TestClient) -> None:
    res = client.get("/override-length")
    assert res.status_code == 200
    # Content-Length and Transfer-Encoding from upstream must be ignored
    assert "Transfer-Encoding" not in res.headers
    assert res.headers.get("Content-Length") == str(len(b"data"))
    # Other headers are preserved
    assert res.headers.get("X-Other") == "1"


def test_content_type_optional_when_media_type_missing(client: TestClient) -> None:
    res = client.get("/no-type")
    assert res.status_code == 204
    # Content-Length should be 0 for empty body
    assert res.headers.get("Content-Length") == "0"
    # No content-type added because media_type=None
    assert "Content-Type" not in res.headers

