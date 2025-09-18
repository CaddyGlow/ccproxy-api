"""Smoketest using recorded mocks for provider endpoints.

- RECORD_MOCKS=true to capture real responses into tests/smoketest/mocks/
- Default mode replays mocks via middleware for fast, serverless runs.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest
import structlog
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config import LoggingSettings, ServerSettings, Settings
from ccproxy.services.container import ServiceContainer
from tests.smoketest.mock_util import is_record_mode, make_mock_middleware


pytestmark = pytest.mark.smoketest


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    # Configure settings
    settings = Settings()
    settings.logging = LoggingSettings(level="DEBUG")
    settings.server = ServerSettings()
    settings.enable_plugins = True
    settings.plugins_disable_local_discovery = False

    # Logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.EventRenamer("event"),
            structlog.processors.JSONRenderer(),
        ]
    )

    container: ServiceContainer = create_service_container(settings)
    app = create_app(container)

    # Map endpoints to mock names
    route_map = {
        ("POST", "/copilot/v1/chat/completions"): "copilot_chat_completions",
        ("POST", "/api/v1/chat/completions"): "api_chat_completions",
        ("POST", "/copilot/v1/responses"): "copilot_responses",
        ("POST", "/api/v1/responses"): "api_responses",
        ("POST", "/api/codex/v1/chat/completions"): "codex_chat_completions",
        ("POST", "/api/codex/responses"): "codex_responses",
    }

    # Install mock/record middleware
    app.middleware("http")(make_mock_middleware(route_map))

    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://testserver") as c,
    ):
        # In record mode, wait for real startup to be ready
        if is_record_mode():
            for _ in range(50):
                try:
                    r = await c.get("/health")
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.1)
        yield c


class TestSmokeMocks:
    async def test_copilot_chat_completions(self, client: AsyncClient) -> None:
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False,
        }
        r = await client.post("/copilot/v1/chat/completions", json=payload)
        assert r.status_code == 200
        assert "choices" in r.json()

    async def test_api_chat_completions(self, client: AsyncClient) -> None:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False,
        }
        r = await client.post("/api/v1/chat/completions", json=payload)
        assert r.status_code == 200
        assert "choices" in r.json()

    async def test_copilot_responses(self, client: AsyncClient) -> None:
        payload = {
            "model": "gpt-4o",
            "stream": False,
            "max_completion_tokens": 10,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
        r = await client.post("/copilot/v1/responses", json=payload)
        assert r.status_code == 200
        assert "output" in r.json()

    async def test_api_responses(self, client: AsyncClient) -> None:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "stream": False,
            "max_completion_tokens": 10,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
        r = await client.post("/api/v1/responses", json=payload)
        assert r.status_code == 200
        assert "output" in r.json()

    async def test_copilot_chat_completions_stream(self, client: AsyncClient) -> None:
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        }
        headers = {"Accept": "text/event-stream"}
        async with client.stream(
            "POST", "/copilot/v1/chat/completions", json=payload, headers=headers
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            count = 0
            async for chunk in r.aiter_text():
                if chunk.strip() and chunk.startswith("data: "):
                    count += 1
                    if count >= 3:
                        break
            assert count >= 1

    async def test_api_chat_completions_stream(self, client: AsyncClient) -> None:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        }
        headers = {"Accept": "text/event-stream"}
        async with client.stream(
            "POST", "/api/v1/chat/completions", json=payload, headers=headers
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            count = 0
            async for chunk in r.aiter_text():
                if chunk.strip() and chunk.startswith("data: "):
                    count += 1
                    if count >= 3:
                        break
            assert count >= 1

    async def test_codex_chat_completions(self, client: AsyncClient) -> None:
        payload = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False,
        }
        r = await client.post("/api/codex/v1/chat/completions", json=payload)
        assert r.status_code == 200
        assert "choices" in r.json()

    async def test_codex_responses(self, client: AsyncClient) -> None:
        payload = {
            "model": "gpt-5",
            "stream": False,
            "max_completion_tokens": 10,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
        r = await client.post("/api/codex/responses", json=payload)
        assert r.status_code == 200
        assert "output" in r.json()

    async def test_codex_chat_completions_stream(self, client: AsyncClient) -> None:
        payload = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        }
        headers = {"Accept": "text/event-stream"}
        async with client.stream(
            "POST", "/api/codex/v1/chat/completions", json=payload, headers=headers
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            count = 0
            async for chunk in r.aiter_text():
                if chunk.strip() and chunk.startswith("data: "):
                    count += 1
                    if count >= 3:
                        break
            assert count >= 1

    async def test_codex_responses_stream(self, client: AsyncClient) -> None:
        payload = {
            "model": "gpt-5",
            "stream": True,
            "max_completion_tokens": 10,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
        headers = {"Accept": "text/event-stream"}
        async with client.stream(
            "POST", "/api/codex/responses", json=payload, headers=headers
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            count = 0
            async for chunk in r.aiter_text():
                if chunk.strip() and (
                    chunk.startswith("data: ") or chunk.startswith("event: ")
                ):
                    count += 1
                    if count >= 5:
                        break
            assert count >= 1
