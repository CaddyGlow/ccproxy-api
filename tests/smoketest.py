"""Smoketest suite for CCProxy - quick validation of core endpoints."""

import httpx
import pytest


# Mark all tests in this module as smoketests
pytestmark = pytest.mark.smoketest


class TestSmokeTests:
    """Essential smoketests for CCProxy endpoints."""

    @pytest.fixture
    async def client(self) -> httpx.AsyncClient:
        """HTTP client for making requests."""
        async with httpx.AsyncClient(
            base_url="http://127.0.0.1:8000", timeout=15.0
        ) as client:
            yield client

    async def test_health_endpoint(self, client: httpx.AsyncClient) -> None:
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_copilot_chat_completions(self, client: httpx.AsyncClient) -> None:
        """Test Copilot chat completions endpoint."""
        payload = {
            "model": "gpt-4o",  # Copilot uses gpt-4o
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False,
        }
        response = await client.post("/copilot/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    async def test_api_chat_completions(self, client: httpx.AsyncClient) -> None:
        """Test Claude API chat completions endpoint."""
        payload = {
            "model": "claude-sonnet-4-20250514",  # Claude API uses Claude models
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False,
        }
        response = await client.post("/api/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    async def test_copilot_responses(self, client: httpx.AsyncClient) -> None:
        """Test Copilot responses API endpoint."""
        payload = {
            "model": "gpt-4o",  # Copilot uses gpt-4o
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
        response = await client.post("/copilot/v1/responses", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_api_responses(self, client: httpx.AsyncClient) -> None:
        """Test Claude API responses endpoint."""
        payload = {
            "model": "claude-sonnet-4-20250514",  # Claude API uses Claude models
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
        response = await client.post("/api/v1/responses", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_copilot_chat_completions_stream(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test Copilot chat completions streaming endpoint."""
        payload = {
            "model": "gpt-4o",  # Copilot uses gpt-4o
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        }
        headers = {"Accept": "text/event-stream"}

        event_count = 0
        async with client.stream(
            "POST", "/copilot/v1/chat/completions", json=payload, headers=headers
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            async for chunk in response.aiter_text():
                if chunk.strip() and chunk.startswith("data: "):
                    event_count += 1
                    if event_count >= 3:  # Just validate we get streaming events
                        break

        assert event_count >= 1, "Should receive at least one streaming event"

    async def test_api_chat_completions_stream(self, client: httpx.AsyncClient) -> None:
        """Test Claude API chat completions streaming endpoint."""
        payload = {
            "model": "claude-sonnet-4-20250514",  # Claude API uses Claude models
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        }
        headers = {"Accept": "text/event-stream"}

        event_count = 0
        async with client.stream(
            "POST", "/api/v1/chat/completions", json=payload, headers=headers
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            async for chunk in response.aiter_text():
                if chunk.strip() and chunk.startswith("data: "):
                    event_count += 1
                    if event_count >= 3:  # Just validate we get streaming events
                        break

        assert event_count >= 1, "Should receive at least one streaming event"

    async def test_copilot_responses_stream(self, client: httpx.AsyncClient) -> None:
        """Test Copilot responses API streaming endpoint."""
        payload = {
            "model": "gpt-4o",  # Copilot uses gpt-4o
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

        event_count = 0
        async with client.stream(
            "POST", "/copilot/v1/responses", json=payload, headers=headers
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            async for chunk in response.aiter_text():
                if chunk.strip() and (
                    chunk.startswith("data: ") or chunk.startswith("event: ")
                ):
                    event_count += 1
                    if event_count >= 5:  # Responses API has more events
                        break

        assert event_count >= 1, "Should receive at least one streaming event"

    async def test_codex_chat_completions(self, client: httpx.AsyncClient) -> None:
        """Test Codex chat completions endpoint."""
        payload = {
            "model": "gpt-5",  # Codex uses gpt-5
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False,
        }
        response = await client.post("/api/codex/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    async def test_codex_chat_completions_stream(
        self, client: httpx.AsyncClient
    ) -> None:
        """Test Codex chat completions streaming endpoint."""
        payload = {
            "model": "gpt-5",  # Codex uses gpt-5
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
        }
        headers = {"Accept": "text/event-stream"}

        event_count = 0
        async with client.stream(
            "POST", "/api/codex/v1/chat/completions", json=payload, headers=headers
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            async for chunk in response.aiter_text():
                if chunk.strip() and chunk.startswith("data: "):
                    event_count += 1
                    if event_count >= 3:  # Just validate we get streaming events
                        break

        assert event_count >= 1, "Should receive at least one streaming event"

    async def test_codex_responses(self, client: httpx.AsyncClient) -> None:
        """Test Codex responses endpoint."""
        payload = {
            "model": "gpt-5",  # Codex uses gpt-5
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
        response = await client.post("/codex/responses", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_codex_responses_stream(self, client: httpx.AsyncClient) -> None:
        """Test Codex responses streaming endpoint."""
        payload = {
            "model": "gpt-5",  # Codex uses gpt-5
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

        event_count = 0
        async with client.stream(
            "POST", "/codex/responses", json=payload, headers=headers
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            async for chunk in response.aiter_text():
                if chunk.strip() and (
                    chunk.startswith("data: ") or chunk.startswith("event: ")
                ):
                    event_count += 1
                    if event_count >= 5:  # Responses API has more events
                        break

        assert event_count >= 1, "Should receive at least one streaming event"
