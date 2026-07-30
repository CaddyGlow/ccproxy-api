"""Unit tests for the MiniMax adapter."""

import json
from unittest.mock import Mock

import httpx
import pytest

from ccproxy.core.errors import AuthenticationError
from ccproxy.plugins.minimax.adapter import MiniMaxAdapter
from ccproxy.plugins.minimax.config import MiniMaxConfig


def _make_adapter(config: MiniMaxConfig) -> MiniMaxAdapter:
    return MiniMaxAdapter(
        config=config,
        auth_manager=None,
        http_pool_manager=Mock(),
    )


@pytest.mark.minimax
@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_target_url() -> None:
    adapter = _make_adapter(MiniMaxConfig(api_key="test-key"))
    url = await adapter.get_target_url("/chat/completions")
    assert url == "https://api.minimax.io/v1/chat/completions"


@pytest.mark.minimax
@pytest.mark.unit
@pytest.mark.asyncio
async def test_prepare_provider_request_sets_bearer_key() -> None:
    adapter = _make_adapter(MiniMaxConfig(api_key="secret-value"))
    body = json.dumps({"model": "MiniMax-M3", "messages": []}).encode()
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer client-token",  # must be replaced
    }

    result_body, result_headers = await adapter.prepare_provider_request(
        body, headers, "/chat/completions"
    )

    assert result_body == body
    assert result_headers["authorization"] == "Bearer secret-value"
    assert "x-request-id" in result_headers


@pytest.mark.minimax
@pytest.mark.unit
@pytest.mark.asyncio
async def test_prepare_provider_request_requires_api_key() -> None:
    adapter = _make_adapter(MiniMaxConfig())
    with pytest.raises(AuthenticationError):
        await adapter.prepare_provider_request(b"{}", {}, "/chat/completions")


@pytest.mark.minimax
@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_provider_response_passthrough() -> None:
    adapter = _make_adapter(MiniMaxConfig(api_key="test-key"))
    payload = {"id": "cmpl-1", "choices": []}
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = json.dumps(payload).encode()
    mock_response.headers = {"content-type": "application/json"}

    result = await adapter.process_provider_response(mock_response, "/chat/completions")

    assert result.status_code == 200
    body = result.body
    if isinstance(body, memoryview):
        body = bytes(body)
    assert json.loads(body.decode()) == payload
