import pytest

from ccproxy.core.errors import AuthenticationError
from ccproxy.plugins.codex.adapter import CodexAdapter


class DummyDetection:
    def get_detected_headers(self):
        return None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_codex_adapter_raises_auth_error_when_no_manager():
    adapter = CodexAdapter(
        detection_service=DummyDetection(),
        config=type("C", (), {"base_url": "https://example"})(),
        auth_manager=None,  # type: ignore
        http_pool_manager=None,  # type: ignore
    )
    # Force missing token_manager
    adapter.token_manager = None  # type: ignore

    with pytest.raises(AuthenticationError):
        await adapter._resolve_access_token()
