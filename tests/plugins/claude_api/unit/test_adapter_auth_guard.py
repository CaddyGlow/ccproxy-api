import pytest

from ccproxy.core.errors import AuthenticationError
from ccproxy.plugins.claude_api.adapter import ClaudeAPIAdapter


class DummyDetection:
    pass


class DummyConfig:
    base_url = "https://example"
    support_openai_format = False
    system_prompt_injection_mode = "none"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claude_api_adapter_raises_auth_error_when_no_manager() -> None:
    adapter = ClaudeAPIAdapter(
        detection_service=DummyDetection(),  # type: ignore[arg-type]
        config=DummyConfig(),  # type: ignore[arg-type]
        auth_manager=None,  # type: ignore[arg-type]
        http_pool_manager=None,  # type: ignore[arg-type]
    )
    # Force missing token_manager
    adapter.token_manager = None  # type: ignore

    with pytest.raises(AuthenticationError):
        await adapter._resolve_access_token()
