"""Tests for HandlerConfig and unified dispatch architecture."""

from typing import Any

import pytest
from pydantic import SecretStr

from ccproxy.adapters.base import APIAdapter
from ccproxy.auth.manager import AuthManager
from ccproxy.services.handler_config import HandlerConfig


class MockAuthManager(AuthManager):
    """Mock authentication manager for testing."""

    async def get_access_token(self) -> str:
        """Get mock access token."""
        return "test-token"

    async def get_credentials(self) -> Any:
        """Get mock credentials."""
        from ccproxy.auth.models import ClaudeCredentials, OAuthToken

        oauth_token = OAuthToken(
            accessToken=SecretStr("test-token"),
            refreshToken=SecretStr("test-refresh"),
            expiresAt=None,
            scopes=["test"],
            subscriptionType="test",
            tokenType="Bearer",
        )
        return ClaudeCredentials(claudeAiOauth=oauth_token)

    async def is_authenticated(self) -> bool:
        """Mock authentication check."""
        return True

    async def get_user_profile(self) -> Any:
        """Get mock user profile."""
        return None

    async def get_auth_headers(self) -> dict[str, str]:
        """Get mock auth headers."""
        return {"authorization": "Bearer test-token"}

    async def validate_credentials(self) -> bool:
        """Mock validation always returns True."""
        return True

    def get_provider_name(self) -> str:
        """Get mock provider name."""
        return "mock-provider"

    async def __aenter__(self) -> "MockAuthManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass


class MockAdapter(APIAdapter):
    """Mock API adapter for testing."""

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Mock request adaptation."""
        return {"adapted": True, **request}

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Mock response adaptation."""
        return {"adapted": True, **response}

    async def adapt_stream(self, stream):
        """Mock stream adaptation."""
        async for chunk in stream:
            yield {"adapted": True, **chunk}


class MockTransformer:
    """Mock transformer that implements the protocol."""

    def transform_headers(
        self, headers: dict[str, str], **kwargs: Any
    ) -> dict[str, str]:
        """Mock header transformer."""
        result = headers.copy()
        result["x-transformed"] = "true"
        return result

    def transform_body(self, body: Any) -> Any:
        """Mock body transformer."""
        return body


@pytest.mark.asyncio
async def test_handler_config_creation():
    """Test HandlerConfig can be created with simplified fields."""
    context = HandlerConfig()

    # Check defaults
    assert context.request_adapter is None
    assert context.response_adapter is None
    assert context.request_transformer is None
    assert context.response_transformer is None
    assert context.supports_streaming is True  # default


@pytest.mark.asyncio
async def test_handler_config_with_adapters():
    """Test HandlerConfig with request/response adapters."""
    adapter = MockAdapter()

    context = HandlerConfig(
        request_adapter=adapter,
        response_adapter=adapter,
    )

    assert context.request_adapter == adapter
    assert context.response_adapter == adapter


@pytest.mark.asyncio
async def test_handler_config_with_custom_settings():
    """Test HandlerConfig with custom settings."""
    adapter = MockAdapter()
    transformer = MockTransformer()

    context = HandlerConfig(
        request_adapter=adapter,
        response_adapter=adapter,
        request_transformer=transformer,
        response_transformer=transformer,
        supports_streaming=False,
    )

    assert context.request_adapter == adapter
    assert context.response_adapter == adapter
    assert context.request_transformer == transformer
    assert context.response_transformer == transformer
    assert context.supports_streaming is False


@pytest.mark.asyncio
async def test_auth_manager_interface():
    """Test AuthManager interface methods."""
    auth = MockAuthManager()

    # Test get_auth_headers
    headers = await auth.get_auth_headers()
    assert headers == {"authorization": "Bearer test-token"}

    # Test validate_credentials
    is_valid = await auth.validate_credentials()
    assert is_valid is True

    # Test get_provider_name
    provider_name = auth.get_provider_name()
    assert provider_name == "mock-provider"


def test_handler_config_with_transformer():
    """Test HandlerConfig with request transformer."""
    transformer = MockTransformer()

    context = HandlerConfig(
        request_transformer=transformer,
    )

    assert context.request_transformer == transformer

    # Test transformer works
    headers = {"x-original": "value"}
    transformed = context.request_transformer.transform_headers(headers)
    assert transformed == {"x-original": "value", "x-transformed": "true"}


def test_handler_config_defaults():
    """Test HandlerConfig uses correct defaults."""
    context = HandlerConfig()

    # Check all defaults
    assert context.request_adapter is None
    assert context.response_adapter is None
    assert context.request_transformer is None
    assert context.response_transformer is None
    assert context.supports_streaming is True


@pytest.mark.asyncio
async def test_multiple_handler_configs():
    """Test creating multiple HandlerConfig instances."""
    adapter1 = MockAdapter()
    adapter2 = MockAdapter()
    transformer1 = MockTransformer()
    transformer2 = MockTransformer()

    # Create context with streaming enabled
    streaming_context = HandlerConfig(
        request_adapter=adapter1,
        response_adapter=adapter1,
        request_transformer=transformer1,
        supports_streaming=True,
    )

    # Create context with streaming disabled
    non_streaming_context = HandlerConfig(
        request_adapter=adapter2,
        response_adapter=adapter2,
        request_transformer=transformer2,
        supports_streaming=False,
    )

    # Verify they are independent
    assert streaming_context.request_adapter == adapter1
    assert non_streaming_context.request_adapter == adapter2
    assert streaming_context.supports_streaming is True
    assert non_streaming_context.supports_streaming is False


def test_handler_config_is_immutable():
    """Test that HandlerConfig is immutable (frozen dataclass)."""
    from dataclasses import FrozenInstanceError

    context = HandlerConfig(supports_streaming=True)

    # Attempting to modify should raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        context.supports_streaming = False  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        context.request_adapter = MockAdapter()  # type: ignore[misc]
