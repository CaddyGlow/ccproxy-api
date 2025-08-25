"""Integration tests for adapter-based request handling.

This replaces the deprecated dispatch_request tests with direct adapter testing.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import SecretStr

from ccproxy.auth.manager import AuthManager
from ccproxy.services.adapters.base import BaseAdapter


class MockAuthManager(AuthManager):
    """Mock authentication manager for testing."""

    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    async def get_access_token(self) -> str:
        """Get mock access token."""
        if self.should_fail:
            from ccproxy.auth.exceptions import AuthenticationError

            raise AuthenticationError("Mock auth failure")
        return "test-token"

    async def get_credentials(self) -> Any:
        """Get mock credentials."""
        if self.should_fail:
            from ccproxy.auth.exceptions import AuthenticationError

            raise AuthenticationError("Mock auth failure")
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
        """Check if authenticated."""
        return not self.should_fail

    async def get_user_profile(self) -> Any:
        """Get mock user profile."""
        return None

    async def get_auth_headers(self) -> dict[str, str]:
        """Get mock auth headers."""
        if self.should_fail:
            from ccproxy.auth.exceptions import AuthenticationError

            raise AuthenticationError("Mock auth failure")
        return {"x-api-key": "test-key"}

    async def validate_credentials(self) -> bool:
        """Mock validation."""
        return not self.should_fail

    def get_provider_name(self) -> str:
        """Get mock provider name."""
        return "mock-provider"

    async def __aenter__(self) -> "MockAuthManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""

    def __init__(self, response_data: dict[str, Any] | None = None):
        self.response_data = response_data or {"result": "success"}
        self.handle_request_called = False
        self.last_request: Request | None = None
        self.last_endpoint: str | None = None
        self.last_method: str | None = None

    async def handle_request(
        self, request: Request, endpoint: str, method: str, **kwargs: Any
    ) -> Response | StreamingResponse:
        """Handle a request."""
        self.handle_request_called = True
        self.last_request = request
        self.last_endpoint = endpoint
        self.last_method = method

        # Return a mock response
        return Response(
            content=json.dumps(self.response_data),
            status_code=200,
            headers={"content-type": "application/json"},
        )

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse:
        """Handle streaming request."""

        async def generate():
            yield b'data: {"chunk": 1}\n\n'
            yield b'data: {"chunk": 2}\n\n'
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    async def cleanup(self) -> None:
        """Cleanup mock adapter resources."""
        pass


class MockPluginRegistry:
    """Mock plugin registry for testing."""

    def __init__(self):
        self.adapters = {}

    def get_adapter(self, provider_name: str) -> BaseAdapter | None:
        """Get an adapter for a provider."""
        return self.adapters.get(provider_name)

    def add_adapter(self, provider_name: str, adapter: BaseAdapter) -> None:
        """Add an adapter for testing."""
        self.adapters[provider_name] = adapter


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/v1/messages"
    request.url.query = ""
    request.headers = {"content-type": "application/json"}
    request.body = AsyncMock(return_value=b'{"prompt": "test"}')
    return request


@pytest.fixture
def mock_plugin_registry():
    """Create a mock plugin registry."""
    return MockPluginRegistry()


@pytest.mark.asyncio
async def test_adapter_handle_request_success(mock_request):
    """Test successful request handling through adapter."""
    # Create adapter
    adapter = MockAdapter({"result": "adapter_response"})

    # Call handle_request directly
    response = await adapter.handle_request(
        mock_request, "/v1/messages", "POST", request_id="test-123"
    )

    # Verify the adapter was called
    assert adapter.handle_request_called
    assert adapter.last_request == mock_request
    assert adapter.last_endpoint == "/v1/messages"
    assert adapter.last_method == "POST"

    # Verify response
    assert isinstance(response, Response)
    assert response.status_code == 200
    response_body = response.body
    assert isinstance(response_body, bytes)
    assert json.loads(response_body) == {"result": "adapter_response"}


@pytest.mark.asyncio
async def test_adapter_handle_streaming(mock_request):
    """Test streaming request handling through adapter."""
    # Create adapter
    adapter = MockAdapter()

    # Call handle_streaming directly
    response = await adapter.handle_streaming(
        mock_request, "/v1/messages", request_id="test-123"
    )

    # Verify response is streaming
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"

    # Collect streamed data
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)

    # Verify streamed content
    assert len(chunks) == 3
    assert b'{"chunk": 1}' in chunks[0]
    assert b'{"chunk": 2}' in chunks[1]
    assert b"[DONE]" in chunks[2]


@pytest.mark.asyncio
async def test_plugin_registry_adapter_lookup(mock_plugin_registry):
    """Test adapter lookup through plugin registry."""
    # Add adapters to registry
    adapter1 = MockAdapter({"provider": "claude"})
    adapter2 = MockAdapter({"provider": "codex"})

    mock_plugin_registry.add_adapter("claude", adapter1)
    mock_plugin_registry.add_adapter("codex", adapter2)

    # Test lookup
    claude_adapter = mock_plugin_registry.get_adapter("claude")
    assert claude_adapter == adapter1

    codex_adapter = mock_plugin_registry.get_adapter("codex")
    assert codex_adapter == adapter2

    # Test missing adapter
    missing_adapter = mock_plugin_registry.get_adapter("missing")
    assert missing_adapter is None


@pytest.mark.asyncio
async def test_adapter_with_auth_manager(mock_request):
    """Test adapter with authentication manager."""
    auth_manager = MockAuthManager()

    class AuthAwareAdapter(MockAdapter):
        """Adapter that uses auth manager."""

        def __init__(self, auth_manager: AuthManager):
            super().__init__()
            self.auth_manager = auth_manager

        async def handle_request(
            self, request: Request, endpoint: str, method: str, **kwargs: Any
        ) -> Response:
            """Handle request with auth."""
            # Get auth token
            token = await self.auth_manager.get_access_token()

            # Add to response
            response_data = {
                **self.response_data,
                "auth_token": token,
            }

            return Response(
                content=json.dumps(response_data),
                status_code=200,
                headers={"content-type": "application/json"},
            )

    # Create adapter with auth
    adapter = AuthAwareAdapter(auth_manager)

    # Call handle_request
    response = await adapter.handle_request(mock_request, "/v1/messages", "POST")

    # Verify response includes auth token
    assert isinstance(response, Response)
    body_str = (
        response.body.decode()
        if isinstance(response.body, bytes)
        else str(response.body)
    )
    response_body = json.loads(body_str)
    assert response_body["auth_token"] == "test-token"


@pytest.mark.asyncio
async def test_adapter_auth_failure(mock_request):
    """Test adapter handling auth failure."""
    auth_manager = MockAuthManager(should_fail=True)

    class AuthAwareAdapter(MockAdapter):
        """Adapter that uses auth manager."""

        def __init__(self, auth_manager: AuthManager):
            super().__init__()
            self.auth_manager = auth_manager

        async def handle_request(
            self, request: Request, endpoint: str, method: str, **kwargs: Any
        ) -> Response:
            """Handle request with auth."""
            from ccproxy.auth.exceptions import AuthenticationError

            try:
                token = await self.auth_manager.get_access_token()
            except AuthenticationError:
                # Return 401 response
                return Response(
                    content=json.dumps({"error": "Authentication failed"}),
                    status_code=401,
                    headers={"content-type": "application/json"},
                )

            return Response(
                content=json.dumps(self.response_data),
                status_code=200,
            )

    # Create adapter with failing auth
    adapter = AuthAwareAdapter(auth_manager)

    # Call handle_request
    response = await adapter.handle_request(mock_request, "/v1/messages", "POST")

    # Verify 401 response
    assert isinstance(response, Response)
    assert response.status_code == 401
    body_str = (
        response.body.decode()
        if isinstance(response.body, bytes)
        else str(response.body)
    )
    response_body = json.loads(body_str)
    assert response_body["error"] == "Authentication failed"


@pytest.mark.asyncio
async def test_adapter_error_handling(mock_request):
    """Test adapter error handling."""

    class ErrorAdapter(MockAdapter):
        """Adapter that raises errors."""

        async def handle_request(
            self, request: Request, endpoint: str, method: str, **kwargs: Any
        ) -> Response:
            """Handle request with error."""
            raise HTTPException(status_code=500, detail="Internal error")

    # Create error adapter
    adapter = ErrorAdapter()

    # Call handle_request and expect exception
    with pytest.raises(HTTPException) as exc_info:
        await adapter.handle_request(mock_request, "/v1/messages", "POST")

    # Verify exception details
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal error"
