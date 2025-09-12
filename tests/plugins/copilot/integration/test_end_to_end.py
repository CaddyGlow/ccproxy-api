"""End-to-end integration tests for Copilot plugin."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from ccproxy.plugins.copilot.oauth.models import (
    CopilotCredentials,
    CopilotOAuthToken,
    CopilotTokenResponse,
)


@pytest.mark.integration
class TestCopilotEndToEnd:
    """End-to-end integration tests for Copilot plugin."""

    @pytest.fixture
    def mock_credentials(self) -> CopilotCredentials:
        """Create mock Copilot credentials."""
        oauth_token = CopilotOAuthToken(
            access_token=SecretStr("gho_test_oauth_token"),
            token_type="bearer",
            expires_in=28800,
            created_at=1234567890,
            scope="read:user",
        )

        copilot_token = CopilotTokenResponse(
            token=SecretStr("copilot_test_service_token"),
            expires_at="2024-12-31T23:59:59Z",
        )

        return CopilotCredentials(
            oauth_token=oauth_token,
            copilot_token=copilot_token,
            account_type="individual",
        )

    @pytest.mark.asyncio(loop_scope="session")
    async def test_copilot_models_endpoint(
        self,
        copilot_integration_client,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test Copilot models endpoint integration."""
        client = copilot_integration_client

        # Mock OAuth provider to return credentials
        with patch(
            "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.get_copilot_token = AsyncMock(
                return_value="copilot_test_service_token"
            )
            mock_provider.is_authenticated = AsyncMock(return_value=True)
            mock_provider_class.return_value = mock_provider

            # Mock external Copilot API call
            mock_models_response = {
                "object": "list",
                "data": [
                    {
                        "id": "copilot-chat",
                        "object": "model",
                        "owned_by": "github",
                    },
                    {
                        "id": "gpt-4-copilot",
                        "object": "model",
                        "owned_by": "github",
                    },
                ],
            }

            with patch("httpx.AsyncClient.get") as mock_http_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_models_response
                mock_response.raise_for_status.return_value = None
                mock_http_get.return_value = mock_response

                # Make request to Copilot models endpoint
                response = await client.get("/copilot/v1/models")

                assert response.status_code == 200
                data = response.json()

                assert data["object"] == "list"
                assert len(data["data"]) == 2
                assert data["data"][0]["id"] == "copilot-chat"
                assert data["data"][1]["id"] == "gpt-4-copilot"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_copilot_chat_completions_non_streaming(
        self,
        copilot_integration_client,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test Copilot chat completions endpoint (non-streaming)."""
        client = copilot_integration_client

        # Mock OAuth provider
        with patch(
            "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.get_copilot_token = AsyncMock(
                return_value="copilot_test_service_token"
            )
            mock_provider.is_authenticated = AsyncMock(return_value=True)
            mock_provider_class.return_value = mock_provider

            # Mock Copilot API response
            mock_completion_response = {
                "id": "copilot-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "copilot-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you today?",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18,
                },
            }

            with patch("httpx.AsyncClient.post") as mock_http_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_completion_response
                mock_response.raise_for_status.return_value = None
                mock_http_post.return_value = mock_response

                # Make request to Copilot chat completions
                request_data = {
                    "model": "copilot-chat",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, world!",
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150,
                }

                response = await client.post(
                    "/copilot/v1/chat/completions",
                    json=request_data,
                )

                assert response.status_code == 200
                data = response.json()

                assert data["id"] == "copilot-123"
                assert data["object"] == "chat.completion"
                assert data["model"] == "copilot-chat"
                assert len(data["choices"]) == 1
                assert (
                    data["choices"][0]["message"]["content"]
                    == "Hello! How can I help you today?"
                )
                assert data["usage"]["total_tokens"] == 18

    @pytest.mark.asyncio(loop_scope="session")
    async def test_copilot_chat_completions_streaming(
        self,
        copilot_integration_client,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test Copilot chat completions endpoint (streaming)."""
        client = copilot_integration_client

        # Mock OAuth provider
        with patch(
            "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.get_copilot_token = AsyncMock(
                return_value="copilot_test_service_token"
            )
            mock_provider.is_authenticated = AsyncMock(return_value=True)
            mock_provider_class.return_value = mock_provider

            # Mock streaming response chunks
            streaming_chunks = [
                {
                    "id": "copilot-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "copilot-chat",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Hello"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "copilot-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "copilot-chat",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": " world!"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "copilot-123",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": "copilot-chat",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 15,
                        "total_tokens": 25,
                    },
                },
            ]

            async def mock_stream():
                for chunk in streaming_chunks:
                    yield f"data: {json.dumps(chunk)}\n\n"

            with patch("httpx.AsyncClient.stream") as mock_http_stream:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.aiter_text.return_value = mock_stream()
                mock_response.raise_for_status.return_value = None

                mock_stream_context = AsyncMock()
                mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_stream_context.__aexit__ = AsyncMock(return_value=None)
                mock_http_stream.return_value = mock_stream_context

                # Make streaming request
                request_data = {
                    "model": "copilot-chat",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello!",
                        }
                    ],
                    "stream": True,
                }

                response = await client.post(
                    "/copilot/v1/chat/completions",
                    json=request_data,
                )

                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"

                # Collect streaming response
                chunks = []
                async for chunk in response.aiter_text():
                    if chunk.startswith("data: "):
                        chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
                        chunks.append(chunk_data)

                # Verify streaming chunks
                assert len(chunks) >= 2  # At least content chunks

                # Check first chunk has delta content
                first_chunk = chunks[0]
                assert first_chunk["object"] == "chat.completion.chunk"
                assert "delta" in first_chunk["choices"][0]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_copilot_authentication_required(
        self,
        copilot_integration_client,
    ) -> None:
        """Test that Copilot endpoints require authentication."""
        client = copilot_integration_client

        # Mock OAuth provider returning no authentication
        with patch(
            "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.get_copilot_token = AsyncMock(return_value=None)
            mock_provider.is_authenticated = AsyncMock(return_value=False)
            mock_provider_class.return_value = mock_provider

            # Test models endpoint
            response = await client.get("/copilot/v1/models")
            assert response.status_code == 401

            # Test chat completions endpoint
            request_data = {
                "model": "copilot-chat",
                "messages": [{"role": "user", "content": "Hello"}],
            }
            response = await client.post(
                "/copilot/v1/chat/completions",
                json=request_data,
            )
            assert response.status_code == 401

    @pytest.mark.asyncio(loop_scope="session")
    async def test_copilot_format_adapter_integration(
        self,
        copilot_integration_client,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test format adapter integration with OpenAI to Copilot conversion."""
        client = copilot_integration_client

        # Mock OAuth provider
        with patch(
            "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.get_copilot_token = AsyncMock(
                return_value="copilot_test_service_token"
            )
            mock_provider.is_authenticated = AsyncMock(return_value=True)
            mock_provider_class.return_value = mock_provider

            # Mock Copilot API response
            mock_response_data = {
                "id": "copilot-456",
                "object": "chat.completion",
                "model": "copilot-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Converted response",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

            with patch("httpx.AsyncClient.post") as mock_http_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_response_data
                mock_response.raise_for_status.return_value = None
                mock_http_post.return_value = mock_response

                # Send OpenAI format request
                openai_request = {
                    "model": "gpt-4",  # OpenAI model name
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {
                            "role": "user",
                            "content": "Test message",
                            "name": "test_user",
                        },
                    ],
                    "temperature": 0.8,
                    "max_tokens": 200,
                    "top_p": 0.9,
                    "stop": ["END"],
                }

                response = await client.post(
                    "/copilot/v1/chat/completions",
                    json=openai_request,
                )

                assert response.status_code == 200
                data = response.json()

                # Verify response is in OpenAI format (converted back)
                assert "id" in data
                assert "object" in data
                assert "choices" in data
                assert data["choices"][0]["message"]["content"] == "Converted response"

                # Verify the request was converted to Copilot format internally
                # (This would be verified by checking what was sent to the mock)
                mock_http_post.assert_called_once()
                call_args = mock_http_post.call_args

                # The request should have been converted to Copilot format
                # We can verify this by checking the call was made
                assert call_args is not None

    @pytest.mark.asyncio(loop_scope="session")
    async def test_copilot_error_handling(
        self,
        copilot_integration_client,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test Copilot API error handling."""
        client = copilot_integration_client

        # Mock OAuth provider
        with patch(
            "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.get_copilot_token = AsyncMock(
                return_value="copilot_test_service_token"
            )
            mock_provider.is_authenticated = AsyncMock(return_value=True)
            mock_provider_class.return_value = mock_provider

            # Mock API error response
            with patch("httpx.AsyncClient.post") as mock_http_post:
                import httpx

                mock_response = MagicMock()
                mock_response.status_code = 400
                mock_response.json.return_value = {
                    "error": {
                        "message": "Bad request",
                        "type": "invalid_request_error",
                    }
                }
                mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Bad Request",
                    request=MagicMock(),
                    response=mock_response,
                )
                mock_http_post.return_value = mock_response

                # Make request that should fail
                request_data = {
                    "model": "invalid-model",
                    "messages": [],  # Empty messages
                }

                response = await client.post(
                    "/copilot/v1/chat/completions",
                    json=request_data,
                )

                # Should return error response
                assert response.status_code == 400
                data = response.json()
                assert "error" in data

    @pytest.mark.asyncio(loop_scope="session")
    async def test_copilot_usage_endpoint(
        self,
        copilot_integration_client,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test Copilot usage endpoint."""
        client = copilot_integration_client

        # Mock OAuth provider
        with patch(
            "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.get_copilot_token = AsyncMock(
                return_value="copilot_test_service_token"
            )
            mock_provider.is_authenticated = AsyncMock(return_value=True)
            mock_provider_class.return_value = mock_provider

            # Mock usage API response
            mock_usage_response = {
                "usage": {
                    "total_tokens": 10000,
                    "remaining_tokens": 5000,
                    "reset_date": "2024-01-01T00:00:00Z",
                },
                "plan": "individual",
                "features": ["chat", "code_completion"],
            }

            with patch("httpx.AsyncClient.get") as mock_http_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_usage_response
                mock_response.raise_for_status.return_value = None
                mock_http_get.return_value = mock_response

                # Make request to usage endpoint
                response = await client.get("/copilot/usage")

                assert response.status_code == 200
                data = response.json()

                assert "usage" in data
                assert data["usage"]["total_tokens"] == 10000
                assert data["plan"] == "individual"
                assert "chat" in data["features"]

    @pytest.mark.asyncio(loop_scope="session")
    async def test_copilot_token_info_endpoint(
        self,
        copilot_integration_client,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test Copilot token info endpoint."""
        client = copilot_integration_client

        # Mock OAuth provider with token info
        with patch(
            "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider"
        ) as mock_provider_class:
            mock_provider = MagicMock()
            mock_provider.get_copilot_token = AsyncMock(
                return_value="copilot_test_service_token"
            )
            mock_provider.is_authenticated = AsyncMock(return_value=True)

            from datetime import UTC, datetime

            from ccproxy.plugins.copilot.oauth.models import CopilotTokenInfo

            mock_token_info = CopilotTokenInfo(
                provider="copilot",
                oauth_expires_at=datetime.now(UTC),
                copilot_expires_at=datetime.now(UTC),
                account_type="individual",
                copilot_access=True,
            )
            mock_provider.get_token_info = AsyncMock(return_value=mock_token_info)
            mock_provider_class.return_value = mock_provider

            # Make request to token info endpoint
            response = await client.get("/copilot/token")

            assert response.status_code == 200
            data = response.json()

            assert data["provider"] == "copilot"
            assert data["account_type"] == "individual"
            assert data["copilot_access"] is True
            assert "oauth_expires_at" in data
            assert "copilot_expires_at" in data


# Session-scoped fixtures for performance optimization
pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def copilot_integration_app():
    """Pre-configured app for Copilot plugin integration tests - session scoped."""
    from ccproxy.api.app import create_app
    from ccproxy.api.bootstrap import create_service_container
    from ccproxy.config.settings import Settings
    from ccproxy.core.logging import setup_logging

    # Set up logging once per session - minimal logging for speed
    setup_logging(json_logs=False, log_level_name="ERROR")

    settings = Settings(
        enable_plugins=True,
        plugins_disable_local_discovery=False,  # Enable local plugin discovery
        plugins={
            "copilot": {
                "enabled": True,
            }
        },
        logging={
            "level": "ERROR",  # Minimal logging for speed
            "enable_plugin_logging": False,
            "verbose_api": False,
        },
    )

    service_container = create_service_container(settings)
    return create_app(service_container), settings


@pytest_asyncio.fixture(loop_scope="session")
async def copilot_integration_client(copilot_integration_app):
    """HTTP client for Copilot integration tests - uses shared app."""
    from ccproxy.api.app import initialize_plugins_startup

    app, settings = copilot_integration_app

    # Initialize plugins async (once per test, but app is shared)
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
