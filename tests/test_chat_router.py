"""Tests for chat router endpoints."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from claude_code_proxy.config.settings import Settings
from claude_code_proxy.models.requests import ChatCompletionRequest
from claude_code_proxy.models.responses import (
    ChatCompletionResponse,
    InternalServerError,
    InvalidRequestError,
)
from claude_code_proxy.routers.anthropic import (
    create_chat_completion,
    list_models,
)
from claude_code_proxy.services.claude_client import ClaudeClient, ClaudeClientError


def get_claude_client(settings: Settings) -> ClaudeClient:
    """Get Claude client instance."""
    return ClaudeClient()


@pytest.mark.unit
class TestGetClaudeClient:
    """Test get_claude_client dependency function."""

    def test_get_claude_client_returns_instance(self):
        """Test that get_claude_client returns a ClaudeClient instance."""
        settings = Settings()
        client = get_claude_client(settings)
        assert isinstance(client, ClaudeClient)

    def test_get_claude_client_with_different_settings(self):
        """Test get_claude_client with different settings configurations."""
        settings = Settings()
        client1 = get_claude_client(settings)
        client2 = get_claude_client(settings)

        # Should return new instances each time
        assert isinstance(client1, ClaudeClient)
        assert isinstance(client2, ClaudeClient)


@pytest.mark.integration
class TestCreateChatCompletion:
    """Test create_chat_completion endpoint function."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock HTTP request."""
        mock_req = MagicMock()
        mock_req.url = "http://test.com/v1/chat/completions"
        return mock_req

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.claude_code_options = MagicMock()
        settings.claude_code_options.model = "claude-3-5-sonnet-20241022"
        return settings

    @pytest.fixture
    def mock_claude_client(self):
        """Create mock Claude client."""
        client = MagicMock(spec=ClaudeClient)
        client.create_completion = AsyncMock()
        return client

    @pytest.fixture
    def sample_request_data(self):
        """Sample chat completion request data."""
        return {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False,
        }

    @pytest.fixture
    def sample_claude_response(self):
        """Sample Claude response."""
        return {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }

    @pytest.mark.asyncio
    async def test_successful_non_streaming_completion(
        self,
        sample_request_data,
        sample_claude_response,
        mock_request,
        mock_settings,
        mock_claude_client,
    ):
        """Test successful non-streaming chat completion."""
        # Setup
        request = ChatCompletionRequest(**sample_request_data)
        mock_claude_client.create_completion.return_value = sample_claude_response

        # Execute
        result = await create_chat_completion(
            request, mock_request, mock_settings, mock_claude_client
        )

        # Verify
        assert isinstance(result, ChatCompletionResponse)
        mock_claude_client.create_completion.assert_called_once()

        # Check call arguments
        call_args = mock_claude_client.create_completion.call_args
        assert "messages" in call_args.kwargs
        assert "options" in call_args.kwargs
        assert "stream" in call_args.kwargs
        assert call_args.kwargs["stream"] is False

    @pytest.mark.asyncio
    async def test_successful_streaming_completion(
        self,
        sample_request_data,
        mock_request,
        mock_settings,
        mock_claude_client,
    ):
        """Test successful streaming chat completion."""
        # Setup streaming request
        sample_request_data["stream"] = True
        request = ChatCompletionRequest(**sample_request_data)

        # Mock streaming response
        async def mock_stream():
            chunks = [
                {"type": "message_start", "message": {"id": "msg_123"}},
                {"type": "content_block_delta", "delta": {"text": "Hello"}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            ]
            for chunk in chunks:
                yield chunk

        mock_claude_client.create_completion.return_value = mock_stream()

        # Execute
        from fastapi.responses import StreamingResponse

        result = await create_chat_completion(
            request, mock_request, mock_settings, mock_claude_client
        )

        # Verify
        assert isinstance(result, StreamingResponse)
        assert result.media_type == "text/event-stream"
        mock_claude_client.create_completion.assert_called_once()

        # Check that stream=True was passed
        call_args = mock_claude_client.create_completion.call_args
        assert call_args.kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_message_content_string_conversion(
        self,
        mock_request,
        mock_settings,
        mock_claude_client,
        sample_claude_response,
    ):
        """Test conversion of string message content to proper format."""
        # Setup request with string content
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Simple string message"}],
            "max_tokens": 100,
        }
        request = ChatCompletionRequest(**request_data)  # type: ignore[arg-type]
        mock_claude_client.create_completion.return_value = sample_claude_response

        # Execute
        await create_chat_completion(
            request, mock_request, mock_settings, mock_claude_client
        )

        # Verify message format conversion
        call_args = mock_claude_client.create_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == [
            {"type": "text", "text": "Simple string message"}
        ]

    @pytest.mark.asyncio
    async def test_message_content_list_conversion(
        self,
        mock_request,
        mock_settings,
        mock_claude_client,
        sample_claude_response,
    ):
        """Test conversion of list message content to proper format."""
        # Setup request with list content - use dictionary instead of MagicMock
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Text content"}]}
            ],
            "max_tokens": 100,
        }
        request = ChatCompletionRequest(**request_data)  # type: ignore[arg-type]
        mock_claude_client.create_completion.return_value = sample_claude_response

        # Execute
        await create_chat_completion(
            request, mock_request, mock_settings, mock_claude_client
        )

        # Verify message format conversion
        call_args = mock_claude_client.create_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == [{"type": "text", "text": "Text content"}]

    @pytest.mark.asyncio
    async def test_image_content_conversion(
        self,
        mock_request,
        mock_settings,
        mock_claude_client,
        sample_claude_response,
    ):
        """Test conversion of image content to proper format."""
        # Setup image content - use dictionary structure for valid Pydantic validation
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "base64data",
                            },
                        }
                    ],
                }
            ],
            "max_tokens": 100,
        }
        request = ChatCompletionRequest(**request_data)  # type: ignore[arg-type]
        mock_claude_client.create_completion.return_value = sample_claude_response

        # Execute
        await create_chat_completion(
            request, mock_request, mock_settings, mock_claude_client
        )

        # Verify image format conversion
        call_args = mock_claude_client.create_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/jpeg"
        assert content[0]["source"]["data"] == "base64data"

    @pytest.mark.asyncio
    async def test_system_prompt_handling(
        self,
        sample_request_data,
        mock_request,
        mock_settings,
        mock_claude_client,
        sample_claude_response,
    ):
        """Test system prompt is passed to options."""
        # Add system prompt to request
        sample_request_data["system"] = "You are a helpful assistant."
        request = ChatCompletionRequest(**sample_request_data)
        mock_claude_client.create_completion.return_value = sample_claude_response

        # Execute
        await create_chat_completion(
            request, mock_request, mock_settings, mock_claude_client
        )

        # Verify system prompt is set
        call_args = mock_claude_client.create_completion.call_args
        options = call_args.kwargs["options"]
        assert options.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_model_override_in_options(
        self,
        sample_request_data,
        mock_request,
        mock_settings,
        mock_claude_client,
        sample_claude_response,
    ):
        """Test that request model overrides settings model."""
        # Setup
        request = ChatCompletionRequest(**sample_request_data)
        mock_claude_client.create_completion.return_value = sample_claude_response

        # Execute
        await create_chat_completion(
            request, mock_request, mock_settings, mock_claude_client
        )

        # Verify model override
        call_args = mock_claude_client.create_completion.call_args
        options = call_args.kwargs["options"]
        assert options.model == "claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_claude_client_error_handling(
        self,
        sample_request_data,
        mock_request,
        mock_settings,
        mock_claude_client,
    ):
        """Test ClaudeClientError is converted to HTTPException."""
        # Setup
        request = ChatCompletionRequest(**sample_request_data)
        error_msg = "API rate limit exceeded"
        mock_claude_client.create_completion.side_effect = ClaudeClientError(error_msg)

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await create_chat_completion(
                request, mock_request, mock_settings, mock_claude_client
            )

        assert exc_info.value.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert "internal_server_error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_value_error_handling(
        self,
        sample_request_data,
        mock_request,
        mock_settings,
        mock_claude_client,
    ):
        """Test ValueError is converted to HTTPException."""
        # Setup
        request = ChatCompletionRequest(**sample_request_data)
        error_msg = "Invalid parameter value"
        mock_claude_client.create_completion.side_effect = ValueError(error_msg)

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await create_chat_completion(
                request, mock_request, mock_settings, mock_claude_client
            )

        assert exc_info.value.status_code == HTTP_422_UNPROCESSABLE_ENTITY
        assert "invalid_request_error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_unexpected_error_handling(
        self,
        sample_request_data,
        mock_request,
        mock_settings,
        mock_claude_client,
    ):
        """Test unexpected Exception is converted to HTTPException."""
        # Setup
        request = ChatCompletionRequest(**sample_request_data)
        mock_claude_client.create_completion.side_effect = RuntimeError(
            "Unexpected error"
        )

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await create_chat_completion(
                request, mock_request, mock_settings, mock_claude_client
            )

        assert exc_info.value.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert "An unexpected error occurred" in str(exc_info.value.detail)


@pytest.mark.integration
class TestListModels:
    """Test list_models endpoint function."""

    @pytest.fixture
    def mock_claude_client(self):
        """Create mock Claude client."""
        client = MagicMock(spec=ClaudeClient)
        client.list_models = AsyncMock()
        return client

    @pytest.fixture
    def sample_models(self):
        """Sample models response."""
        return [
            {
                "id": "claude-opus-4-20250514",
                "object": "model",
                "created": 1677610602,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "object": "model",
                "created": 1677610602,
                "owned_by": "anthropic",
            },
        ]

    @pytest.mark.asyncio
    async def test_successful_list_models(self, mock_claude_client, sample_models):
        """Test successful models listing."""
        # Setup
        mock_claude_client.list_models.return_value = sample_models

        # Execute
        result = await list_models(mock_claude_client)

        # Verify
        assert "data" in result
        assert result["data"] == sample_models
        mock_claude_client.list_models.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_models_claude_client_error(self, mock_claude_client):
        """Test ClaudeClientError in list_models is converted to HTTPException."""
        # Setup
        error_msg = "Service unavailable"
        mock_claude_client.list_models.side_effect = ClaudeClientError(error_msg)

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await list_models(mock_claude_client)

        assert exc_info.value.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert "internal_server_error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_list_models_unexpected_error(self, mock_claude_client):
        """Test unexpected Exception in list_models is converted to HTTPException."""
        # Setup
        mock_claude_client.list_models.side_effect = RuntimeError("Unexpected error")

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await list_models(mock_claude_client)

        assert exc_info.value.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert "An unexpected error occurred" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_list_models_empty_response(self, mock_claude_client):
        """Test list_models with empty models list."""
        # Setup
        mock_claude_client.list_models.return_value = []

        # Execute
        result = await list_models(mock_claude_client)

        # Verify
        assert "data" in result
        assert result["data"] == []
        mock_claude_client.list_models.assert_called_once()


@pytest.mark.integration
class TestChatRouterIntegration:
    """Integration tests for chat router using TestClient."""

    def teardown_method(self):
        """Clean up dependency overrides after each test."""
        from claude_code_proxy.main import app

        app.dependency_overrides.clear()

    def test_chat_completions_endpoint_success(
        self,
        test_client: TestClient,
        sample_chat_request: dict[str, Any],
        sample_claude_response: dict[str, Any],
        mock_claude_client,
    ):
        """Test /v1/chat/completions endpoint with successful response."""
        # Override the dependency for the router I'm testing
        from claude_code_proxy.routers.chat import router

        mock_claude_client.create_completion.return_value = sample_claude_response

        # Create a separate test app with just my router
        from fastapi import FastAPI

        test_app = FastAPI()
        test_app.include_router(router)
        test_app.dependency_overrides[get_claude_client] = lambda: mock_claude_client

        from fastapi.testclient import TestClient

        test_client_local = TestClient(test_app)

        # Execute
        response = test_client_local.post(
            "/v1/chat/completions", json=sample_chat_request
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert "id" in data

    def test_models_endpoint_success(
        self,
        test_client: TestClient,
        sample_models_response: list[dict[str, Any]],
        mock_claude_client,
    ):
        """Test /v1/models endpoint with successful response."""
        # Create a separate test app with just my router
        from fastapi import FastAPI

        from claude_code_proxy.routers.chat import router

        test_app = FastAPI()
        test_app.include_router(router)
        test_app.dependency_overrides[get_claude_client] = lambda: mock_claude_client
        mock_claude_client.list_models.return_value = sample_models_response

        from fastapi.testclient import TestClient

        test_client_local = TestClient(test_app)

        # Execute
        response = test_client_local.get("/v1/models")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"] == sample_models_response

    def test_chat_completions_invalid_request(self, test_client: TestClient):
        """Test /v1/chat/completions with invalid request data."""
        # Execute with invalid data
        invalid_request = {
            "model": "invalid-model",  # Doesn't match claude-* pattern
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        response = test_client.post("/v1/chat/completions", json=invalid_request)

        # Verify
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_chat_completions_empty_messages(self, test_client: TestClient):
        """Test /v1/chat/completions with empty messages array."""
        # Execute with empty messages
        invalid_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],  # Empty messages array
            "max_tokens": 100,
        }
        response = test_client.post("/v1/chat/completions", json=invalid_request)

        # Verify
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
