"""Test HTTP transformer logic for request and response transformations.

This module provides comprehensive tests for HTTPRequestTransformer and HTTPResponseTransformer
classes, covering all transformation methods including path transformation, header creation,
body transformation, system prompt injection, and OpenAI format detection/conversion.

Tests follow the TESTING.md requirements with proper type hints and no internal mocks.
"""

import json
from typing import Any, cast
from unittest.mock import patch

import pytest

from ccproxy.core.http_transformers import (
    HTTPRequestTransformer,
    HTTPResponseTransformer,
    get_detected_system_field,
    get_fallback_system_field,
)
from ccproxy.core.types import (
    ProxyMethod,
    ProxyProtocol,
    ProxyRequest,
    ProxyResponse,
    TransformContext,
)


class TestHTTPRequestTransformer:
    """Test HTTP request transformer functionality."""

    @pytest.fixture
    def request_transformer(self) -> HTTPRequestTransformer:
        """Create HTTP request transformer instance for testing."""
        return HTTPRequestTransformer()

    def test_transform_path_openai_chat_completions(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test path transformation for OpenAI chat completions endpoint."""
        result = request_transformer.transform_path("/v1/chat/completions")
        assert result == "/v1/messages"

    def test_transform_path_openai_prefix_removal(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test removal of /openai prefix from paths."""
        result = request_transformer.transform_path("/openai/v1/chat/completions")
        assert result == "/v1/messages"

    def test_transform_path_api_prefix_removal(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test removal of /api prefix from paths."""
        result = request_transformer.transform_path("/api/v1/messages")
        assert result == "/v1/messages"

    def test_transform_path_anthropic_messages_passthrough(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test that Anthropic messages path passes through unchanged."""
        result = request_transformer.transform_path("/v1/messages")
        assert result == "/v1/messages"

    def test_transform_path_models_endpoint(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test that models endpoint passes through unchanged."""
        result = request_transformer.transform_path("/v1/models")
        assert result == "/v1/models"

    def test_create_proxy_headers_basic_functionality(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test basic proxy header creation functionality."""
        original_headers = {
            "Content-Type": "application/json",
            "User-Agent": "test-client",
        }
        access_token = "test-token-123"

        result = request_transformer.create_proxy_headers(
            original_headers, access_token
        )

        # Check authentication header
        assert result["Authorization"] == "Bearer test-token-123"

        # Check Claude CLI identity headers
        assert result["x-app"] == "cli"
        assert result["User-Agent"] == "claude-cli/1.0.60 (external, cli)"

        # Check Anthropic API headers
        assert "anthropic-beta" in result
        assert "claude-code-20250219" in result["anthropic-beta"]
        assert result["anthropic-version"] == "2023-06-01"
        assert result["anthropic-dangerous-direct-browser-access"] == "true"

        # Check Stainless SDK headers
        assert result["X-Stainless-Lang"] == "js"
        assert result["X-Stainless-Package-Version"] == "0.55.1"

    def test_create_proxy_headers_excludes_problematic_headers(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test that problematic headers are excluded from proxy headers."""
        original_headers = {
            "Host": "localhost:8000",
            "Authorization": "Bearer old-token",
            "X-Api-Key": "old-key",
            "X-Forwarded-For": "127.0.0.1",
            "Content-Type": "application/json",
        }
        access_token = "new-token-456"

        result = request_transformer.create_proxy_headers(
            original_headers, access_token
        )

        # Ensure problematic headers are excluded
        assert "Host" not in result
        assert "X-Forwarded-For" not in result
        assert "X-Api-Key" not in result

        # Ensure Authorization is replaced with new token
        assert result["Authorization"] == "Bearer new-token-456"

        # Ensure safe headers are preserved
        assert result["Content-Type"] == "application/json"

    def test_create_proxy_headers_sets_default_headers(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test that default headers are set when missing."""
        original_headers: dict[str, str] = {}
        access_token = "test-token"

        result = request_transformer.create_proxy_headers(
            original_headers, access_token
        )

        # Check default headers are set
        assert result["Content-Type"] == "application/json"
        assert result["Accept"] == "application/json"
        assert result["Connection"] == "keep-alive"

    def test_create_proxy_headers_without_access_token(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test proxy header creation without access token."""
        original_headers = {"Content-Type": "application/json"}
        access_token = ""

        result = request_transformer.create_proxy_headers(
            original_headers, access_token
        )

        # Should not have Authorization header
        assert "Authorization" not in result

        # Should still have Claude CLI headers
        assert result["x-app"] == "cli"
        assert "anthropic-beta" in result

    def test_create_proxy_headers_excludes_compression_headers(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test that compression headers are excluded from proxy headers."""
        original_headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Encoding": "gzip",
            "User-Agent": "test-client",
        }
        access_token = "test-token-123"

        result = request_transformer.create_proxy_headers(
            original_headers, access_token
        )

        # Should exclude compression headers to prevent decompression issues
        assert "Accept-Encoding" not in result
        assert "accept-encoding" not in result
        assert "Content-Encoding" not in result
        assert "content-encoding" not in result

        # Should preserve safe headers
        assert result["Content-Type"] == "application/json"
        assert result["Authorization"] == "Bearer test-token-123"

    def test_create_proxy_headers_excludes_compression_headers_case_insensitive(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test that compression headers are excluded case-insensitively."""
        original_headers = {
            "Content-Type": "application/json",
            "accept-encoding": "gzip",  # lowercase
            "ACCEPT-ENCODING": "deflate",  # uppercase
            "Accept-Encoding": "br",  # mixed case
            "content-encoding": "gzip",  # lowercase
            "CONTENT-ENCODING": "deflate",  # uppercase
            "Content-Encoding": "br",  # mixed case
        }
        access_token = "test-token"

        result = request_transformer.create_proxy_headers(
            original_headers, access_token
        )

        # Should exclude all variations of compression headers
        assert "accept-encoding" not in result
        assert "ACCEPT-ENCODING" not in result
        assert "Accept-Encoding" not in result
        assert "content-encoding" not in result
        assert "CONTENT-ENCODING" not in result
        assert "Content-Encoding" not in result

        # Should preserve safe headers
        assert result["Content-Type"] == "application/json"

    def test_transform_system_prompt_no_existing_system(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation when no system prompt exists."""
        body_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data).encode("utf-8")

        result = request_transformer.transform_system_prompt(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should inject Claude Code system prompt
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        assert len(result_data["system"]) == 1
        assert (
            result_data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert result_data["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_transform_system_prompt_string_system_existing(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation with existing string system prompt."""
        body_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data).encode("utf-8")

        result = request_transformer.transform_system_prompt(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should prepend Claude Code prompt to existing system
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        assert len(result_data["system"]) == 2
        assert (
            result_data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert result_data["system"][1]["text"] == "You are a helpful assistant."

    def test_transform_system_prompt_array_system_existing(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation with existing array system prompt."""
        body_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "You are a helpful assistant."}],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data).encode("utf-8")

        result = request_transformer.transform_system_prompt(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should prepend Claude Code prompt
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        assert len(result_data["system"]) == 2
        assert (
            result_data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert result_data["system"][1]["text"] == "You are a helpful assistant."

    def test_transform_system_prompt_already_has_claude_code(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation when Claude Code prompt already exists."""
        body_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": [
                {
                    "type": "text",
                    "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                },
                {"type": "text", "text": "Additional instructions"},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data).encode("utf-8")

        result = request_transformer.transform_system_prompt(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should prepend Claude Code prompt with cache control and keep original structure
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        assert len(result_data["system"]) == 3
        assert (
            result_data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert result_data["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert (
            result_data["system"][1]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert result_data["system"][2]["text"] == "Additional instructions"

    def test_transform_system_prompt_invalid_json(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation with invalid JSON."""
        body = b"invalid json content"

        result = request_transformer.transform_system_prompt(body)

        # Should return original body unchanged
        assert result == body

    def test_transform_system_prompt_minimal_mode(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation in minimal mode."""
        body_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data).encode("utf-8")

        result = request_transformer.transform_system_prompt(
            body, injection_mode="minimal"
        )
        result_data = json.loads(result.decode("utf-8"))

        # Should prepend only Claude Code prompt in minimal mode
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        assert len(result_data["system"]) == 2
        assert (
            result_data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert result_data["system"][1]["text"] == "You are a helpful assistant."

    def test_transform_system_prompt_full_mode_with_app_state(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation in full mode with app state."""
        # Mock app state with detected system prompts
        from ccproxy.models.detection import SystemPromptData

        mock_app_state = type("MockAppState", (), {})()
        mock_claude_data = type("MockClaudeData", (), {})()
        mock_claude_data.system_prompt = SystemPromptData(
            system_field=[
                {
                    "type": "text",
                    "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Additional context from Claude CLI."},
                {"type": "text", "text": "More system instructions."},
            ]
        )
        mock_app_state.claude_detection_data = mock_claude_data

        body_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data).encode("utf-8")

        result = request_transformer.transform_system_prompt(
            body, mock_app_state, injection_mode="full"
        )
        result_data = json.loads(result.decode("utf-8"))

        # Should prepend all detected system prompts in full mode
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        assert len(result_data["system"]) == 4  # 3 detected + 1 original
        assert (
            result_data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert result_data["system"][1]["text"] == "Additional context from Claude CLI."
        assert result_data["system"][2]["text"] == "More system instructions."
        assert result_data["system"][3]["text"] == "You are a helpful assistant."

    def test_transform_system_prompt_full_mode_no_app_state(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation in full mode without app state."""
        body_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data).encode("utf-8")

        result = request_transformer.transform_system_prompt(
            body, injection_mode="full"
        )
        result_data = json.loads(result.decode("utf-8"))

        # Should fall back to minimal behavior when no app state
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        assert len(result_data["system"]) == 2
        assert (
            result_data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert result_data["system"][1]["text"] == "You are a helpful assistant."

    def test_is_openai_request_path_based_detection(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test OpenAI request detection based on path."""
        body = b'{"model": "claude-3-5-sonnet-20241022"}'

        # Test OpenAI-specific paths
        assert (
            request_transformer._is_openai_request("/openai/v1/chat/completions", body)
            is True
        )
        assert (
            request_transformer._is_openai_request("/v1/chat/completions", body) is True
        )

        # Test Anthropic paths
        assert request_transformer._is_openai_request("/v1/messages", body) is False
        assert request_transformer._is_openai_request("/v1/models", body) is False

    def test_is_openai_request_model_based_detection(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test OpenAI request detection based on model name."""
        path = "/v1/messages"

        # Test OpenAI models
        openai_models = ["gpt-4", "gpt-3.5-turbo", "o1-preview", "text-davinci-003"]
        for model in openai_models:
            body = json.dumps({"model": model}).encode("utf-8")
            assert request_transformer._is_openai_request(path, body) is True

        # Test Anthropic models
        anthropic_body = json.dumps({"model": "claude-3-5-sonnet-20241022"}).encode(
            "utf-8"
        )
        assert request_transformer._is_openai_request(path, anthropic_body) is False

    def test_is_openai_request_message_format_detection(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test OpenAI request detection based on message format."""
        path = "/v1/messages"

        # Test OpenAI format with system message in messages array
        openai_body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ],
            }
        ).encode("utf-8")
        assert request_transformer._is_openai_request(path, openai_body) is True

        # Test Anthropic format with separate system field
        anthropic_body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "system": "You are helpful",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")
        assert request_transformer._is_openai_request(path, anthropic_body) is False

    def test_is_openai_request_invalid_json(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test OpenAI request detection with invalid JSON body."""
        path = "/v1/messages"
        body = b"invalid json"

        result = request_transformer._is_openai_request(path, body)
        assert result is False

    @patch("ccproxy.adapters.openai.adapter.OpenAIAdapter")
    def test_transform_openai_to_anthropic_success(
        self, mock_adapter_class: Any, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test successful OpenAI to Anthropic transformation."""
        # Setup mock adapter
        mock_adapter = mock_adapter_class.return_value
        mock_adapter.adapt_request.return_value = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        openai_body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        ).encode("utf-8")

        result = request_transformer._transform_openai_to_anthropic(openai_body)
        result_data = json.loads(result.decode("utf-8"))

        # Should use adapter to transform
        mock_adapter.adapt_request.assert_called_once()
        assert result_data["model"] == "claude-3-5-sonnet-20241022"
        assert "max_tokens" in result_data

    @patch("ccproxy.adapters.openai.adapter.OpenAIAdapter")
    def test_transform_openai_to_anthropic_failure(
        self, mock_adapter_class: Any, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test OpenAI to Anthropic transformation failure handling."""
        # Setup mock adapter to raise exception
        mock_adapter_class.side_effect = Exception("Transformation failed")

        original_body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        ).encode("utf-8")

        result = request_transformer._transform_openai_to_anthropic(original_body)

        # Should return original body on failure
        assert result == original_body

    def test_transform_request_body_openai_detection_and_transformation(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test request body transformation with OpenAI detection."""
        path = "/v1/chat/completions"
        openai_body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        ).encode("utf-8")

        with (
            patch.object(request_transformer, "_is_openai_request", return_value=True),
            patch.object(
                request_transformer, "_transform_openai_to_anthropic"
            ) as mock_transform,
            patch.object(request_transformer, "transform_system_prompt") as mock_system,
        ):
            mock_transform.return_value = b'{"transformed": true}'
            mock_system.return_value = b'{"final": true}'

            result = request_transformer.transform_request_body(openai_body, path)

            # Should detect OpenAI and transform
            mock_transform.assert_called_once_with(openai_body)
            mock_system.assert_called_once_with(
                b'{"transformed": true}', None, "minimal"
            )
            assert result == b'{"final": true}'

    def test_transform_request_body_anthropic_passthrough(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test request body transformation for Anthropic requests."""
        path = "/v1/messages"
        anthropic_body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")

        with (
            patch.object(request_transformer, "_is_openai_request", return_value=False),
            patch.object(request_transformer, "transform_system_prompt") as mock_system,
        ):
            mock_system.return_value = b'{"system_transformed": true}'

            result = request_transformer.transform_request_body(anthropic_body, path)

            # Should only apply system prompt transformation
            mock_system.assert_called_once_with(anthropic_body, None, "minimal")
            assert result == b'{"system_transformed": true}'

    def test_transform_request_body_empty_body(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test request body transformation with empty body."""
        path = "/v1/messages"
        empty_body = b""

        result = request_transformer.transform_request_body(empty_body, path)

        # Should return empty body unchanged
        assert result == empty_body

    async def test_transform_request_full_integration(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test full request transformation integration."""
        # Create a proxy request
        request = ProxyRequest(
            method=ProxyMethod.POST,
            url="http://localhost:8000/openai/v1/chat/completions?param=value",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer old-token",
            },
            params={"param": "value"},
            body=json.dumps(
                {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
            ).encode("utf-8"),
            protocol=ProxyProtocol.HTTPS,
            timeout=30,
            metadata={"client_id": "test"},
        )

        # Create context with access token as dict (per implementation)
        context: dict[str, str] = {"access_token": "new-access-token"}

        # Transform the request
        result = await request_transformer._transform_request(
            request, cast(TransformContext, context)
        )

        # Check URL transformation (current implementation behavior)
        assert "https://api.anthropic.com" in result.url
        assert "param=value" in result.url

        # Check headers - should have access token from context
        assert result.headers["Authorization"] == "Bearer new-access-token"
        assert result.headers["x-app"] == "cli"
        assert "anthropic-beta" in result.headers

        # Check body transformation occurred
        assert result.body is not None
        if isinstance(result.body, bytes):
            body_data = json.loads(result.body.decode("utf-8"))
            # Should have Claude Code system prompt
            assert "system" in body_data

        # Check other attributes
        assert result.method == "POST"
        assert result.params == {}  # Should be empty as params are in URL
        assert result.metadata == {"client_id": "test"}


class TestHTTPResponseTransformer:
    """Test HTTP response transformer functionality."""

    @pytest.fixture
    def response_transformer(self) -> HTTPResponseTransformer:
        """Create HTTP response transformer instance for testing."""
        return HTTPResponseTransformer()

    def test_transform_response_body_passthrough(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test response body transformation passes through unchanged."""
        original_body = b'{"message": "Hello", "id": "msg_123"}'
        path = "/v1/messages"

        result = response_transformer.transform_response_body(original_body, path)

        # Currently just passes through
        assert result == original_body

    def test_transform_response_headers_basic_functionality(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test basic response header transformation."""
        original_headers = {
            "Content-Type": "application/json",
            "Content-Length": "100",
            "Server": "anthropic-api",
            "Transfer-Encoding": "chunked",
        }
        path = "/v1/messages"
        content_length = 150

        result = response_transformer.transform_response_headers(
            original_headers, path, content_length
        )

        # Should update content length
        assert result["Content-Length"] == "150"

        # Should preserve safe headers
        assert result["Content-Type"] == "application/json"
        assert result["Server"] == "anthropic-api"

        # Should exclude problematic headers
        assert "Transfer-Encoding" not in result

    def test_transform_response_headers_preserves_important_headers(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test that important headers are preserved in transformation."""
        original_headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "X-RateLimit-Remaining": "100",
            "X-Request-ID": "req_123",
        }
        path = "/v1/messages"
        content_length = 50

        result = response_transformer.transform_response_headers(
            original_headers, path, content_length
        )

        # Should preserve all important headers
        assert result["Content-Type"] == "application/json"
        assert result["Cache-Control"] == "no-cache"
        assert result["X-RateLimit-Remaining"] == "100"
        assert result["X-Request-ID"] == "req_123"

    def test_is_openai_request_path_detection(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test OpenAI request detection in response transformer."""
        # Test OpenAI paths
        assert (
            response_transformer._is_openai_request("/openai/v1/chat/completions")
            is True
        )
        assert response_transformer._is_openai_request("/v1/chat/completions") is True

        # Test Anthropic paths
        assert response_transformer._is_openai_request("/v1/messages") is False
        assert response_transformer._is_openai_request("/v1/models") is False

    async def test_transform_response_full_integration(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test full response transformation integration."""
        # Create a proxy response
        response_body = json.dumps(
            {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
            }
        )

        response = ProxyResponse(
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Content-Length": "50",
                "Server": "anthropic",
                "Transfer-Encoding": "chunked",
            },
            body=response_body.encode("utf-8"),
            metadata={"request_id": "req_456"},
        )

        # Create context with original path in metadata
        context = TransformContext()
        context.set("original_path", "/v1/messages")

        # Transform the response
        result = await response_transformer._transform_response(response, context)

        # Check status code preserved
        assert result.status_code == 200

        # Check headers transformation
        assert result.headers["Content-Type"] == "application/json"
        assert "Transfer-Encoding" not in result.headers
        # Content-Length should be recalculated based on actual body length
        assert "Content-Length" in result.headers

        # Check body preserved
        if isinstance(result.body, bytes):
            body_data = json.loads(result.body.decode("utf-8"))
            assert body_data["id"] == "msg_123"
            assert body_data["type"] == "message"

        # Check metadata preserved
        assert result.metadata == {"request_id": "req_456"}

    async def test_transform_response_with_string_body(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test response transformation with string body."""
        response = ProxyResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body='{"message": "test"}',
            metadata={},
        )

        context = TransformContext()
        context.set("original_path", "/v1/messages")
        result = await response_transformer._transform_response(response, context)

        # Should handle string body correctly
        assert result.body is not None
        if isinstance(result.body, bytes):
            assert json.loads(result.body.decode("utf-8"))["message"] == "test"

    async def test_transform_response_with_dict_body(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test response transformation with dict body."""
        response = ProxyResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"message": "test", "id": "123"},
            metadata={},
        )

        context = TransformContext()
        context.set("original_path", "/v1/messages")
        result = await response_transformer._transform_response(response, context)

        # Should handle dict body correctly
        assert result.body is not None
        if isinstance(result.body, bytes):
            body_data = json.loads(result.body.decode("utf-8"))
            assert body_data["message"] == "test"
            assert body_data["id"] == "123"

    async def test_transform_response_context_variations(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test response transformation with different context types."""
        response = ProxyResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body=b'{"test": true}',
            metadata={},
        )

        # Test with dict context
        dict_context: dict[str, str] = {"original_path": "/v1/messages"}
        result1 = await response_transformer._transform_response(
            response, cast(TransformContext, dict_context)
        )
        assert result1.status_code == 200

        # Test with no context
        result2 = await response_transformer._transform_response(response, None)
        assert result2.status_code == 200

        # Test with empty context
        result3 = await response_transformer._transform_response(
            response, TransformContext()
        )
        assert result3.status_code == 200


class TestClaudeCodePrompt:
    """Test Claude Code prompt utility function."""

    def test_get_fallback_system_field_structure(self) -> None:
        """Test fallback system field structure and content."""
        prompt_list = get_fallback_system_field()

        assert isinstance(prompt_list, list)
        assert len(prompt_list) == 1

        prompt = prompt_list[0]
        assert isinstance(prompt, dict)
        assert prompt["type"] == "text"
        assert (
            prompt["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert prompt["cache_control"] == {"type": "ephemeral"}

    def test_get_fallback_system_field_consistency(self) -> None:
        """Test that get_fallback_system_field returns consistent results."""
        prompt1 = get_fallback_system_field()
        prompt2 = get_fallback_system_field()

        assert prompt1 == prompt2
        assert prompt1 is not prompt2  # Should be different instances

    def test_get_detected_system_field_with_app_state_minimal(self) -> None:
        """Test detected system field with app state in minimal mode."""
        from ccproxy.models.detection import SystemPromptData

        # Mock app state with detected system field (list format)
        mock_app_state = type("MockAppState", (), {})()
        mock_claude_data = type("MockClaudeData", (), {})()
        mock_claude_data.system_prompt = SystemPromptData(
            system_field=[
                {
                    "type": "text",
                    "text": "Custom Claude Code prompt",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Additional context"},
            ]
        )
        mock_app_state.claude_detection_data = mock_claude_data

        result = get_detected_system_field(mock_app_state, "minimal")

        assert isinstance(result, list)
        assert len(result) == 1  # Minimal mode returns only first message
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Custom Claude Code prompt"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_get_detected_system_field_with_app_state_full(self) -> None:
        """Test detected system field with app state in full mode."""
        from ccproxy.models.detection import SystemPromptData

        # Mock app state with multiple detected system prompts
        mock_app_state = type("MockAppState", (), {})()
        mock_claude_data = type("MockClaudeData", (), {})()
        mock_claude_data.system_prompt = SystemPromptData(
            system_field=[
                {
                    "type": "text",
                    "text": "You are Claude Code",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Additional context from CLI."},
                {"type": "text", "text": "More system instructions."},
            ]
        )
        mock_app_state.claude_detection_data = mock_claude_data

        result = get_detected_system_field(mock_app_state, "full")

        assert isinstance(result, list)
        assert len(result) == 3  # Full mode returns all messages
        assert all(isinstance(prompt, dict) for prompt in result)
        assert all(prompt["type"] == "text" for prompt in result)
        assert result[0]["text"] == "You are Claude Code"
        assert result[1]["text"] == "Additional context from CLI."
        assert result[2]["text"] == "More system instructions."
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_get_detected_system_field_no_app_state(self) -> None:
        """Test getting detected system field without app state."""
        result = get_detected_system_field(None, "minimal")
        assert result is None

        result = get_detected_system_field(None, "full")
        assert result is None

    def test_get_detected_system_field_string_format(self) -> None:
        """Test detected system field with string format in minimal mode."""
        from ccproxy.models.detection import SystemPromptData

        # Mock app state with string system field
        mock_app_state = type("MockAppState", (), {})()
        mock_claude_data = type("MockClaudeData", (), {})()
        mock_claude_data.system_prompt = SystemPromptData(
            system_field="You are Claude Code, string format."
        )
        mock_app_state.claude_detection_data = mock_claude_data

        # Test both minimal and full modes with string
        result_minimal = get_detected_system_field(mock_app_state, "minimal")
        assert result_minimal == "You are Claude Code, string format."

        result_full = get_detected_system_field(mock_app_state, "full")
        assert result_full == "You are Claude Code, string format."


@pytest.mark.unit
class TestHTTPTransformersEdgeCases:
    """Test edge cases and error conditions for HTTP transformers."""

    @pytest.fixture
    def request_transformer(self) -> HTTPRequestTransformer:
        """Create HTTP request transformer instance for testing."""
        return HTTPRequestTransformer()

    @pytest.fixture
    def response_transformer(self) -> HTTPResponseTransformer:
        """Create HTTP response transformer instance for testing."""
        return HTTPResponseTransformer()

    def test_request_transformer_with_metrics_collector(self) -> None:
        """Test request transformer initialization with metrics collector."""
        from unittest.mock import Mock

        mock_collector = Mock()

        transformer = HTTPRequestTransformer()
        transformer.metrics_collector = mock_collector
        assert transformer.metrics_collector == mock_collector

    def test_response_transformer_with_metrics_collector(self) -> None:
        """Test response transformer initialization with metrics collector."""
        from unittest.mock import Mock

        mock_collector = Mock()

        transformer = HTTPResponseTransformer()
        transformer.metrics_collector = mock_collector
        assert transformer.metrics_collector == mock_collector

    def test_transform_path_edge_cases(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test path transformation edge cases."""
        # Empty path
        assert request_transformer.transform_path("") == ""

        # Root path
        assert request_transformer.transform_path("/") == "/"

        # Complex nested paths
        assert (
            request_transformer.transform_path("/api/openai/v1/chat/completions")
            == "/v1/messages"
        )

        # Path with query parameters (should not affect transformation)
        assert (
            request_transformer.transform_path("/v1/chat/completions?stream=true")
            == "/v1/chat/completions?stream=true"
        )

    def test_create_proxy_headers_case_insensitive_exclusion(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test that header exclusion is case-insensitive."""
        original_headers = {
            "HOST": "localhost",  # uppercase
            "Authorization": "Bearer token",  # mixed case
            "x-api-key": "key",  # lowercase
            "Content-Type": "application/json",
        }
        access_token = "new-token"

        result = request_transformer.create_proxy_headers(
            original_headers, access_token
        )

        # All variations should be excluded
        assert "HOST" not in result
        assert "Authorization" in result  # Should be replaced, not excluded
        assert result["Authorization"] == "Bearer new-token"
        assert "x-api-key" not in result
        assert result["Content-Type"] == "application/json"

    def test_transform_system_prompt_unicode_handling(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test system prompt transformation with Unicode content."""
        body_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": "Vous êtes un assistant français. 你好世界! 🌍",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data, ensure_ascii=False).encode("utf-8")

        result = request_transformer.transform_system_prompt(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should handle Unicode correctly
        assert len(result_data["system"]) == 2
        assert (
            result_data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert "français" in result_data["system"][1]["text"]
        assert "你好世界" in result_data["system"][1]["text"]
        assert "🌍" in result_data["system"][1]["text"]

    async def test_transform_request_url_construction_edge_cases(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test URL construction with various edge cases."""
        # Request without query parameters
        request1 = ProxyRequest(
            method=ProxyMethod.POST,
            url="http://localhost:8000/v1/messages",
            headers={},
            params={},
            body=b"{}",
            protocol=ProxyProtocol.HTTPS,
            timeout=30,
            metadata={},
        )

        result1 = await request_transformer._transform_request(request1, None)
        assert "https://api.anthropic.com" in result1.url

        # Request with complex URL structure
        request2 = ProxyRequest(
            method=ProxyMethod.GET,
            url="http://localhost:8000/path/with/slashes?key=value&other=param",
            headers={},
            params={"key": "value", "other": "param"},
            body=None,
            protocol=ProxyProtocol.HTTPS,
            timeout=30,
            metadata={},
        )

        result2 = await request_transformer._transform_request(request2, None)
        assert "https://api.anthropic.com" in result2.url
        assert "key=value" in result2.url
        assert "other=param" in result2.url

    def test_response_content_length_calculation_edge_cases(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test content length calculation with various body types."""
        # Test with different body types
        headers = {"Content-Type": "application/json"}

        # Bytes body
        result1 = response_transformer.transform_response_headers(
            headers, "/v1/messages", 100
        )
        assert result1["Content-Length"] == "100"

        # Zero length
        result2 = response_transformer.transform_response_headers(
            headers, "/v1/messages", 0
        )
        assert result2["Content-Length"] == "0"

        # Large content length
        result3 = response_transformer.transform_response_headers(
            headers,
            "/v1/messages",
            1048576,  # 1MB
        )
        assert result3["Content-Length"] == "1048576"

    def test_response_headers_excludes_content_encoding(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test that content-encoding header is excluded to prevent compression issues."""
        original_headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "Content-Length": "100",
            "Server": "anthropic-api",
        }
        path = "/v1/messages"
        content_length = 150

        result = response_transformer.transform_response_headers(
            original_headers, path, content_length
        )

        # Should exclude content-encoding to prevent decompression issues
        assert "Content-Encoding" not in result
        assert "content-encoding" not in result

        # Should preserve other headers
        assert result["Content-Type"] == "application/json"
        assert result["Server"] == "anthropic-api"
        assert result["Content-Length"] == "150"

    def test_response_headers_excludes_compression_headers_case_insensitive(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test that compression headers are excluded case-insensitively."""
        original_headers = {
            "Content-Type": "application/json",
            "content-encoding": "gzip",  # lowercase
            "CONTENT-ENCODING": "deflate",  # uppercase
            "Content-Encoding": "br",  # mixed case
            "Transfer-Encoding": "chunked",
            "Server": "anthropic-api",
        }
        path = "/v1/messages"
        content_length = 200

        result = response_transformer.transform_response_headers(
            original_headers, path, content_length
        )

        # Should exclude all variations of content-encoding
        assert "content-encoding" not in result
        assert "CONTENT-ENCODING" not in result
        assert "Content-Encoding" not in result

        # Should also exclude transfer-encoding
        assert "Transfer-Encoding" not in result

        # Should preserve safe headers
        assert result["Content-Type"] == "application/json"
        assert result["Server"] == "anthropic-api"
        assert result["Content-Length"] == "200"


class TestCompressionRegressionPrevention:
    """Test suite specifically for preventing compression-related regressions.

    This test class contains tests that specifically prevent the compression
    issue where HTTPX auto-decompresses responses but content-encoding headers
    are still forwarded, causing clients to try to decompress already
    decompressed data.
    """

    @pytest.fixture
    def request_transformer(self) -> HTTPRequestTransformer:
        """Create HTTP request transformer instance for testing."""
        return HTTPRequestTransformer()

    @pytest.fixture
    def response_transformer(self) -> HTTPResponseTransformer:
        """Create HTTP response transformer instance for testing."""
        return HTTPResponseTransformer()

    def test_compression_regression_response_headers_stripped(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test that compression headers are stripped from responses to prevent decompression errors.

        This test prevents the specific regression where:
        1. HTTPX automatically decompresses compressed responses
        2. But content-encoding headers are still forwarded to clients
        3. Clients try to decompress already decompressed data
        4. Results in "Error -3 while decompressing data: incorrect header check"
        """
        # Simulate a compressed response from upstream API
        upstream_headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",  # This would cause issues if forwarded
            "Content-Length": "100",
            "Server": "anthropic-api",
            "Cache-Control": "no-cache",
        }

        # After HTTPX decompression, the content length would be different
        actual_content_length = 250  # Decompressed content is larger

        result = response_transformer.transform_response_headers(
            upstream_headers, "/v1/messages", actual_content_length
        )

        # CRITICAL: Content-Encoding must be stripped to prevent client decompression
        assert "Content-Encoding" not in result
        assert "content-encoding" not in result

        # Content-Length should be updated to reflect decompressed size
        assert result["Content-Length"] == "250"

        # Other headers should be preserved
        assert result["Content-Type"] == "application/json"
        assert result["Server"] == "anthropic-api"
        assert result["Cache-Control"] == "no-cache"

    def test_compression_regression_request_headers_stripped(
        self, request_transformer: HTTPRequestTransformer
    ) -> None:
        """Test that compression headers are stripped from requests to prevent issues.

        This test prevents issues where clients send compression-related headers
        that could cause problems in the proxy flow.
        """
        # Simulate a client request with compression headers
        client_headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip, deflate, br",  # Could cause upstream compression
            "Content-Encoding": "gzip",  # Client trying to send compressed data
            "User-Agent": "test-client",
        }
        access_token = "test-token"

        result = request_transformer.create_proxy_headers(client_headers, access_token)

        # CRITICAL: Compression headers must be stripped to prevent issues
        assert "Accept-Encoding" not in result
        assert "accept-encoding" not in result
        assert "Content-Encoding" not in result
        assert "content-encoding" not in result

        # Other headers should be preserved
        assert result["Content-Type"] == "application/json"
        assert result["Authorization"] == "Bearer test-token"

    def test_compression_regression_multiple_encodings(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test that multiple compression encodings are all stripped properly."""
        # Test with multiple compression formats
        upstream_headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip, br",  # Multiple encodings
            "Vary": "Accept-Encoding",
            "X-Content-Type-Options": "nosniff",
        }

        result = response_transformer.transform_response_headers(
            upstream_headers, "/v1/messages", 100
        )

        # All compression-related headers should be stripped
        assert "Content-Encoding" not in result
        assert "content-encoding" not in result

        # Non-compression headers should be preserved
        assert result["Vary"] == "Accept-Encoding"
        assert result["X-Content-Type-Options"] == "nosniff"

    async def test_compression_regression_full_response_flow(
        self, response_transformer: HTTPResponseTransformer
    ) -> None:
        """Test full response transformation flow prevents compression issues."""
        # Simulate a full response with compression headers
        response_body = json.dumps(
            {
                "id": "msg_123",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ).encode("utf-8")

        # Simulate what upstream API would send (with compression headers)
        upstream_response = ProxyResponse(
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",  # This would cause client issues
                "Content-Length": "50",  # Original compressed size
                "Server": "anthropic-api",
                "X-Request-ID": "req_123",
            },
            body=response_body,  # This is already decompressed by HTTPX
            metadata={"request_id": "req_123"},
        )

        context = TransformContext()
        context.set("original_path", "/v1/messages")

        # Transform the response
        result = await response_transformer._transform_response(
            upstream_response, context
        )

        # CRITICAL: Content-Encoding must be stripped
        assert "Content-Encoding" not in result.headers
        assert "content-encoding" not in result.headers

        # Content-Length should be recalculated for decompressed body
        assert "Content-Length" in result.headers
        assert result.headers["Content-Length"] == str(len(response_body))

        # Other headers should be preserved
        assert result.headers["Content-Type"] == "application/json"
        assert result.headers["Server"] == "anthropic-api"
        assert result.headers["X-Request-ID"] == "req_123"

        # Body should be intact
        assert result.body == response_body
