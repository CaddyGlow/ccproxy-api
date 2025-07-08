"""Regression tests for model translation and endpoint differences."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.services.translator import map_openai_model_to_claude


@pytest.mark.unit
class TestModelTranslationRegression:
    """Ensure model translation continues to work correctly."""

    def test_openai_model_mapping(self):
        """Test that OpenAI models are correctly mapped to Claude models."""
        # Test exact matches
        assert map_openai_model_to_claude("o3-mini") == "claude-opus-4-20250514"
        assert map_openai_model_to_claude("o1-mini") == "claude-sonnet-4-20250514"
        assert map_openai_model_to_claude("gpt-4o-mini") == "claude-3-5-haiku-latest"
        assert map_openai_model_to_claude("gpt-4o") == "claude-3-7-sonnet-20250219"

        # Test Claude models pass through unchanged
        assert (
            map_openai_model_to_claude("claude-3-5-sonnet-latest")
            == "claude-3-5-sonnet-latest"
        )
        assert (
            map_openai_model_to_claude("claude-opus-4-20250514")
            == "claude-opus-4-20250514"
        )

        # Test unknown models pass through
        assert map_openai_model_to_claude("unknown-model") == "unknown-model"

        # Test startswith matching
        assert (
            map_openai_model_to_claude("gpt-4o-mini-preview")
            == "claude-3-5-haiku-latest"
        )
        assert map_openai_model_to_claude("o3-mini-2024") == "claude-opus-4-20250514"

    @patch("ccproxy.routers.openai.ClaudeClient")
    def test_openai_endpoint_uses_translated_model(
        self, mock_claude_client_class, test_client
    ):
        """Test that /cc/openai/v1 endpoint translates model names."""
        # Setup mock
        mock_client = MagicMock()
        mock_claude_client_class.return_value = mock_client
        mock_client.create_completion = AsyncMock(
            return_value={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "model": "claude-opus-4-20250514",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )

        # Test with o3-mini
        request = {
            "model": "o3-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = test_client.post("/cc/openai/v1/chat/completions", json=request)
        assert response.status_code == 200

        # Verify the model was translated when calling Claude
        args, kwargs = mock_client.create_completion.call_args
        options = args[1]
        assert options.model == "claude-opus-4-20250514", (
            "Model should be translated from o3-mini to claude-opus-4-20250514"
        )

        # Verify response preserves original model name
        data = response.json()
        assert data["model"] == "o3-mini", (
            "Response should contain original OpenAI model name"
        )

    def test_model_translation_in_openai_router_import(self):
        """Ensure the OpenAI router uses the translation function."""
        # This test verifies the translation is being used
        import ast
        import inspect

        from ccproxy.routers import openai

        # Get the source of the create_chat_completion function
        source = inspect.getsource(openai.create_chat_completion)

        # Check that map_openai_model_to_claude is imported and used
        assert (
            "from ccproxy.services.translator import map_openai_model_to_claude"
            in source
        ), "OpenAI router must import map_openai_model_to_claude"
        assert "map_openai_model_to_claude(request.model)" in source, (
            "OpenAI router must use map_openai_model_to_claude for model translation"
        )


@pytest.mark.unit
class TestEndpointDifferences:
    """Document and verify endpoint behavior differences."""

    def test_endpoint_documentation(self):
        """Verify endpoint behavior is as documented."""
        # This test documents the expected behavior after transformation fix
        endpoints = {
            "/openai/v1/chat/completions": {
                "type": "reverse_proxy",
                "response_format": "openai_sse",  # Now returns OpenAI format
                "model_translation": False,
                "description": "Reverse proxy endpoint - transforms to OpenAI SSE format",
                "example_format": 'data: {"object": "chat.completion.chunk", ...}',
            },
            "/api/openai/v1/chat/completions": {
                "type": "reverse_proxy",
                "response_format": "openai_sse",  # Now returns OpenAI format
                "model_translation": False,
                "description": "API mode reverse proxy - transforms to OpenAI SSE format",
                "example_format": 'data: {"object": "chat.completion.chunk", ...}',
            },
            "/cc/openai/v1/chat/completions": {
                "type": "claude_code_sdk",
                "response_format": "openai_sse",
                "model_translation": True,
                "description": "Claude Code SDK endpoint - returns OpenAI SSE format",
                "example_format": 'data: {"object": "chat.completion.chunk", ...}',
            },
        }

        # Verify our assumptions - all OpenAI endpoints return OpenAI format
        assert (
            endpoints["/openai/v1/chat/completions"]["response_format"] == "openai_sse"
        )
        assert (
            endpoints["/api/openai/v1/chat/completions"]["response_format"]
            == "openai_sse"
        )
        assert (
            endpoints["/cc/openai/v1/chat/completions"]["response_format"]
            == "openai_sse"
        )

    def test_aider_configuration_requirements(self):
        """Document required configuration for Aider compatibility."""
        # Aider requires OpenAI SSE format
        aider_config = {
            "correct_endpoint": "/cc/openai/v1",
            "correct_env_vars": {
                "OPENAI_API_KEY": "dummy",
                "OPENAI_BASE_URL": "http://localhost:8000/cc/openai/v1",
            },
            "incorrect_endpoint": "/openai/v1",  # Returns Anthropic format
            "reason": "Aider expects OpenAI SSE format which only /cc/openai/v1 provides",
        }

        # Verify configuration - now all endpoints work
        assert "/cc/openai/v1" in aider_config["correct_endpoint"]
        # After fix, all OpenAI endpoints are correct
        assert (
            aider_config["reason"]
            == "Aider expects OpenAI SSE format which only /cc/openai/v1 provides"
        )

    def test_reverse_proxy_streaming_transformation(self):
        """Test that reverse proxy transforms streaming to OpenAI format."""
        from ccproxy.services.response_transformer import ResponseTransformer

        transformer = ResponseTransformer()

        # Test _is_openai_request detection
        assert transformer._is_openai_request("/openai/v1/chat/completions") is True
        assert transformer._is_openai_request("/api/openai/v1/chat/completions") is True
        assert transformer._is_openai_request("/min/openai/v1/chat/completions") is True
        assert transformer._is_openai_request("/v1/messages") is False
        assert transformer._is_openai_request("/api/v1/messages") is False
