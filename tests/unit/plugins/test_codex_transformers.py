"""Unit tests for Codex transformer classes."""

import json
from unittest.mock import MagicMock

import pytest

from ccproxy.config.cors import CORSSettings
from plugins.codex.transformers.request import CodexRequestTransformer
from plugins.codex.transformers.response import CodexResponseTransformer


class TestCodexRequestTransformer:
    """Test cases for CodexRequestTransformer."""

    def test_init_without_detection_service(self):
        """Test initialization without detection service."""
        transformer = CodexRequestTransformer()
        assert transformer.detection_service is None

    def test_init_with_detection_service(self):
        """Test initialization with detection service."""
        mock_service = MagicMock()
        transformer = CodexRequestTransformer(mock_service)
        assert transformer.detection_service is mock_service

    def test_transform_headers_basic(self):
        """Test basic header transformation."""
        transformer = CodexRequestTransformer()
        headers = {
            "content-type": "application/json",
            "host": "localhost:8000",  # Should be removed
            "custom-header": "value",
        }

        result = transformer.transform_headers(headers, "test-session", "test-token")

        assert "host" not in result
        assert result["session_id"] == "test-session"
        assert result["Authorization"] == "Bearer test-token"
        assert result["custom-header"] == "value"
        assert result["originator"] == "codex_cli_rs"

    def test_transform_headers_with_detection_service(self):
        """Test header transformation with detection service."""
        mock_service = MagicMock()
        mock_cached_data = MagicMock()
        mock_cached_data.headers.to_headers_dict.return_value = {
            "version": "0.22.0",
            "custom-codex": "detected",
        }
        mock_cached_data.codex_version = "0.22.0"
        mock_service.get_cached_data.return_value = mock_cached_data

        transformer = CodexRequestTransformer(mock_service)
        headers = {"content-type": "application/json"}

        result = transformer.transform_headers(headers, "test-session", "test-token")

        assert result["version"] == "0.22.0"
        assert result["custom-codex"] == "detected"
        assert result["session_id"] == "test-session"  # Should not be overridden

    def test_transform_body_with_instructions(self):
        """Test body transformation when instructions already exist."""
        transformer = CodexRequestTransformer()
        body_data = {"instructions": "existing instructions", "model": "gpt-5"}
        body_bytes = json.dumps(body_data).encode()

        result = transformer.transform_body(body_bytes)
        assert result is not None
        result_data = json.loads(result.decode())

        assert result_data["instructions"] == "existing instructions"

    def test_transform_body_without_instructions(self):
        """Test body transformation when instructions are missing."""
        transformer = CodexRequestTransformer()
        body_data = {"model": "gpt-5"}
        body_bytes = json.dumps(body_data).encode()

        result = transformer.transform_body(body_bytes)
        assert result is not None
        result_data = json.loads(result.decode())

        assert "instructions" in result_data
        assert "Codex CLI" in result_data["instructions"]

    def test_transform_body_invalid_json(self):
        """Test body transformation with invalid JSON."""
        transformer = CodexRequestTransformer()
        invalid_body = b"invalid json"

        result = transformer.transform_body(invalid_body)

        assert result == invalid_body  # Should return original

    def test_transform_body_empty(self):
        """Test body transformation with empty body."""
        transformer = CodexRequestTransformer()

        result = transformer.transform_body(None)
        assert result is None

        result = transformer.transform_body(b"")
        assert result == b""

    def test_get_instructions_with_detection_service(self):
        """Test getting instructions from detection service."""
        mock_service = MagicMock()
        mock_cached_data = MagicMock()
        mock_instructions = MagicMock()
        mock_instructions.instructions_field = "detected instructions"
        mock_cached_data.instructions = mock_instructions
        mock_service.get_cached_data.return_value = mock_cached_data

        transformer = CodexRequestTransformer(mock_service)
        instructions = transformer._get_instructions()

        assert instructions == "detected instructions"

    def test_get_instructions_fallback(self):
        """Test fallback instructions when detection service fails."""
        transformer = CodexRequestTransformer()
        instructions = transformer._get_instructions()

        assert instructions is not None
        assert "Codex CLI" in instructions
        assert "OpenAI" in instructions


class TestCodexResponseTransformer:
    """Test cases for CodexResponseTransformer."""

    def test_init(self):
        """Test initialization."""
        transformer = CodexResponseTransformer()
        # Just ensure it initializes without error
        assert transformer is not None

    def test_transform_headers_basic_no_cors_settings(self):
        """Test basic header transformation without CORS settings (secure fallback)."""
        transformer = CodexResponseTransformer()
        headers = {
            "content-type": "application/json",
            "content-length": "123",  # Should be removed
            "custom-header": "value",
        }

        result = transformer.transform_headers(headers)

        assert "content-length" not in result
        assert result["content-type"] == "application/json"
        assert result["custom-header"] == "value"
        # Without CORS settings and no Origin header, no CORS headers should be added
        assert "Access-Control-Allow-Origin" not in result

    def test_transform_headers_localhost_origin_fallback(self):
        """Test header transformation with localhost origin uses secure fallback."""
        transformer = CodexResponseTransformer()
        headers = {"content-type": "application/json"}
        request_headers = {"origin": "http://localhost:3000"}

        result = transformer.transform_headers(headers, request_headers=request_headers)

        assert result["content-type"] == "application/json"
        # Secure fallback allows localhost origins
        assert result["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert (
            "Content-Type, Authorization, Accept, Origin, X-Requested-With"
            in result["Access-Control-Allow-Headers"]
        )
        assert (
            "GET, POST, PUT, DELETE, OPTIONS" in result["Access-Control-Allow-Methods"]
        )

    def test_transform_headers_non_localhost_origin_fallback(self):
        """Test header transformation with non-localhost origin uses secure fallback (blocks)."""
        transformer = CodexResponseTransformer()
        headers = {"content-type": "application/json"}
        request_headers = {"origin": "https://evil.example.com"}

        result = transformer.transform_headers(headers, request_headers=request_headers)

        assert result["content-type"] == "application/json"
        # Secure fallback blocks non-localhost origins
        assert "Access-Control-Allow-Origin" not in result

    def test_transform_headers_with_cors_settings(self):
        """Test header transformation with proper CORS settings."""
        cors_settings = CORSSettings(
            origins=["https://app.example.com", "https://test.example.com"],
            methods=["GET", "POST"],
            headers=["Content-Type", "Authorization"],
        )
        transformer = CodexResponseTransformer(cors_settings)
        headers = {"content-type": "application/json"}
        request_headers = {"origin": "https://app.example.com"}

        result = transformer.transform_headers(headers, request_headers=request_headers)

        assert result["content-type"] == "application/json"
        # CORS settings properly applied
        assert result["Access-Control-Allow-Origin"] == "https://app.example.com"
        assert result["Access-Control-Allow-Methods"] == "GET, POST"
        assert result["Access-Control-Allow-Headers"] == "Content-Type, Authorization"

    def test_transform_headers_with_cors_settings_blocked_origin(self):
        """Test header transformation with CORS settings blocks unauthorized origin."""
        cors_settings = CORSSettings(
            origins=["https://app.example.com"], methods=["GET", "POST"]
        )
        transformer = CodexResponseTransformer(cors_settings)
        headers = {"content-type": "application/json"}
        request_headers = {"origin": "https://evil.example.com"}

        result = transformer.transform_headers(headers, request_headers=request_headers)

        assert result["content-type"] == "application/json"
        # Origin not in allowed list, so no CORS headers
        assert "Access-Control-Allow-Origin" not in result

    def test_transform_headers_excludes_problematic(self):
        """Test that problematic headers are excluded."""
        transformer = CodexResponseTransformer()
        headers = {
            "content-length": "123",
            "transfer-encoding": "chunked",
            "content-encoding": "gzip",
            "connection": "keep-alive",
            "keep-header": "value",
        }

        result = transformer.transform_headers(headers)

        # All problematic headers should be excluded
        assert "content-length" not in result
        assert "transfer-encoding" not in result
        assert "content-encoding" not in result
        assert "connection" not in result

        # Non-problematic header should remain
        assert result["keep-header"] == "value"

        # CORS headers should be added
        assert "Access-Control-Allow-Origin" in result

    def test_transform_body_passthrough(self):
        """Test that body transformation is passthrough."""
        transformer = CodexResponseTransformer()

        # Test with bytes
        body_bytes = b"test body"
        result = transformer.transform_body(body_bytes)
        assert result == body_bytes

        # Test with None
        result = transformer.transform_body(None)
        assert result is None


@pytest.mark.asyncio
async def test_transformers_integration():
    """Test request and response transformers work together."""
    request_transformer = CodexRequestTransformer()
    response_transformer = CodexResponseTransformer()

    # Test request transformation
    request_headers = {"content-type": "application/json"}
    request_body = json.dumps({"model": "gpt-5"}).encode()

    transformed_headers = request_transformer.transform_headers(
        request_headers, "test-session", "test-token"
    )
    transformed_body = request_transformer.transform_body(request_body)

    # Verify request transformation
    assert transformed_headers["session_id"] == "test-session"
    assert transformed_body is not None
    assert "instructions" in json.loads(transformed_body.decode())

    # Test response transformation
    response_headers = {"content-type": "application/json", "content-length": "100"}
    response_body = b'{"output": "test response"}'

    transformed_response_headers = response_transformer.transform_headers(
        response_headers
    )
    transformed_response_body = response_transformer.transform_body(response_body)

    # Verify response transformation
    assert "content-length" not in transformed_response_headers
    assert "Access-Control-Allow-Origin" in transformed_response_headers
    assert transformed_response_body == response_body  # Passthrough
