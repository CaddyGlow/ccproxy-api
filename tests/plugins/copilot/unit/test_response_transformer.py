"""Unit tests for Copilot response transformer."""

import json

import pytest

from ccproxy.plugins.copilot.transformers.response import CopilotResponseTransformer


class TestCopilotResponseTransformer:
    """Test cases for CopilotResponseTransformer."""

    @pytest.fixture
    def transformer(self) -> CopilotResponseTransformer:
        """Create response transformer instance."""
        return CopilotResponseTransformer()

    def test_transform_headers_basic(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test basic header transformation."""
        headers = {"Content-Type": "application/json"}

        result = transformer.transform_headers(headers)

        assert result["Content-Type"] == "application/json"
        # Provider headers are not added by default to avoid Content-Length issues

    def test_transform_headers_removes_problematic_headers(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test that problematic headers are removed."""
        headers = {
            "Content-Type": "application/json",
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "Content-Length": "123",
            "Content-Encoding": "gzip",
        }

        result = transformer.transform_headers(headers)

        # These headers should be removed to prevent Content-Length mismatches
        assert "Transfer-Encoding" not in result
        assert "Connection" not in result
        assert "Content-Length" not in result  # Will be recalculated by HTTP adapter
        assert "Content-Encoding" not in result
        assert result["Content-Type"] == "application/json"

    def test_transform_body_dict(self, transformer: CopilotResponseTransformer) -> None:
        """Test body transformation with dict input."""
        body = {"message": "Hello world", "status": "success"}

        result = transformer.transform_body(body)

        expected = json.dumps(body).encode("utf-8")
        assert result == expected

    def test_transform_body_string(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test body transformation with string input."""
        body = "Hello world"

        result = transformer.transform_body(body)

        expected = body.encode("utf-8")
        assert result == expected

    def test_transform_body_bytes(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test body transformation with bytes input."""
        body = b"Hello world"

        result = transformer.transform_body(body)

        assert result == body

    def test_separate_body_and_header_transformation(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test body and header transformation separately."""
        body = {"message": "Hello world", "count": 42}
        headers = {"Content-Type": "application/json", "Content-Length": "999"}

        # Test body transformation
        result_body = transformer.transform_body(body)
        expected_body = json.dumps(body).encode("utf-8")
        assert result_body == expected_body

        # Test header transformation - Content-Length should be excluded
        result_headers = transformer.transform_headers(headers)
        assert result_headers["Content-Type"] == "application/json"
        assert "Content-Length" not in result_headers  # Excluded to prevent mismatches

    def test_transform_error_response_basic(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test error response transformation."""
        error = ValueError("Test error")
        status_code = 400

        headers, body = transformer.transform_error_response(error, status_code)

        # Check headers
        assert headers["Content-Type"] == "application/json"
        assert "Content-Length" not in headers  # Should be excluded

        # Check body structure
        error_data = json.loads(body.decode("utf-8"))
        assert "error" in error_data
        assert error_data["error"]["message"] == "Test error"
        assert error_data["error"]["type"] == "client_error"

    def test_transform_error_response_different_status_codes(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test error response with different status codes."""
        error = RuntimeError("Server error")

        headers_500, body_500 = transformer.transform_error_response(error, 500)
        headers_400, body_400 = transformer.transform_error_response(error, 400)

        # Check both have proper content type and no Content-Length
        assert headers_500["Content-Type"] == "application/json"
        assert headers_400["Content-Type"] == "application/json"
        assert "Content-Length" not in headers_500
        assert "Content-Length" not in headers_400

        # Check error types differ based on status code
        error_500 = json.loads(body_500.decode("utf-8"))
        error_400 = json.loads(body_400.decode("utf-8"))

        assert error_500["error"]["type"] == "internal_error"
        assert error_400["error"]["type"] == "client_error"

    def test_transform_streaming_headers(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test streaming headers generation."""
        result = transformer.transform_streaming_headers()

        assert result["content-type"] == "text/event-stream"
        assert result["cache-control"] == "no-cache"
        assert result["connection"] == "keep-alive"

    def test_cors_headers_integration(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test CORS headers integration."""
        cors_settings = {
            "allow_origin": "https://example.com",
            "allow_methods": "GET, POST",
            "allow_headers": "Content-Type, Authorization",
        }
        transformer_with_cors = CopilotResponseTransformer(cors_settings=cors_settings)

        headers = {"Content-Type": "application/json"}
        result = transformer_with_cors.transform_headers(headers)

        assert result["access-control-allow-origin"] == "https://example.com"
        assert result["access-control-allow-methods"] == "GET, POST"
        assert result["access-control-allow-headers"] == "Content-Type, Authorization"

    def test_prepare_response_context(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test response context preparation."""
        headers = {"Content-Type": "application/json", "Content-Length": "100"}
        body = {"test": "data"}
        status_code = 200

        context = transformer.prepare_response_context(headers, body, status_code)

        assert context["status_code"] == 200
        assert context["provider"] == "copilot"
        assert "Content-Length" not in context["headers"]  # Should be excluded
        assert context["headers"]["Content-Type"] == "application/json"

        expected_body = json.dumps(body).encode("utf-8")
        assert context["body"] == expected_body

    def test_content_length_edge_cases(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test edge cases with Content-Length handling."""
        # Empty body
        empty_body = {}
        result_body = transformer.transform_body(empty_body)
        expected_body = json.dumps(empty_body).encode("utf-8")
        assert result_body == expected_body
        assert len(result_body) == 2  # "{}" = 2 bytes

        # Headers with various casing for Content-Length
        headers = {
            "content-length": "100",
            "Content-Length": "200",
            "CONTENT-LENGTH": "300",
            "Content-Type": "application/json",
        }
        result_headers = transformer.transform_headers(headers)

        # All variations should be removed
        assert "content-length" not in result_headers
        assert "Content-Length" not in result_headers
        assert "CONTENT-LENGTH" not in result_headers
        assert result_headers["Content-Type"] == "application/json"
