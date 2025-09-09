"""Unit tests for Copilot response transformer."""

import json
from typing import Any

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
        assert result["X-Copilot-Provider"] == "ccproxy"
        assert result["X-Provider-Plugin"] == "copilot"

    def test_transform_headers_removes_problematic_headers(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test that problematic headers are removed."""
        headers = {
            "Content-Type": "application/json",
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "Content-Length": "123",
        }

        result = transformer.transform_headers(headers)

        assert "Transfer-Encoding" not in result
        assert "Connection" not in result
        assert result["Content-Length"] == "123"  # Content-Length should be preserved

    def test_transform_body_dict(self, transformer: CopilotResponseTransformer) -> None:
        """Test body transformation with dict input."""
        body = {"message": "Hello world", "status": "success"}

        result = transformer.transform_body(body)

        expected = json.dumps(body).encode("utf-8")
        assert result == expected
        assert isinstance(result, bytes)

    def test_transform_body_string(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test body transformation with string input."""
        body = "Hello world"

        result = transformer.transform_body(body)

        assert result == b"Hello world"

    def test_transform_body_bytes(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test body transformation with bytes input."""
        body = b"Hello world"

        result = transformer.transform_body(body)

        assert result == b"Hello world"

    def test_transform_body_and_headers(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test combined body and header transformation with content-length update."""
        body = {"message": "Hello world", "count": 42}
        headers = {"Content-Type": "application/json"}

        result_headers, result_body = transformer.transform_body_and_headers(
            body, headers
        )

        # Check body is correct
        expected_body = json.dumps(body).encode("utf-8")
        assert result_body == expected_body

        # Check headers include correct content-length
        assert result_headers["Content-Length"] == str(len(result_body))
        assert result_headers["Content-Type"] == "application/json"
        assert result_headers["X-Copilot-Provider"] == "ccproxy"

    def test_transform_body_and_headers_preserves_existing_content_length(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test that content-length is updated even when already present."""
        body = {"short": "msg"}  # Small body
        headers = {
            "Content-Type": "application/json",
            "Content-Length": "999",  # Wrong length
        }

        result_headers, result_body = transformer.transform_body_and_headers(
            body, headers
        )

        # Content-Length should be updated to correct value
        actual_length = len(result_body)
        assert result_headers["Content-Length"] == str(actual_length)
        assert result_headers["Content-Length"] != "999"

    def test_transform_error_response_includes_content_length(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test that error responses include correct content-length."""
        error = ValueError("Test error message")

        headers, body = transformer.transform_error_response(error, status_code=400)

        # Check content-length is set correctly
        assert "Content-Length" in headers
        assert headers["Content-Length"] == str(len(body))

        # Verify body is valid JSON
        error_data = json.loads(body.decode("utf-8"))
        assert "error" in error_data
        assert error_data["error"]["message"] == "Test error message"

    def test_transform_error_response_different_status_codes(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test error response transformation with different status codes."""
        error = RuntimeError("Internal server error")

        headers_500, body_500 = transformer.transform_error_response(
            error, status_code=500
        )
        headers_400, body_400 = transformer.transform_error_response(
            error, status_code=400
        )

        # Both should have correct content-length
        assert headers_500["Content-Length"] == str(len(body_500))
        assert headers_400["Content-Length"] == str(len(body_400))

        # Content should be different due to different error types
        error_500 = json.loads(body_500.decode("utf-8"))
        error_400 = json.loads(body_400.decode("utf-8"))

        assert error_500["error"]["type"] == "internal_error"
        assert error_400["error"]["type"] == "client_error"

    def test_transform_streaming_headers(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test streaming header transformation."""
        result = transformer.transform_streaming_headers()

        assert result["Content-Type"] == "text/event-stream"
        assert result["Cache-Control"] == "no-cache"
        assert result["Connection"] == "keep-alive"
        assert result["X-Copilot-Provider"] == "ccproxy"

    def test_prepare_response_context(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test response context preparation."""
        body = {"data": "test"}
        headers = {"Content-Type": "application/json"}

        context = transformer.prepare_response_context(
            body=body,
            headers=headers,
            status_code=200,
            endpoint="/test",
        )

        assert context["status_code"] == 200
        assert context["endpoint"] == "/test"
        assert "body" in context
        assert "headers" in context

    def test_cors_headers_integration(self) -> None:
        """Test CORS headers are added when configured."""
        cors_settings = {
            "allow_origin": "*",
            "allow_methods": "GET,POST,PUT,DELETE",
            "allow_headers": "Content-Type,Authorization",
        }

        transformer = CopilotResponseTransformer(cors_settings=cors_settings)
        headers = {"Content-Type": "application/json"}

        result = transformer.transform_headers(headers)

        assert result["Access-Control-Allow-Origin"] == "*"
        assert result["Access-Control-Allow-Methods"] == "GET,POST,PUT,DELETE"
        assert result["Access-Control-Allow-Headers"] == "Content-Type,Authorization"

    def test_content_length_edge_cases(
        self, transformer: CopilotResponseTransformer
    ) -> None:
        """Test content-length handling with edge cases."""
        # Empty body
        headers, body = transformer.transform_body_and_headers({}, {})
        assert headers["Content-Length"] == str(len(body))
        assert headers["Content-Length"] == "2"  # Empty dict: "{}"

        # Large body
        large_body = {"data": "x" * 1000}
        headers, body = transformer.transform_body_and_headers(large_body, {})
        expected_length = len(json.dumps(large_body).encode("utf-8"))
        assert headers["Content-Length"] == str(expected_length)
        assert len(body) == expected_length
