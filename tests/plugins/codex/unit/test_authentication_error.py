"""Test authentication error handling."""

from ccproxy.plugins.codex.transformers.request import CodexRequestTransformer


class TestAuthenticationError:
    """Test that authentication errors are properly raised."""

    def test_request_transformer_no_token_does_not_inject_auth(self) -> None:
        """Transformer should not inject Authorization when no token provided."""
        transformer = CodexRequestTransformer(detection_service=None)

        headers = {"Content-Type": "application/json"}

        # Should not inject Authorization when access_token is None
        result = transformer.transform_headers(
            headers, session_id="test", access_token=None
        )

        # Should still return headers but without Authorization
        assert "Authorization" not in result
        assert result["session_id"] == "test"

    def test_request_transformer_accepts_valid_token(self) -> None:
        """Test that transformer accepts valid token."""
        transformer = CodexRequestTransformer(detection_service=None)

        headers = {"Content-Type": "application/json"}

        # Should not raise when access_token is provided
        result = transformer.transform_headers(
            headers, session_id="test", access_token="valid_token_123"
        )

        assert "Authorization" in result
        assert result["Authorization"] == "Bearer valid_token_123"
        assert result["session_id"] == "test"
