"""Test authentication error handling."""

from plugins.codex.transformers.request import CodexRequestTransformer


class TestAuthenticationError:
    """Test that authentication errors are properly raised."""

    def test_request_transformer_warns_when_no_token(self):
        """Test that transformer warns but continues when no token provided."""
        transformer = CodexRequestTransformer(detection_service=None)

        headers = {"Content-Type": "application/json"}

        # Should NOT raise error, just warn and add empty Bearer token
        result = transformer.transform_headers(
            headers, session_id="test", access_token=None
        )

        # Should still return headers with empty Bearer token
        assert "Authorization" in result
        assert result["Authorization"] == "Bearer "
        assert result["session_id"] == "test"

    def test_request_transformer_accepts_valid_token(self):
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
