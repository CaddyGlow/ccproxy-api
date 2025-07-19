"""Example usage of composable auth fixtures.

This file demonstrates how to use the new auth fixture hierarchy
for different authentication testing scenarios.
"""

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient


class TestAuthModeExamples:
    """Examples of testing different auth modes."""

    def test_no_auth_endpoint(self, client_no_auth: TestClient) -> None:
        """Test endpoint that requires no authentication."""
        response = client_no_auth.get("/api/models")
        assert response.status_code == 200

    def test_bearer_auth_endpoint(
        self,
        client_bearer_auth: TestClient,
        auth_mode_bearer_token: dict[str, Any],
        auth_headers_factory: Callable[..., Any],
    ) -> None:
        """Test endpoint with bearer token authentication."""
        headers = auth_headers_factory(auth_mode_bearer_token)
        response = client_bearer_auth.get("/api/models", headers=headers)
        assert response.status_code == 200

    def test_configured_auth_endpoint(
        self,
        client_configured_auth: TestClient,
        auth_mode_configured_token: dict[str, Any],
        auth_headers_factory: Callable[..., Any],
    ) -> None:
        """Test endpoint with server-configured auth token."""
        headers = auth_headers_factory(auth_mode_configured_token)
        response = client_configured_auth.get("/api/models", headers=headers)
        assert response.status_code == 200

    def test_credentials_auth_endpoint(
        self,
        client_credentials_auth: TestClient,
    ) -> None:
        """Test endpoint with credentials-based authentication."""
        # Credentials auth doesn't require headers - handled by auth manager
        response = client_credentials_auth.get("/api/models")
        assert response.status_code == 200


class TestAuthNegativeScenarios:
    """Examples of testing authentication failures."""

    def test_invalid_bearer_token(
        self,
        client_bearer_auth: TestClient,
        auth_mode_bearer_token: dict[str, Any],
        invalid_auth_headers_factory: Callable[..., Any],
        auth_test_utils: dict[str, Any],
    ) -> None:
        """Test with invalid bearer token."""
        headers = invalid_auth_headers_factory(auth_mode_bearer_token)
        response = client_bearer_auth.get("/api/models", headers=headers)

        assert auth_test_utils["is_auth_error"](response)
        error_detail = auth_test_utils["extract_auth_error_detail"](response)
        assert error_detail is not None

    def test_invalid_configured_token(
        self,
        client_configured_auth: TestClient,
        auth_mode_configured_token: dict[str, Any],
        invalid_auth_headers_factory: Callable[..., Any],
        auth_test_utils: dict[str, Any],
    ) -> None:
        """Test with invalid configured token."""
        headers = invalid_auth_headers_factory(auth_mode_configured_token)
        response = client_configured_auth.get("/api/models", headers=headers)

        assert auth_test_utils["is_auth_error"](response)
        assert response.status_code == 401

    def test_missing_auth_header(
        self,
        client_bearer_auth: TestClient,
        auth_test_utils: dict[str, Any],
    ) -> None:
        """Test with missing authentication header."""
        response = client_bearer_auth.get("/api/models")  # No headers

        assert auth_test_utils["is_auth_error"](response)
        assert response.status_code == 401


class TestAuthFactoryPatterns:
    """Examples of using auth factories for custom scenarios."""

    def test_custom_auth_configuration(
        self,
        app_factory: Callable[..., Any],
        client_factory: Callable[..., Any],
        auth_test_utils: dict[str, Any],
    ) -> None:
        """Test with custom authentication configuration."""
        # Define custom auth config
        custom_config = {
            "mode": "custom_bearer",
            "requires_token": True,
            "has_configured_token": False,
            "test_token": "custom-test-token-12345",
        }

        # Create app and client with custom config
        app = app_factory(custom_config)
        client = client_factory(app)

        # Test with custom auth
        headers = {"Authorization": f"Bearer {custom_config['test_token']}"}
        response = client.get("/api/models", headers=headers)

        assert auth_test_utils["is_auth_success"](response)

    def test_multiple_token_scenarios(
        self,
        app_factory: Callable[..., Any],
        client_factory: Callable[..., Any],
    ) -> None:
        """Test multiple token scenarios with same app."""
        config = {
            "mode": "bearer_token",
            "requires_token": True,
            "has_configured_token": False,
            "test_token": "multi-test-token-123",
        }

        app = app_factory(config)
        client = client_factory(app)

        # Test valid token
        valid_headers = {"Authorization": f"Bearer {config['test_token']}"}
        response = client.get("/api/models", headers=valid_headers)
        assert response.status_code == 200

        # Test invalid token
        invalid_headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/api/models", headers=invalid_headers)
        assert response.status_code == 401


class TestAuthParametrizedPatterns:
    """Examples of parametrized testing across auth modes."""

    @pytest.mark.parametrize(
        "auth_setup",
        [
            ("no_auth", "client_no_auth", None),
            ("bearer", "client_bearer_auth", "auth_mode_bearer_token"),
            ("configured", "client_configured_auth", "auth_mode_configured_token"),
        ],
    )
    def test_models_endpoint_all_auth_modes(
        self,
        request: pytest.FixtureRequest,
        auth_setup: tuple[str, str, str | None],
        auth_headers_factory: Callable[..., Any],
    ) -> None:
        """Test /v1/models endpoint across all auth modes."""
        mode_name, client_fixture, config_fixture = auth_setup
        client = request.getfixturevalue(client_fixture)

        if config_fixture:
            config = request.getfixturevalue(config_fixture)
            headers = auth_headers_factory(config)
        else:
            headers = {}

        response = client.get("/api/models", headers=headers)
        assert response.status_code == 200

        # Verify response structure
        data = response.json()
        assert "object" in data
        assert data["object"] == "list"

    @pytest.mark.parametrize(
        "auth_mode,expected_status",
        [
            ("bearer", 401),  # Invalid token should fail
            ("configured", 401),  # Invalid token should fail
        ],
    )
    def test_invalid_tokens_parametrized(
        self,
        request: pytest.FixtureRequest,
        auth_mode: str,
        expected_status: int,
        invalid_auth_headers_factory: Callable[..., Any],
    ) -> None:
        """Test invalid tokens across bearer and configured modes."""
        client_fixture = f"client_{auth_mode}_auth"
        config_fixture = f"auth_mode_{auth_mode}_token"

        client = request.getfixturevalue(client_fixture)
        config = request.getfixturevalue(config_fixture)
        headers = invalid_auth_headers_factory(config)

        response = client.get("/api/models", headers=headers)
        assert response.status_code == expected_status


class TestOAuthFlowSimulation:
    """Examples of OAuth flow testing."""

    def test_successful_oauth_flow(
        self,
        oauth_flow_simulator: dict[str, Any],
        mock_oauth: object,  # HTTPXMock fixture
    ) -> None:
        """Test successful OAuth flow simulation."""
        oauth_data = oauth_flow_simulator["successful_oauth"]()

        assert oauth_data["access_token"] == "oauth-access-token-12345"
        assert oauth_data["refresh_token"] == "oauth-refresh-token-67890"
        assert oauth_data["token_type"] == "Bearer"
        assert oauth_data["expires_in"] == 3600

    def test_oauth_error_flow(
        self,
        oauth_flow_simulator: dict[str, Any],
    ) -> None:
        """Test OAuth error flow simulation."""
        error_data = oauth_flow_simulator["oauth_error"]()

        assert error_data["error"] == "invalid_grant"
        assert "authorization grant is invalid" in error_data["error_description"]

    def test_token_refresh_flow(
        self,
        oauth_flow_simulator: dict[str, Any],
    ) -> None:
        """Test token refresh flow simulation."""
        refresh_data = oauth_flow_simulator["token_refresh"]()

        assert "refreshed-access-token" in refresh_data["access_token"]
        assert "new-refresh-token" in refresh_data["refresh_token"]
        assert refresh_data["token_type"] == "Bearer"


class TestAuthUtilities:
    """Examples of using auth test utilities."""

    def test_auth_response_detection(
        self,
        client_bearer_auth: TestClient,
        auth_test_utils: dict[str, Any],
    ) -> None:
        """Test auth response detection utilities."""
        # Test auth error detection
        response = client_bearer_auth.get("/api/models")  # No auth header
        assert auth_test_utils["is_auth_error"](response)
        assert not auth_test_utils["is_auth_success"](response)

        # Test error detail extraction
        error_detail = auth_test_utils["extract_auth_error_detail"](response)
        assert error_detail is not None
        assert isinstance(error_detail, str)

    def test_auth_success_detection(
        self,
        client_bearer_auth: TestClient,
        auth_mode_bearer_token: dict[str, Any],
        auth_headers_factory: Callable[..., Any],
        auth_test_utils: dict[str, Any],
    ) -> None:
        """Test auth success detection utilities."""
        headers = auth_headers_factory(auth_mode_bearer_token)
        response = client_bearer_auth.get("/api/models", headers=headers)

        assert auth_test_utils["is_auth_success"](response)
        assert not auth_test_utils["is_auth_error"](response)

        # Error detail should be None for successful auth
        error_detail = auth_test_utils["extract_auth_error_detail"](response)
        assert error_detail is None
