"""Authentication fixtures for ccproxy tests.

This module provides composable authentication fixtures that support all auth modes
without requiring test skips. The fixtures are organized into:

- Auth Mode Configurations: Define different authentication scenarios
- Settings Factories: Create appropriate Settings for each auth mode
- App and Client Factories: Create FastAPI apps and test clients
- Utilities: Helper functions for auth testing
- OAuth Simulators: Mock OAuth flows for testing

Usage:
    # Use specific auth mode fixtures
    def test_with_bearer_auth(client_bearer_auth, auth_mode_bearer_token, auth_headers_factory):
        headers = auth_headers_factory(auth_mode_bearer_token)
        response = client_bearer_auth.get("/v1/models", headers=headers)
        assert response.status_code == 200

    # Use factories for custom configurations
    def test_custom_auth(app_factory, auth_test_utils):
        custom_config = {"mode": "custom", "requires_token": True}
        app = app_factory(custom_config)
        # ... test logic
"""

__all__ = [
    # Auth mode configurations
    "auth_mode_none",
    "auth_mode_bearer_token",
    "auth_mode_configured_token",
    "auth_mode_credentials",
    "auth_mode_credentials_with_fallback",
    # Factories
    "auth_settings_factory",
    "auth_headers_factory",
    "invalid_auth_headers_factory",
    "app_factory",
    "client_factory",
    # Convenience fixtures
    "app_no_auth",
    "app_bearer_auth",
    "app_configured_auth",
    "app_credentials_auth",
    "client_no_auth",
    "client_bearer_auth",
    "client_configured_auth",
    "client_credentials_auth",
    # Utilities
    "auth_test_utils",
    "oauth_flow_simulator",
]
