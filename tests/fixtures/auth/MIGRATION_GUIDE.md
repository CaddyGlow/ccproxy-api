# Auth Fixtures Migration Guide

## Overview

The new composable auth fixture hierarchy replaces the previous mixed auth fixtures with a clear, mode-based approach that eliminates test skips and provides better separation of concerns.

## Old vs New Auth Patterns

### Before (Old Fixtures)
```python
def test_auth_endpoint(client_with_auth, auth_headers):
    """Old pattern - mixed auth concerns, potential skips"""
    response = client_with_auth.get("/v1/models", headers=auth_headers)
    if response.status_code == 401:
        pytest.skip("Authentication not properly configured for test")
    assert response.status_code == 200
```

### After (New Composable Fixtures)
```python
def test_auth_endpoint_bearer(client_bearer_auth, auth_mode_bearer_token, auth_headers_factory):
    """New pattern - explicit auth mode, no skips"""
    headers = auth_headers_factory(auth_mode_bearer_token)
    response = client_bearer_auth.get("/v1/models", headers=headers)
    assert response.status_code == 200

def test_auth_endpoint_configured(client_configured_auth, auth_mode_configured_token, auth_headers_factory):
    """Test with server-configured token"""
    headers = auth_headers_factory(auth_mode_configured_token)
    response = client_configured_auth.get("/v1/models", headers=headers)
    assert response.status_code == 200

def test_auth_endpoint_no_auth(client_no_auth):
    """Test without authentication"""
    response = client_no_auth.get("/v1/models")
    assert response.status_code == 200
```

## Migration Steps

### 1. Replace Old Auth Fixtures

**Before:**
```python
def test_something(client_with_auth, auth_headers):
    # Old pattern
```

**After:**
```python
def test_something_bearer(client_bearer_auth, auth_mode_bearer_token, auth_headers_factory):
    headers = auth_headers_factory(auth_mode_bearer_token)
    # Test with bearer token

def test_something_configured(client_configured_auth, auth_mode_configured_token, auth_headers_factory):
    headers = auth_headers_factory(auth_mode_configured_token)
    # Test with configured token

def test_something_no_auth(client_no_auth):
    # Test without auth
```

### 2. Remove Test Skips

**Before:**
```python
def test_streaming(client_with_auth, auth_headers):
    response = client_with_auth.post("/v1/chat/completions", headers=auth_headers, json=data)
    if response.status_code == 401:
        pytest.skip("Authentication not properly configured for test")
```

**After:**
```python
@pytest.mark.parametrize("auth_mode,client_fixture,headers_factory", [
    ("bearer", "client_bearer_auth", "auth_headers_factory"),
    ("configured", "client_configured_auth", "auth_headers_factory"),
    ("none", "client_no_auth", None),
])
def test_streaming(request, auth_mode, client_fixture, headers_factory):
    client = request.getfixturevalue(client_fixture)

    if headers_factory:
        auth_config = request.getfixturevalue(f"auth_mode_{auth_mode}_token" if auth_mode != "bearer" else f"auth_mode_{auth_mode}_token")
        headers_func = request.getfixturevalue(headers_factory)
        headers = headers_func(auth_config)
    else:
        headers = {}

    response = client.post("/v1/chat/completions", headers=headers, json=data)
    assert response.status_code == 200
```

### 3. Use Custom Configurations

**For advanced scenarios:**
```python
def test_custom_auth_scenario(app_factory, client_factory, auth_test_utils):
    # Define custom auth configuration
    custom_config = {
        "mode": "custom_bearer",
        "requires_token": True,
        "has_configured_token": False,
        "test_token": "custom-test-token-123"
    }

    # Create app and client with custom config
    app = app_factory(custom_config)
    client = client_factory(app)

    # Test with custom headers
    headers = {"Authorization": f"Bearer {custom_config['test_token']}"}
    response = client.get("/v1/models", headers=headers)

    # Use auth test utilities
    assert auth_test_utils["is_auth_success"](response)
```

## Available Auth Modes

### 1. No Authentication (`auth_mode_none`)
- Server requires no authentication
- Use `client_no_auth` for testing

### 2. Bearer Token (`auth_mode_bearer_token`)  
- Server accepts any valid bearer token
- No server-side auth_token configured
- Use `client_bearer_auth` for testing

### 3. Configured Token (`auth_mode_configured_token`)
- Server has specific auth_token configured
- Bearer token must match server token
- Use `client_configured_auth` for testing

### 4. Credentials (`auth_mode_credentials`)
- Claude SDK credentials-based authentication
- OAuth flow simulation available
- Use `client_credentials_auth` for testing

### 5. Credentials with Fallback (`auth_mode_credentials_with_fallback`)
- Primary: Credentials authentication
- Fallback: Bearer token authentication
- Use with OAuth flow simulator

## Test Utilities

### Auth Test Utils
```python
def test_auth_validation(client_bearer_auth, auth_test_utils):
    response = client_bearer_auth.get("/v1/models")

    # Check auth status
    assert auth_test_utils["is_auth_error"](response)
    assert not auth_test_utils["is_auth_success"](response)

    # Extract error details
    error_detail = auth_test_utils["extract_auth_error_detail"](response)
    assert "Bearer token required" in error_detail
```

### OAuth Flow Simulator
```python
def test_oauth_flow(oauth_flow_simulator, mock_oauth):
    # Simulate successful OAuth
    oauth_response = oauth_flow_simulator["successful_oauth"]()
    assert oauth_response["access_token"] == "oauth-access-token-12345"

    # Simulate OAuth error
    error_response = oauth_flow_simulator["oauth_error"]()
    assert error_response["error"] == "invalid_grant"

    # Simulate token refresh
    refresh_response = oauth_flow_simulator["token_refresh"]()
    assert "refreshed-access-token" in refresh_response["access_token"]
```

## Benefits of New Approach

1. **No Test Skips**: Every auth mode has proper configuration
2. **Clear Separation**: Each auth mode is explicitly defined
3. **Composable**: Mix and match auth components as needed
4. **Type Safe**: All fixtures have proper type hints
5. **Maintainable**: Clear organization in dedicated auth module
6. **Comprehensive**: Covers all authentication scenarios

## Common Patterns

### Testing Multiple Auth Modes
```python
@pytest.mark.parametrize("auth_setup", [
    ("no_auth", "client_no_auth", None),
    ("bearer", "client_bearer_auth", "auth_mode_bearer_token"),
    ("configured", "client_configured_auth", "auth_mode_configured_token"),
])
def test_endpoint_all_auth_modes(request, auth_setup):
    mode_name, client_fixture, config_fixture = auth_setup
    client = request.getfixturevalue(client_fixture)

    if config_fixture:
        config = request.getfixturevalue(config_fixture)
        headers_factory = request.getfixturevalue("auth_headers_factory")
        headers = headers_factory(config)
    else:
        headers = {}

    response = client.get("/v1/models", headers=headers)
    assert response.status_code == 200
```

### Negative Testing
```python
def test_invalid_auth(client_configured_auth, auth_mode_configured_token, invalid_auth_headers_factory):
    headers = invalid_auth_headers_factory(auth_mode_configured_token)
    response = client_configured_auth.get("/v1/models", headers=headers)
    assert response.status_code == 401
```
