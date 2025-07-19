# FastAPI Factory Migration Guide

This guide explains how to migrate from the old combinatorial FastAPI fixtures to the new factory pattern.

## Problem Solved

**Before**: We had multiple similar fixtures that created combinatorial explosion:
- `app`, `app_with_mock_claude`, `app_with_unavailable_claude`, `app_with_auth`
- `client`, `client_with_mock_claude`, `client_with_unavailable_claude`, `client_with_auth`
- Each combination required a separate fixture (exponential growth)

**After**: One factory that can compose different configurations dynamically.

## Migration Examples

### Basic Usage (No Changes Required)

```python
# OLD: Using existing fixtures
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

# NEW: Same test works with factory (backward compatible)
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
```

### Mocked Claude Service

```python
# OLD: Required separate fixture
def test_claude_api(client_with_mock_claude):
    response = client_with_mock_claude.post("/api/v1/chat/completions", json={...})
    assert response.status_code == 200

# NEW: Use factory for flexibility
def test_claude_api(fastapi_client_factory, mock_claude_service):
    client = fastapi_client_factory.create_client(claude_service_mock=mock_claude_service)
    response = client.post("/api/v1/chat/completions", json={...})
    assert response.status_code == 200
```

### Authentication

```python
# OLD: Required separate fixture
def test_authenticated_endpoint(client_with_auth):
    headers = {"Authorization": "Bearer test-token-12345"}
    response = client_with_auth.post("/api/v1/chat/completions",
                                     headers=headers, json={...})
    assert response.status_code == 200

# NEW: Use factory with auth enabled
def test_authenticated_endpoint(fastapi_client_factory, auth_settings):
    client = fastapi_client_factory.create_client(
        settings=auth_settings,
        auth_enabled=True
    )
    headers = {"Authorization": "Bearer test-token-12345"}
    response = client.post("/api/v1/chat/completions",
                          headers=headers, json={...})
    assert response.status_code == 200
```

### Complex Combinations (Major Improvement)

```python
# OLD: Would require creating new fixture for each combination
# No clean way to combine auth + mock_claude + custom_settings

# NEW: Easy composition
def test_auth_with_mock_and_custom_settings(
    fastapi_client_factory, auth_settings, mock_claude_service
):
    # Modify settings for this test
    auth_settings.server.log_level = "DEBUG"

    client = fastapi_client_factory.create_client(
        settings=auth_settings,
        claude_service_mock=mock_claude_service,
        auth_enabled=True
    )

    headers = {"Authorization": "Bearer test-token-12345"}
    response = client.post("/api/v1/chat/completions",
                          headers=headers, json={...})
    assert response.status_code == 200
```

### Async Clients

```python
# OLD: Separate async fixtures needed
@pytest.mark.asyncio
async def test_async_endpoint(async_client_with_mock_claude_streaming):
    response = await async_client_with_mock_claude_streaming.post(...)
    assert response.status_code == 200

# NEW: Factory approach
@pytest.mark.asyncio
async def test_async_endpoint(fastapi_client_factory, mock_claude_service_streaming):
    async with fastapi_client_factory.create_async_client(
        claude_service_mock=mock_claude_service_streaming
    ) as client:
        response = await client.post(...)
        assert response.status_code == 200
```

### Custom Dependency Overrides

```python
# NEW: Advanced usage with custom overrides
def test_custom_dependencies(fastapi_client_factory, test_settings):
    from ccproxy.api.dependencies import get_credentials_manager

    # Mock credentials manager
    mock_creds = AsyncMock()

    client = fastapi_client_factory.create_client(
        settings=test_settings,
        dependency_overrides={
            get_credentials_manager: lambda: mock_creds
        }
    )

    response = client.get("/api/models")
    assert response.status_code == 200
```

## Factory API Reference

### FastAPIAppFactory

```python
factory = FastAPIAppFactory(default_settings=test_settings)

app = factory.create_app(
    settings=None,                    # Optional: override default settings
    dependency_overrides=None,        # Optional: custom dependency overrides
    claude_service_mock=None,         # Optional: mock Claude service
    auth_enabled=False,               # Optional: enable authentication
    **kwargs                          # Optional: additional configuration
)
```

### FastAPIClientFactory

```python
client_factory = FastAPIClientFactory(app_factory)

# Sync client
client = client_factory.create_client(...)

# Async client
async with client_factory.create_async_client(...) as client:
    # Use client
    pass
```

## Available Fixtures

### New Factory Fixtures

- `fastapi_app_factory`: Main factory for creating FastAPI apps
- `fastapi_client_factory`: Main factory for creating test clients
- `async_client_factory`: Factory-based async client factory (returns the factory itself)

### Legacy Fixtures (Maintained for Backward Compatibility)

All existing fixtures still work but now delegate to the factory internally:

- `app`, `client`, `async_client` - Basic fixtures
- `app_with_mock_claude`, `client_with_mock_claude` - Mocked Claude service
- `app_with_auth`, `client_with_auth` - Authentication enabled
- `app_with_unavailable_claude`, `client_with_unavailable_claude` - Unavailable Claude

### Additional Factory-Based Legacy Fixtures

For common combinations:

- `app_factory_with_auth_and_mock` - Auth + Mock Claude
- `client_factory_with_auth_and_mock` - Client with Auth + Mock Claude

## Migration Strategy

### Phase 1: No Changes Required (Done)
- All existing tests continue to work
- Legacy fixtures maintained for backward compatibility

### Phase 2: Gradual Migration (Recommended)
- New tests should use the factory pattern
- Complex scenarios should migrate to factories for better flexibility
- Simple tests can remain using legacy fixtures

### Phase 3: Full Migration (Optional)
- Eventually deprecate and remove legacy fixtures
- All tests use factory pattern for consistency

## Benefits of Factory Pattern

1. **Eliminates Combinatorial Explosion**: One factory instead of N×M fixtures
2. **Improved Composability**: Mix and match configurations easily
3. **Better Test Flexibility**: Dynamic configuration per test
4. **Reduced Code Duplication**: Single implementation, multiple configurations
5. **Easier Maintenance**: One place to update FastAPI app creation logic
6. **Type Safety**: Proper type hints throughout
7. **Backward Compatibility**: Existing tests don't break

## Example: Before vs After for New Combinations

```python
# BEFORE: Would need to create new fixtures for each combination
# This would require creating 8+ new fixtures:
# - app_with_auth_and_mock_streaming
# - client_with_auth_and_mock_streaming  
# - app_with_auth_and_unavailable_claude
# - client_with_auth_and_unavailable_claude
# - etc.

# AFTER: One factory handles all combinations
def test_all_combinations(
    fastapi_client_factory,
    auth_settings,
    mock_claude_service_streaming,
    mock_claude_service_unavailable
):
    # Auth + Mock Streaming
    client1 = fastapi_client_factory.create_client(
        settings=auth_settings,
        claude_service_mock=mock_claude_service_streaming,
        auth_enabled=True
    )

    # Auth + Unavailable Claude
    client2 = fastapi_client_factory.create_client(
        settings=auth_settings,
        claude_service_mock=mock_claude_service_unavailable,
        auth_enabled=True
    )

    # Custom combination with overrides
    client3 = fastapi_client_factory.create_client(
        settings=auth_settings,
        claude_service_mock=mock_claude_service_streaming,
        auth_enabled=True,
        dependency_overrides={...}
    )
```

This demonstrates how the factory pattern scales linearly instead of exponentially with configuration options.
