# Streamlined Testing Guide for CCProxy

## Philosophy

After aggressive refactoring and architecture realignment, our testing philosophy is:
- **Clean boundaries**: Unit tests for isolated components, integration tests for cross-component behavior
- **Fast execution**: Unit tests run in milliseconds, mypy completes in seconds  
- **Modern patterns**: Type-safe fixtures, clear separation of concerns
- **Minimal mocking**: Only mock external services, test real internal behavior

## Quick Start

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/auth/          # Authentication tests
pytest tests/unit/services/      # Service layer tests
pytest tests/integration/        # Cross-component integration tests (core)
pytest tests/plugins             # All plugin tests
pytest tests/plugins/metrics     # Single plugin tests
pytest tests/performance/        # Performance benchmarks

# Run with coverage
make test-coverage

# Type checking and quality (now sub-second)
make typecheck
make pre-commit
```

## Streamlined Test Structure

**Clean architecture after aggressive refactoring** - Removed 180+ tests and 3000+ lines of problematic code:

```
tests/
├── conftest.py              # Essential fixtures (515 lines, was 1117)
├── unit/                    # True unit tests (mock at service boundaries)
│   ├── api/                 # Remaining lightweight API tests
│   │   ├── test_mcp_route.py # MCP permission routes
│   │   ├── test_plugins_status.py # Plugin status endpoint
│   │   ├── test_reset_endpoint.py # Reset endpoint
│   │   └── test_analytics_pagination_service.py # Pagination service
│   ├── services/            # Core service tests
│   │   ├── test_adapters.py # OpenAI↔Anthropic conversion
│   │   ├── test_streaming.py # Streaming functionality
│   │   ├── test_confirmation_service.py # Confirmation service (cleaned)
│   │   ├── test_scheduler.py # Scheduler (simplified)
│   │   ├── test_scheduler_tasks.py # Task management
│   │   ├── test_claude_sdk_client.py # Claude SDK client
│   │   └── test_pricing.py  # Token pricing
│   ├── auth/                # Authentication tests
│   │   ├── test_auth.py     # Core auth (cleaned of HTTP testing)
│   │   ├── test_oauth_registry.py # OAuth registry
│   │   ├── test_authentication_error.py # Error handling
│   │   └── test_refactored_auth.py # Refactored patterns
│   ├── config/              # Configuration tests
│   │   ├── test_claude_sdk_options.py # Claude SDK config
│   │   ├── test_claude_sdk_parser.py # Config parsing
│   │   ├── test_config_precedence.py # Priority handling
│   │   └── test_terminal_handler.py # Terminal handling
│   ├── utils/               # Utility tests
│   │   ├── test_binary_resolver.py # Binary resolution
│   │   ├── test_startup_helpers.py # Startup utilities
│   │   └── test_version_checker.py # Version checking
│   ├── cli/                 # CLI command tests
│   │   ├── test_cli_config.py # CLI configuration
│   │   ├── test_cli_serve.py # Server CLI
│   │   └── test_cli_confirmation_handler.py # Confirmation CLI
│   ├── test_caching.py      # Caching functionality
│   ├── test_plugin_system.py # Plugin system (cleaned)
│   └── test_hook_ordering.py # Hook ordering
├── integration/             # Cross-component tests (moved from unit)
│   ├── test_analytics_pagination.py # Full analytics flow
│   ├── test_confirmation_integration.py # Permission flows
│   ├── test_metrics_plugin.py # Metrics collection
│   ├── test_plugin_format_adapters_v2.py # Format adapter system
│   ├── test_plugins_health.py # Plugin health checks
│   └── docker/             # Docker integration tests (moved)
│       └── test_docker.py  # Docker functionality
├── performance/             # Performance tests (separated)
│   └── test_format_adapter_performance.py # Benchmarks
├── factories/               # Simplified factories (362 lines, was 651)
│   ├── __init__.py         # Factory exports
│   └── fastapi_factory.py  # Streamlined FastAPI factories
├── fixtures/               # Essential fixtures only
│   ├── claude_sdk/         # Claude SDK mocking
│   ├── external_apis/      # External API mocking
│   └── responses.json      # Mock data
├── helpers/                # Test utilities
├── ccproxy/plugins/                # Plugin tests (centralized)
│   ├── my_plugin/
│   │   ├── unit/          # Plugin unit tests
│   │   └── integration/   # Plugin integration tests
└── test_handler_config.py  # Handler configuration tests
```

## Writing Tests

### Clean Architecture Principles

**Unit Tests** (tests/unit/):
- Mock at **service boundaries only** - never mock internal components
- Test **pure functions and single components** in isolation
- **No HTTP layer testing** - use service layer mocks instead
- **No timing dependencies** - all asyncio.sleep() removed
- **No database operations** - moved to integration tests

**Integration Tests** (tests/integration/):
- Test **cross-component interactions** with minimal mocking
- Include **HTTP client testing with FastAPI TestClient**
- Test **background workers and async coordination**
- Validate configuration end-to-end

### Mocking Strategy (Simplified)

- **External APIs only**: Claude API, OAuth endpoints, Docker processes
- **Internal services**: Use real implementations with dependency injection
- **Configuration**: Use test settings objects, not mocks
- **No mock explosion**: Removed 300+ redundant test fixtures

## Type Safety and Code Quality

**REQUIREMENT**: All test files MUST pass type checking and linting. This is not optional.

### Type Safety Requirements

1. **All test files MUST pass mypy type checking** - No `Any` types unless absolutely necessary
2. **All test files MUST pass ruff formatting and linting** - Code must be properly formatted
3. **Add proper type hints to all test functions and fixtures** - Include return types and parameter types
4. **Import necessary types** - Use `from typing import` for type annotations

### Required Type Annotations

- **Test functions**: Must have `-> None` return type annotation
- **Fixtures**: Must have proper return type hints
- **Parameters**: Must have type hints where not inferred from fixtures
- **Variables**: Add type hints for complex objects when not obvious

### Examples with Proper Typing

#### Basic Test Function with Types

```python
from typing import Any
import pytest
from fastapi.testclient import TestClient

def test_service_endpoint(client: TestClient) -> None:
    """Test service endpoint with proper typing."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert "models" in data
```

#### Fixture with Type Annotations

```python
from typing import Generator
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.fixture
def app() -> FastAPI:
    """Create test FastAPI application."""
    from ccproxy.api.app import create_app
    return create_app()

@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client
```

## Streamlined Fixtures Architecture

### Essential Fixtures (Simplified)

After aggressive cleanup, we maintain only essential, well-typed fixtures:

#### Core Integration Fixtures

- `integration_app_factory` - Dynamic FastAPI app creation with plugin configs
- `integration_client_factory` - Creates async HTTP clients with custom settings
- `metrics_integration_client` - Session-scoped client for metrics tests (high performance)
- `disabled_plugins_client` - Session-scoped client with plugins disabled
- `base_integration_settings` - Minimal settings for fast test execution
- `test_settings` - Clean test configuration
- `isolated_environment` - Temporary directory isolation

#### Authentication (Streamlined)

- `auth_settings` - Basic auth configuration
- `claude_sdk_environment` - Claude SDK test environment
- Simple auth patterns without combinatorial explosion

#### Essential Service Mocks (External Only)

- External API mocking only (Claude API, OAuth endpoints)
- No internal service mocking - use real implementations
- Removed 200+ redundant mock fixtures

#### Test Data

- `claude_responses` - Essential Claude API responses
- `mock_claude_stream` - Streaming response patterns
- Removed complex test data generators

## Test Markers

- `@pytest.mark.unit` - Fast unit tests (default)
- `@pytest.mark.integration` - Cross-component integration tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.asyncio` - Async test functions

## Best Practices

1. **Clean boundaries** - Unit tests mock at service boundaries only
2. **Fast execution** - Unit tests run in milliseconds, no timing dependencies
3. **Type safety** - All fixtures properly typed, mypy compliant
4. **Real components** - Test actual internal behavior, not mocked responses
5. **Performance-optimized patterns** - Use session-scoped fixtures for expensive operations
6. **Modern async patterns** - `@pytest.mark.asyncio(loop_scope="session")` for integration tests
7. **No overengineering** - Removed 180+ tests, 3000+ lines of complexity

### Performance Guidelines

#### When to Use Session-Scoped Fixtures
- **Plugin integration tests** - Plugin initialization is expensive
- **Database/external service tests** - Connection setup overhead
- **Complex app configuration** - Multiple services, middleware stacks
- **Consistent test state needed** - Tests require same app configuration

#### When to Use Factory Patterns  
- **Dynamic configurations** - Each test needs different plugin settings
- **Isolation required** - Tests might interfere with shared state
- **Simple setup** - Minimal overhead for app creation

#### Logging Performance Tips
- **Use `ERROR` level** - Minimal logging for faster test execution
- **Disable JSON logs** - `json_logs=False` for better performance
- **Disable plugin logging** - `enable_plugin_logging=False` in test settings
- **Manual setup required** - Call `setup_logging()` explicitly in test environment

## Common Patterns

### Performance-Optimized Integration Patterns

#### Session-Scoped Pattern (Recommended for Plugin Tests)

```python
import pytest
from httpx import AsyncClient

# Use session-scoped app creation for expensive plugin initialization
@pytest.mark.asyncio(loop_scope="session")
async def test_plugin_functionality(metrics_integration_client) -> None:
    """Test plugin with session-scoped app for optimal performance."""
    # App is created once per test session, client per test
    resp = await metrics_integration_client.get("/metrics")
    assert resp.status_code == 200
    assert "prometheus_metrics" in resp.text
```

#### Factory Pattern for Dynamic Configuration

```python
@pytest.mark.asyncio
async def test_dynamic_plugin_config(integration_client_factory) -> None:
    """Test with dynamic plugin configuration."""
    client = await integration_client_factory({
        "metrics": {"enabled": True, "custom_setting": "value"}
    })
    async with client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
```

### Basic Unit Test Pattern

```python
from ccproxy.utils.caching import TTLCache

def test_cache_basic_operations() -> None:
    """Test cache basic operations."""
    cache: TTLCache[str, int] = TTLCache(maxsize=10, ttl=60)

    # Test real cache behavior
    cache["key"] = 42
    assert cache["key"] == 42
    assert len(cache) == 1
```

### Integration Test Patterns

#### Session-Scoped App Pattern (High Performance)

For integration tests that need consistent app state and optimal performance:

```python
import pytest
from httpx import AsyncClient

# Session-scoped app creation (expensive operations done once)
@pytest.fixture(scope="session")
def metrics_integration_app():
    """Pre-configured app for metrics plugin integration tests."""
    from ccproxy.core.logging import setup_logging
    from ccproxy.config.settings import Settings
    from ccproxy.api.bootstrap import create_service_container
    from ccproxy.api.app import create_app

    # Set up logging once per session
    setup_logging(json_logs=False, log_level_name="ERROR")

    settings = Settings(
        enable_plugins=True,
        plugins={
            "metrics": {
                "enabled": True,
                "metrics_endpoint_enabled": True,
            }
        },
        logging={
            "level": "ERROR",  # Minimal logging for speed
            "enable_plugin_logging": False,
            "verbose_api": False,
        },
    )

    service_container = create_service_container(settings)
    return create_app(service_container), settings

# Test-scoped client (reuses shared app)
@pytest.fixture
async def metrics_integration_client(metrics_integration_app):
    """HTTP client for metrics integration tests."""
    from httpx import ASGITransport, AsyncClient
    from ccproxy.api.app import initialize_plugins_startup

    app, settings = metrics_integration_app
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

# Test using session-scoped pattern
@pytest.mark.asyncio(loop_scope="session")
async def test_metrics_endpoint_available(metrics_integration_client) -> None:
    """Test metrics endpoint availability."""
    resp = await metrics_integration_client.get("/metrics")
    assert resp.status_code == 200
    assert b"# HELP" in resp.content or b"# TYPE" in resp.content
```

#### Dynamic Factory Pattern (Flexible Configuration)

For tests that need different configurations:

```python
@pytest.mark.asyncio
async def test_custom_plugin_config(integration_client_factory) -> None:
    """Test with custom plugin configuration."""
    client = await integration_client_factory({
        "metrics": {
            "enabled": True,
            "metrics_endpoint_enabled": True,
            "include_labels": True,
        }
    })

    async with client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        # Test custom configuration behavior
        assert "custom_label" in resp.text
```

### Testing with Configuration

```python
from pathlib import Path
from ccproxy.config.settings import Settings

def test_config_loading(tmp_path: Path) -> None:
    """Test configuration file loading."""
    config_file: Path = tmp_path / "config.toml"
    config_file.write_text("port = 8080")

    settings: Settings = Settings(_config_file=config_file)
    assert settings.server.port == 8080
```

## Quality Checks Commands

```bash
# Type checking (MUST pass) - now sub-second
make typecheck
uv run mypy tests/

# Linting and formatting (MUST pass)
make lint
make format
uv run ruff check tests/
uv run ruff format tests/

# Run all quality checks
make pre-commit
```

## Dev Scripts (Optional Helpers)

Convenience scripts live in `scripts/` to speed up local testing and debugging:

- `scripts/debug-no-stream-all.sh`: exercise non-streaming endpoints quickly
- `scripts/debug-stream-all.sh`: exercise streaming endpoints
- `scripts/show_request.sh` / `scripts/last_request.sh`: inspect recent requests
- `scripts/test_streaming_metrics_all.py`: ad-hoc streaming metrics checks
- `scripts/run_integration_tests.py`: advanced integration runner (filters, timing)

These are optional helpers for dev workflows; standard Make targets and pytest remain the primary interface.

## Running Tests

### Make Commands

```bash
make test                 # Run all tests with coverage
make test-unit            # Fast unit tests only
make test-integration     # Integration tests (core + plugins)
make test-integration-plugin PLUGIN=metrics  # Single plugin integration
make test-plugins         # Only plugin tests
make test-coverage        # With coverage report
```

### Direct pytest

```bash
pytest -v                          # Verbose output
pytest -k "test_auth"              # Run matching tests
pytest --lf                        # Run last failed
pytest -x                          # Stop on first failure
pytest --pdb                       # Debug on failure
pytest -m unit                     # Unit tests only
pytest -m integration              # Integration tests only
pytest tests/plugins               # All plugin tests
pytest tests/plugins/metrics -m unit  # Single plugin unit tests

Note: tests run with `--import-mode=importlib` via Makefile to avoid module name clashes.
```

## For New Developers

1. **Start here**: Read this file and `tests/fixtures/integration.py`
2. **Run tests**: `make test` to ensure everything works (606 optimized tests)
3. **Choose pattern**:
   - Session-scoped fixtures for plugin tests (`metrics_integration_client`)
   - Factory patterns for dynamic configs (`integration_client_factory`)
   - Unit tests for isolated components
4. **Performance first**: Use `ERROR` logging level, session-scoped apps for expensive operations
5. **Type safety**: All test functions need `-> None` return type, proper fixture typing
6. **Modern async**: Use `@pytest.mark.asyncio(loop_scope="session")` for integration tests
7. **Mock external only**: Don't mock internal components, test real behavior

## Migration from Old Architecture

**All existing test patterns still work** - but new tests should use the performance-optimized patterns:

### Current Recommended Patterns (2024)

- **Session-scoped integration fixtures** - `metrics_integration_client`, `disabled_plugins_client`
- **Async factory patterns** - `integration_client_factory` for dynamic configs
- **Manual logging setup** - `setup_logging(json_logs=False, log_level_name="ERROR")`
- **Session loop scope** - `@pytest.mark.asyncio(loop_scope="session")` for integration tests
- **Service container pattern** - `create_service_container()` + `create_app()`
- **Plugin lifecycle management** - `initialize_plugins_startup()` in fixtures

### Performance Optimizations Applied

- **Minimal logging** - ERROR level only, no JSON logging, plugin logging disabled
- **Session-scoped apps** - Expensive plugin initialization done once per session  
- **Streamlined fixtures** - 515 lines (was 1117), focused on essential patterns
- **Real component testing** - Mock external APIs only, test actual internal behavior

Plugin tests are now centralized under `tests/plugins/<plugin>/{unit,integration}` instead of co-located in `plugins/<plugin>/tests`. Update any paths and imports accordingly.

The architecture has been significantly optimized for performance while maintaining full functionality.
