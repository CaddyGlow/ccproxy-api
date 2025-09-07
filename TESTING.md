# Streamlined Testing Guide for CCProxy

## Philosophy

After aggressive refactoring and architecture realignment, our testing philosophy is:
- **Clean boundaries**: Unit tests for isolated components, integration tests for cross-component behavior
- **Fast execution**: Unit tests run in milliseconds, mypy completes in seconds  
- **Modern patterns**: Type-safe fixtures, clear separation of concerns
- **Minimal mocking**: Only mock external services, test real internal behavior

## Quick Start

```bash
# Run all tests (now 606 focused tests)
make test

# Run specific test categories
pytest tests/unit/auth/          # Authentication tests
pytest tests/unit/services/      # Service layer tests
pytest tests/integration/        # Cross-component integration tests
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
- Validate **configuration and feature flags end-to-end**

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

#### Core FastAPI Fixtures

- `fastapi_app_factory` - Creates FastAPI apps with any configuration  
- `fastapi_client_factory` - Creates test clients (simplified, 362 lines vs 651)
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
5. **Simplified factories** - Streamlined FastAPI factory (44% smaller)
6. **Modern patterns** - @pytest.mark.asyncio, proper async fixtures
7. **No overengineering** - Removed 180+ tests, 3000+ lines of complexity

## Common Patterns

### Streamlined Factory Pattern

```python
from fastapi.testclient import TestClient

def test_service_integration(fastapi_client_factory, test_settings) -> None:
    """Test service with real internal components."""
    client = fastapi_client_factory.create_client(
        settings=test_settings
    )
    # Test real service behavior, not mocked responses
    response = client.get("/api/models")
    assert response.status_code == 200
    assert "models" in response.json()
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

### Integration Test Pattern

```python
from fastapi.testclient import TestClient

def test_full_request_flow(client: TestClient) -> None:
    """Test complete request flow (integration test)."""
    # This tests real HTTP layer + services + configuration
    response = client.post("/api/v1/messages", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
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

## Running Tests

### Make Commands

```bash
make test              # Run all tests (606 tests, fast)
make test-unit         # Fast unit tests only (sub-second)
make test-integration  # Integration tests
make test-performance  # Performance benchmarks
make test-coverage     # With coverage report
```

### Direct pytest

```bash
pytest -v                          # Verbose output
pytest -k "test_auth"              # Run matching tests
pytest --lf                        # Run last failed
pytest -x                          # Stop on first failure
pytest --pdb                       # Debug on failure
pytest -m unit                     # Unit tests only
pytest -m integration             # Integration tests only
```

## For New Developers

1. **Start here**: Read this file and `tests/conftest.py`
2. **Run tests**: `make test` to ensure everything works
3. **Add new test**: Copy existing test pattern, modify as needed
4. **Mock external only**: Don't mock internal components
5. **Type safety**: All test functions need proper type hints
6. **Fast by default**: Unit tests should run in milliseconds

## Migration from Old Architecture

**All existing test patterns still work** - but new tests should use the streamlined patterns:

- Use `fastapi_client_factory` for flexible test clients
- Mock at service boundaries, not HTTP layer
- Prefer unit tests with real internal components
- Move complex scenarios to integration tests
- Use proper type hints on all fixtures

The architecture has been significantly simplified while maintaining functionality and improving maintainability.