# Contributing to CCProxy API

Thank you for your interest in contributing to CCProxy API! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git
- (Optional) [bun](https://bun.sh/) for Claude Code SDK installation

### Initial Setup

1. **Clone and setup the repository:**
   ```bash
   git clone https://github.com/CaddyGlow/ccproxy-api.git
   cd ccproxy-api
   make setup  # Installs dependencies and sets up dev environment
   ```

   > **Note**: Pre-commit hooks are automatically installed with `make setup`

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# Or: fix/bug-description, docs/update-something
```

### 2. Make Changes

- Follow existing code patterns (see CONVENTIONS.md)
- Add tests for new functionality
- Update documentation as needed

### 3. Quality Checks (Required Before Commits)

```bash
# Recommended: Run comprehensive checks with auto-fixes
make pre-commit

# Alternative: Run individual checks
make format      # Format code
make lint        # Check linting
make typecheck   # Check types
make test-unit   # Run fast tests
```

### 4. Commit Your Changes

Pre-commit hooks run automatically on commit:
```bash
git add specific/files.py  # Never use git add .
git commit -m "feat: add new feature"

# If hooks modify files, stage and commit again:
git add .
git commit -m "feat: add new feature"
```

### 5. Full Validation

Before pushing:
```bash
make ci  # Runs full CI pipeline locally (pre-commit + tests)
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

## Code Quality Standards

### Quality Gates

All code must pass these checks before merging:

| Check | Command | Purpose | Auto-fix |
|-------|---------|---------|----------|
| **Formatting** | `make format` | Code style consistency | ✅ |
| **Linting** | `make lint` | Error detection | Partial (`make lint-fix`) |
| **Type Checking** | `make typecheck` | Type safety | ❌ |
| **Tests** | `make test` | Functionality | ❌ |
| **Pre-commit** | `make pre-commit` | All checks combined | ✅ |

## Architecture: DI & Services

This project uses a container-first dependency injection (DI) pattern. Follow these rules when adding or refactoring code:

- Use the service container exclusively
  - Access services via `app.state.service_container` or FastAPI dependencies.
  - Never create new global singletons or module-level caches for services.

- Register services in the factory
  - Add new services to `ccproxy/services/factories.py` using `container.register_service(...)`.
  - Prefer constructor injection and small factory methods over service locators.

- Hook system is required
  - `HookManager` is created at startup and registered in the container.
  - FastAPI dep `HookManagerDep` is required; do not make it optional.

- No deprecated globals
  - Do not use `ccproxy.services.http_pool.get_pool_manager()` or any global helpers.
  - Always resolve `HTTPPoolManager` via `container.get_pool_manager()`.

- Settings access
  - Use `Settings.from_config(...)` in CLI/tools and tests. The legacy `get_settings()` helper was removed.

### Adding a New Service

1) Register in the factory:

```python
# ccproxy/services/factories.py
self._container.register_service(MyService, factory=self.create_my_service)

def create_my_service(self) -> MyService:
    settings = self._container.get_service(Settings)
    return MyService(settings)
```

2) Resolve via container in runtime code:

```python
container: ServiceContainer = request.app.state.service_container
svc = container.get_service(MyService)
```

3) For FastAPI dependencies, use the shared helper:

```python
# ccproxy/api/dependencies.py
MyServiceDep = Annotated[MyService, Depends(get_service(MyService))]
```

### Streaming and Hooks

- `StreamingHandler` must be constructed with a `HookManager` (the factory enforces this).
- Do not patch dependencies after construction; ensure ordering via DI.

### Testing with the Container

- Prefer constructing a `ServiceContainer(Settings.from_config(...))` in tests.
- Override services by re-registering instances for the type under test:

```python
container.register_service(MyService, instance=FakeMyService())
```

This pattern keeps tests isolated and avoids cross-test state.
### Running Tests

The CCProxy test suite uses a streamlined architecture with 606 focused tests organized by type:

```bash
# All tests with coverage (recommended)
make test

# Fast unit tests only - isolated components, service boundary mocking
make test-unit

# Integration tests - cross-component behavior, minimal mocking  
make test-integration

# Plugin tests - centralized plugin testing
make test-plugins

# Performance tests - benchmarks and load testing
make test-performance

# Coverage report with HTML output
make test-coverage

# Specific patterns
make test-file FILE=unit/auth/test_auth.py
make test-match MATCH="authentication"
make test-watch  # Auto-run on file changes
```

#### Test Organization

- **Unit tests** (`tests/unit/`): Fast, isolated tests with mocking at service boundaries only
- **Integration tests** (`tests/integration/`): Cross-component tests with minimal mocking
- **Plugin tests** (`tests/plugins/`): Centralized plugin testing by plugin name
- **Performance tests** (`tests/performance/`): Dedicated performance benchmarks

#### Test Architecture Principles

- **Clean boundaries**: Mock external services only, test real internal behavior
- **Type safety**: All tests require `-> None` return annotations and proper typing
- **Fast execution**: Unit tests run in milliseconds with no timing dependencies
- **Modern patterns**: Session-scoped fixtures, async factory patterns, streamlined fixtures

## Plugin Development

### Creating a New Plugin

1. **Create plugin structure:**
   ```
   ccproxy/plugins/your_plugin/
   ├── __init__.py
   ├── adapter.py          # Main interface (required)
   ├── plugin.py           # Plugin declaration (required)
   ├── routes.py           # API routes (optional)
   ├── transformers/       # Request/response transformation
   │   ├── request.py
   │   └── response.py
   ├── detection_service.py # Capability detection (optional)
   ├── format_adapter.py   # Protocol conversion (optional)
   └── auth/               # Authentication (optional)
       └── manager.py
   ```

2. **Implement the adapter (delegation pattern):**
   ```python
   from ccproxy.adapters.base import BaseAdapter

   class YourAdapter(BaseAdapter):
       async def handle_request(self, request, endpoint, method):
           context = self._build_provider_context()
           return await self.proxy_service.handle_provider_request(
               request, endpoint, method, context
           )
   ```

3. **Register in pyproject.toml:**
   ```toml
   [project.entry-points."ccproxy.plugins"]
   your_plugin = "plugins.your_plugin.plugin:Plugin"
   ```

4. **Add tests:**
   ```bash
   # Plugin tests are centralized under tests/plugins/
   tests/plugins/your_plugin/unit/test_manifest.py
   tests/plugins/your_plugin/unit/test_adapter.py
   tests/plugins/your_plugin/integration/test_basic.py
   ```

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add user authentication
fix: resolve connection pool timeout  
docs: update API documentation
test: add streaming integration tests
refactor: extract pricing service
chore: update dependencies
```

## CI/CD Pipeline

### GitHub Actions Workflows

| Workflow | Trigger | Checks |
|----------|---------|--------|
| **CI** | Push/PR to main, develop | Linting, types, tests (Python 3.11-3.13) |
| **Build** | Push to main | Docker image build and push |
| **Release** | Git tag/release | PyPI publish, Docker release |
| **Docs** | Push to main/dev | Documentation build and deploy |

### Local CI Testing

Test the full CI pipeline locally:
```bash
make ci  # Same as GitHub Actions CI workflow
```

## Common Development Tasks

### Running the Dev Server
```bash
make dev  # Starts with debug logging and auto-reload
```

### Debugging Requests
```bash
# Enable verbose logging
CCPROXY_VERBOSE_API=true \
CCPROXY_REQUEST_LOG_DIR=/tmp/ccproxy/request \
make dev

# View last request
scripts/show_request.sh
```

### Building and Testing Docker
```bash
make docker-build
make docker-run
```

### Documentation
```bash
make docs-build  # Build docs
make docs-serve  # Serve locally at http://localhost:8000
```

## Troubleshooting

### Type Errors
```bash
make typecheck
# Or for detailed output:
uv run mypy . --show-error-codes
```

### Formatting Issues
```bash
make format  # Auto-fixes most issues
```

### Linting Errors
```bash
make lint-fix  # Auto-fix what's possible
# Manual fix required for remaining issues
```

### Test Failures
```bash
# Run specific failing test with verbose output
uv run pytest tests/test_file.py::test_function -vvs

# Debug with print statements
uv run pytest tests/test_file.py -s
```

### Pre-commit Hook Failures
```bash
# Run manually to see all issues
make pre-commit

# Skip hooks temporarily (not recommended)
git commit --no-verify -m "WIP: debugging"
```

## Project Structure

```
ccproxy-api/
├── ccproxy/           # Core application
│   ├── api/           # FastAPI routes and middleware
│   ├── auth/          # Authentication system
│   ├── config/        # Configuration management
│   ├── core/          # Core utilities and interfaces
│   ├── models/        # Pydantic models
│   └── services/      # Business logic services
├── ccproxy/plugins/           # Provider plugins
│   ├── claude_api/    # Claude API plugin
│   ├── claude_sdk/    # Claude SDK plugin
│   ├── codex/         # OpenAI Codex plugin
│   └── ...           # Other plugins
├── tests/            # Test suite
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── fixtures/     # Test fixtures
├── docs/             # Documentation
├── scripts/          # Utility scripts
└── Makefile          # Development commands
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/CaddyGlow/ccproxy-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CaddyGlow/ccproxy-api/discussions)
- **Documentation**: See `docs/` directory and inline code documentation

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).
