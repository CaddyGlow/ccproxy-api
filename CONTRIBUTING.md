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

### Running Tests

```bash
# All tests with coverage
make test

# Fast unit tests only
make test-unit

# Integration tests
make test-integration

# Specific test file
make test-file FILE=unit/api/test_api.py

# Tests matching pattern
make test-match MATCH="authentication"

# Watch mode (auto-run on changes)
make test-watch

# Coverage report
make test-coverage
```

## Plugin Development

### Creating a New Plugin

1. **Create plugin structure:**
   ```
   plugins/your_plugin/
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
   tests/unit/plugins/test_your_plugin.py
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
├── plugins/           # Provider plugins
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
