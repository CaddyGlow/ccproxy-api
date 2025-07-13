# Contributing to Claude Code Proxy API

Thank you for your interest in contributing to Claude Code Proxy API! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### Initial Setup

1. **Clone and setup the repository:**
   ```bash
   git clone https://github.com/CaddyGlow/claude-code-proxy-api.git
   cd claude-code-proxy-api
   make setup  # Installs dependencies and sets up dev environment
   ```

   > **Note**: Pre-commit hooks are automatically installed with `make setup` and `make dev-install`

## Code Quality Standards

This project maintains high code quality through automated checks that run both locally (via pre-commit) and in CI.

### Pre-commit Hooks vs Individual Commands

| Check | Pre-commit Hook | Individual Make Command | Purpose |
|-------|----------------|----------------------|---------|
| **Linting** | `ruff check --fix` | `make lint` | Code style and error detection |
| **Formatting** | `ruff format` | `make format` | Consistent code formatting |
| **Type Checking** | `mypy` | `make typecheck` | Static type validation |
| **Security** | `bandit` *(disabled)* | *(not available)* | Security vulnerability scanning |
| **File Hygiene** | Various hooks | *(not available individually)* | Trailing whitespace, EOF, etc. |
| **Tests** | *(not included)* | `make test` | Unit and integration tests |

**Key Differences:**

- **Pre-commit hooks**: Auto-fix issues, comprehensive file checks, runs on commit
- **Individual commands**: Granular control, useful for debugging specific issues
- **CI pipeline**: Runs pre-commit + tests (most comprehensive)

### Running Quality Checks

**Recommended Workflow:**
```bash
# Comprehensive checks with auto-fixes (RECOMMENDED)
make pre-commit    # or: uv run pre-commit run --all-files

# Full CI pipeline (pre-commit + tests)
make ci
```

**Alternative Commands:**
```bash
# Pre-commit only (runs automatically on commit)
uv run pre-commit run              # Run on staged files
uv run pre-commit run --all-files  # Run on all files

# Individual checks (for debugging)
make lint          # Linting only
make typecheck     # Type checking only  
make format        # Format code
make test          # Tests only
```

### Why Use Pre-commit for Most Checks?

Pre-commit hooks handle most quality checks because:

- **Auto-fixing**: Automatically fixes formatting and many linting issues
- **Comprehensive**: Includes file hygiene checks not available in individual commands
- **Consistent**: Same checks run locally and in CI
- **Fast**: Only checks changed files by default

**Tests run separately because:**

- **Speed**: Tests can be slow and would make commits frustrating
- **Scope**: Unit tests should pass, but integration tests might need external services  
- **CI Coverage**: Full test suite with coverage runs in CI pipeline (`make ci`)

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the existing patterns
- Add tests for new functionality
- Update documentation as needed

### 3. Pre-commit Validation
Pre-commit hooks will automatically run when you commit:
```bash
git add .
git commit -m "feat: add new feature"
# Pre-commit hooks run automatically and may modify files
# If files are modified, you'll need to add and commit again
```

### 4. Run Full Validation
```bash
make ci  # Runs pre-commit hooks + tests (recommended)

# Alternative: run components separately
make pre-commit  # Comprehensive checks with auto-fixes
make test        # Tests with coverage
```

### 5. Create Pull Request

- Push your branch and create a PR
- CI will run the full pipeline
- Address any CI failures

## CI/CD Workflows

The project uses **split CI/CD workflows** for efficient, parallel testing of backend and frontend components.

### Workflow Architecture

We use **two independent GitHub Actions workflows** rather than a single monolithic one:

| Workflow | Triggers | Purpose | Duration |
|----------|----------|---------|----------|
| **Backend CI** | Changes to `ccproxy/**`, `tests/**`, `pyproject.toml` | Python code quality & tests | ~3-5 min |
| **Frontend CI** | Changes to `dashboard/**` | TypeScript/Svelte quality & build | ~2-3 min |

### Backend Workflow (`.github/workflows/backend.yml`)

**Jobs:**
1. **Quality Checks** - ruff linting + mypy type checking
2. **Tests** - Unit tests across Python 3.10, 3.11, 3.12
3. **Build Verification** - Package build + CLI installation test

**Commands tested:**
```bash
make dev-install  # Dependency installation
make check        # Quality checks (lint + typecheck)
make test-unit    # Fast unit tests
make build        # Package build
```

### Frontend Workflow (`.github/workflows/frontend.yml`)

**Jobs:**
1. **Quality Checks** - Biome linting/formatting + TypeScript checks  
2. **Build & Test** - Dashboard build + verification + artifact upload

**Commands tested:**
```bash
bun install       # Dependency installation
bun run lint      # Biome linting
bun run format:check  # Biome formatting check
bun run check     # TypeScript + Biome checks
bun run build     # Dashboard build
bun run build:prod   # Production build + copy to ccproxy/static/
```

### Dashboard Development

The dashboard is a **SvelteKit SPA** with its own toolchain:

**Dependencies:**
```bash
# Install dashboard dependencies
make dashboard-install
# Or manually:
cd dashboard && bun install
```

**Quality Checks:**
```bash
# All dashboard checks
make dashboard-check
# Individual checks:
cd dashboard && bun run lint          # Biome linting  
cd dashboard && bun run format:check  # Format checking
cd dashboard && bun run check         # TypeScript + Biome
```

**Building:**
```bash
# Build for production (includes copy to ccproxy/static/)
make dashboard-build
# Or manually:
cd dashboard && bun run build:prod
```

**Cleaning:**
```bash
# Clean dashboard build artifacts
make dashboard-clean
```

### Path-Based Triggers

Workflows only run when relevant files change:

**Backend triggers:**
- `ccproxy/**` - Core Python application code
- `tests/**` - Test files
- `pyproject.toml` - Python dependencies
- `uv.lock` - Dependency lock file
- `Makefile` - Build configuration

**Frontend triggers:**
- `dashboard/**` - All dashboard files (SvelteKit app)

**Benefits:**
- **Faster feedback** - Only relevant checks run
- **Parallel execution** - Both workflows can run simultaneously
- **Resource efficiency** - Saves CI minutes
- **Clear failure isolation** - Know exactly what broke

### CI Status Checks

Both workflows must pass for PR merges:

- ✅ **Backend CI** - All Python quality checks and tests pass
- ✅ **Frontend CI** - All TypeScript/Svelte checks and build succeeds

### Local Testing

Test workflows locally before pushing:

**Backend:**
```bash
make check     # Same checks as CI quality job
make test-unit # Same tests as CI (without matrix)
make build     # Same build verification as CI
```

**Frontend:**
```bash
make dashboard-check  # Same checks as CI quality job
make dashboard-build  # Same build as CI
```

**Full pipeline:**
```bash
make ci               # Backend: pre-commit + tests
make dashboard-build  # Frontend: checks + build
```

### Troubleshooting CI Failures

**Backend failures:**
1. **Lint/Type errors**: Run `make check` locally and fix issues
2. **Test failures**: Run `make test-unit` and debug specific tests  
3. **Build failures**: Run `make build` and check for import errors

**Frontend failures:**
1. **TypeScript errors**: Run `cd dashboard && bun run check`
2. **Lint/Format errors**: Run `cd dashboard && bun run lint` and `bun run format`
3. **Build failures**: Run `cd dashboard && bun run build` and check for missing dependencies

**Path trigger issues:**
- Verify your changes match the path patterns in workflow files
- Force workflow run with empty commit: `git commit --allow-empty -m "trigger CI"`

## Code Style Guidelines

### Python Style

- **Line Length**: 88 characters (ruff default)
- **Imports**: Use absolute imports, sorted by isort
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for public functions/classes

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add user authentication
fix: resolve connection pool timeout
docs: update API documentation
test: add integration tests for streaming
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Quick test run (no coverage)  
make test-fast

# Run specific test file
make test-file FILE=test_auth.py

# Run tests matching a pattern
make test-match MATCH="auth"
```

### Writing Tests

- Put all tests in `tests/` directory
- Name test files clearly: `test_feature.py`
- Most tests should hit your API endpoints (integration-style)
- Only write isolated unit tests for complex logic
- Use fixtures in `conftest.py` for common setup
- Mock external services (Claude SDK, OAuth endpoints)

### What to Test

**Focus on:**
- API endpoints (both Anthropic and OpenAI formats)
- Authentication flows
- Request/response format conversion
- Error handling
- Streaming responses

**Skip:**
- Simple configuration
- Third-party library internals
- Logging

## Security

### Security Scanning
The project uses [Bandit](https://bandit.readthedocs.io/) for security scanning:

```bash
# Run security scan (currently disabled in pre-commit but available)
uv run bandit -c pyproject.toml -r ccproxy/
```

### Security Guidelines

- Never commit secrets or API keys
- Use environment variables for sensitive configuration
- Follow principle of least privilege
- Validate all inputs

## Documentation

### Building Documentation
```bash
make docs-build   # Build static documentation
make docs-serve   # Serve documentation locally
make docs-clean   # Clean documentation build files
```

### Development Server
```bash
make dev          # Start development server with auto-reload
make setup        # Quick setup for new contributors
```

### Documentation Files

- **API Docs**: Auto-generated from docstrings
- **User Guide**: Manual documentation in `docs/`
- **Examples**: Working examples in `examples/`

## Troubleshooting

### Pre-commit Issues
If pre-commit hooks fail:

1. **Check the output**: Pre-commit shows what failed and why
2. **Fix issues**: Address linting/formatting issues
3. **Re-stage and commit**: `git add . && git commit`

### Common Issues

**Mypy errors:**
```bash
# Run mypy manually to see full output
uv run mypy .
```

**Ruff formatting:**
```bash
# Auto-fix most issues
uv run ruff check --fix .
uv run ruff format .
```

**Test failures:**
```bash
# Run specific failing test
uv run pytest tests/test_specific.py::test_function -v
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/CaddyGlow/claude-code-proxy-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CaddyGlow/claude-code-proxy-api/discussions)
- **Documentation**: See `docs/` directory

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
