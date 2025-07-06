# Contributing

## Overview

We welcome contributions to the Claude Code Proxy API! This guide will help you get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/CaddyGlow/claude-code-proxy-api.git
   cd claude-code-proxy-api
   ```

2. **Set up development environment**
   ```bash
   # Using devenv (preferred)
   devenv shell

   # Or using uv
   uv sync
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Quality

Before submitting changes, ensure your code passes all quality checks:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy .

# Run tests
pytest

# Run all checks
ruff format . && ruff check . && mypy . && pytest
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation if needed
   - Follow existing code patterns

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Standards

- **Python 3.11+** compatibility
- **Type hints** for all functions and methods
- **Docstrings** for all public APIs
- **Tests** for all new functionality
- **ruff** formatting and linting
- **mypy** type checking

## Commit Messages

Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation updates
- `test:` - Test updates
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Testing

- Write unit tests for new functionality
- Use appropriate pytest markers
- Ensure tests pass in CI/CD
- Maintain test coverage above 80%

## Documentation

- Update relevant documentation
- Add examples for new features
- Update API reference if needed
- Test documentation builds locally
