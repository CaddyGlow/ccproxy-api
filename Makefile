.PHONY: help install dev-install clean fix fix-hard test test-unit test-integration test-plugins test-file test-match test-coverage lint lint-fix typecheck format format-check check check-boundaries pre-commit ci build build-backend build-dashboard dashboard docker-build docker-run docker-compose-up docker-compose-down dev prod docs-install docs-build docs-serve docs-clean docs-deploy setup

# Determine Docker tag from git (fallback to 'latest')
$(eval VERSION_DOCKER := $(shell git describe --tags --always --dirty=-dev 2>/dev/null || echo latest))

# Common variables
UV ?= uv
UV_RUN := $(UV) run
PYTEST := $(UV_RUN) pytest --import-mode=importlib
RUFF := $(UV_RUN) ruff
MYPY := $(UV_RUN) mypy
PRE_COMMIT := $(UV_RUN) pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  clean        - Clean build artifacts"
	@echo ""
	@echo "Testing commands:"
	@echo "  test         - Run all tests with coverage"
	@echo "  test-unit    - Run fast unit tests only (excluding real API and integration)"
	@echo "  test-integration - Run integration tests across all plugins (parallel)"
	@echo "  test-coverage - Run tests with detailed coverage report"
	@echo ""
	@echo "Code quality:"
	@echo "  lint         - Run linting checks"
	@echo "  typecheck    - Run type checking"
	@echo "  format       - Format code"
	@echo "  check        - Run all checks (lint + typecheck)"
	@echo "  check-boundaries - Ensure core does not import from plugins"
	@echo "  pre-commit   - Run pre-commit hooks (comprehensive checks + auto-fixes)"
	@echo "  ci           - Run full CI pipeline (pre-commit + test)"
	@echo ""
	@echo "Build and deployment:"
	@echo "  build        - Build Python package"
	@echo "  build-backend - Alias for backend build only"
	@echo "  build-dashboard - Build dashboard only"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo ""
	@echo "Dashboard (frontend):"
	@echo "  dashboard         - Show dashboard commands (run make -C dashboard help)"
	@echo ""
	@echo "Documentation:"
	@echo "  docs-install - Install documentation dependencies"
	@echo "  docs-build   - Build documentation"
	@echo "  docs-serve   - Serve documentation locally"
	@echo "  docs-clean   - Clean documentation build files"

# Installation targets
install:
	$(UV) sync --no-dev

dev-install:
	$(UV) sync --all-extras --all-groups --dev
	$(PRE_COMMIT) install

# Cleanup
clean:
	rm -rf dist build *.egg-info node_modules
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .coverage coverage.xml pnpm-lock.yaml

# Fix code with unsafe fixes
fix-hard:
	$(RUFF) check . --fix --unsafe-fixes || true
	$(RUFF) check . --select F401 --fix --unsafe-fixes || true # Used variable import
	$(RUFF) check . --select I --fix --unsafe-fixes || true  # Import order
	$(RUFF) format . || true


fix: format lint-fix
	$(RUFF) check . --fix --unsafe-fixes

# Run all tests with coverage
test:
	@echo "Running all tests with coverage..."
	@if [ ! -d "tests" ]; then echo "Error: tests/ directory not found. Create tests/ directory and add test files."; exit 1; fi
	$(PYTEST) -v --cov=ccproxy --cov-report=term

# New test suite targets

# Run fast unit tests only (exclude tests marked with 'real_api' and 'integration')
test-unit:
	@echo "Running fast unit tests (excluding real API calls and integration tests)..."
	$(PYTEST) -v -m "not integration" --tb=short

# Run integration tests across all plugins
test-integration:
	@echo "Running integration tests across all plugins..."
	$(PYTEST) -v -m "integration and not slow and not real_api" --tb=short -n auto tests/

# Run tests with detailed coverage report (HTML + terminal)
test-coverage: check
	@echo "Running tests with detailed coverage report..."
	$(PYTEST) -v --cov=ccproxy --cov-report=term-missing --cov-report=html -m "not slow and not real_api"
	@echo "HTML coverage report generated in htmlcov/"

# Run plugin tests only
test-plugins:
	@echo "Running plugin tests under tests/plugins..."
	$(PYTEST) -v --tb=short --no-cov tests/plugins

# Run specific test file (with quality checks)
test-file: check
	@echo "Running specific test file: $(FILE)"
	$(PYTEST) -v $(FILE)

# Run tests matching a pattern (with quality checks)
test-match: check
	@echo "Running tests matching pattern: $(MATCH)"
	$(PYTEST) -v -k "$(MATCH)"

# Code quality
lint:
	$(RUFF) check .

lint-fix:
	$(RUFF) check . --fix

typecheck:
	$(MYPY) .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

# Combined checks (individual targets for granular control)
check: lint typecheck format-check check-boundaries

check-boundaries:
	$(UV_RUN) python scripts/check_import_boundaries.py


# Pre-commit hooks (comprehensive checks + auto-fixes)
pre-commit:
	$(PRE_COMMIT) run --all-files

# Full CI pipeline (comprehensive: pre-commit does more checks + auto-fixes)
ci:
	$(PRE_COMMIT) run --all-files
	$(MAKE) test

# Build targets
build:
	$(UV) build

build-backend: build

build-dashboard:
	$(MAKE) -C dashboard build

# Dashboard delegation
dashboard:
	@echo "Dashboard commands:"
	@echo "Use 'make -C dashboard <target>' to run dashboard commands"
	@echo "Available dashboard targets:"
	@$(MAKE) -C dashboard help

# Docker targets
docker-build:
	docker build -t ghcr.io/caddyglow/ccproxy:$(VERSION_DOCKER) .

docker-run:
	docker run --rm -p 8000:8000 ghcr.io/caddyglow/ccproxy:$(VERSION_DOCKER)

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down

# Development server
dev:
	LOGGING__LEVEL=trace \
		LOGGING__FILE=/tmp/ccproxy/ccproxy.log \
		LOGGING__VERBOSE_API=true \
		LOGGING__PLUGIN_LOG_BASE_DIR=/tmp/ccproxy \
		PLUGINS__REQUEST_TRACER__ENABLED=true \
		PLUGINS__ACCESS_LOG__ENABLED=true \
		PLUGINS__ACCESS_LOG__CLIENT_LOG_FILE=/tmp/ccproxy/combined_access.log \
		PLUGINS__ACCESS_LOG__CLIENT_FORMAT=combined \
		HTTP__COMPRESSION_ENABLED=false \
		SERVER__RELOAD=true \
		SERVER__WORKERS=1 \
		$(UV_RUN) ccproxy-api serve

prod:
	$(UV_RUN) ccproxy serve

# Documentation targets
docs-install:
	$(UV) sync --all-groups

docs-build: docs-install
	$(UV_RUN) mkdocs build

docs-serve: docs-install
	$(UV_RUN) mkdocs serve

docs-clean:
	rm -rf site docs/.cache

docs-deploy: docs-build
	@echo "Documentation built and ready for deployment"
	@echo "Upload the 'site/' directory to your web server"

# Quick development setup
setup: dev-install
	@echo "Development environment ready!"
	@echo "Run 'make dev' to start the server"
