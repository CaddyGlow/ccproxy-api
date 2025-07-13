.PHONY: help install dev-install clean test test-unit test-real-api test-watch test-fast test-file test-match test-coverage lint typecheck format check pre-commit ci build dashboard-install dashboard-check dashboard-format dashboard-build dashboard-clean docker-build docker-run docs-install docs-build docs-serve docs-clean

$(eval VERSION_DOCKER := $(shell uv run python3 scripts/format_version.py docker))

# Common variables
UV_RUN := uv run

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  clean        - Clean build artifacts"
	@echo ""
	@echo "Testing commands (all include type checking and linting as prerequisites):"
	@echo "  test         - Run all tests with coverage (after quality checks)"
	@echo "  test-unit    - Run fast unit tests only (marked 'unit' or no 'real_api' marker)"
	@echo "  test-real-api - Run tests with real API calls (marked 'real_api', slow)"
	@echo "  test-watch   - Auto-run tests on file changes (with quality checks)"
	@echo "  test-fast    - Run tests without coverage (quick, after quality checks)"
	@echo "  test-coverage - Run tests with detailed coverage report"
	@echo ""
	@echo "Code quality:"
	@echo "  lint         - Run linting checks"
	@echo "  typecheck    - Run type checking"
	@echo "  format       - Format code"
	@echo "  check        - Run all checks (lint + typecheck)"
	@echo "  pre-commit   - Run pre-commit hooks (comprehensive checks + auto-fixes)"
	@echo "  ci           - Run full CI pipeline (pre-commit + test)"
	@echo ""
	@echo "Build and deployment:"
	@echo "  build        - Build Python package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo ""
	@echo "Dashboard (frontend):"
	@echo "  dashboard-install - Install dashboard dependencies"
	@echo "  dashboard-check   - Run dashboard quality checks (TypeScript + Biome)"
	@echo "  dashboard-format  - Format dashboard code"
	@echo "  dashboard-build   - Build dashboard for production"
	@echo "  dashboard-clean   - Clean dashboard build artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  docs-install - Install documentation dependencies"
	@echo "  docs-build   - Build documentation"
	@echo "  docs-serve   - Serve documentation locally"
	@echo "  docs-clean   - Clean documentation build files"

# Installation targets
install:
	uv sync --no-dev

dev-install:
	uv sync --all-extras --dev
	uv run pre-commit install
	@if command -v bun >/dev/null 2>&1; then \
		bun install -g @anthropic-ai/claude-code; \
		$(MAKE) dashboard-install; \
	else \
		echo "Warning: Bun not available, skipping Claude Code and dashboard installation"; \
	fi

# Cleanup
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .coverage
	rm -f coverage.xml
	rm -rf node_modules/
	rm -f pnpm-lock.yaml
	$(MAKE) dashboard-clean

# Testing targets (all enforce type checking and linting)
#
# All test commands include quality checks (mypy + ruff) as prerequisites to ensure
# tests only run on properly formatted and type-checked code. This prevents wasting
# time running tests on code that would fail CI anyway.
#
# Test markers:
#   - 'real_api': Tests that make actual API calls (slow, require network/auth)
#   - 'unit': Fast unit tests (< 1s each, no external dependencies)
#   - Tests without 'real_api' marker are considered unit tests by default

fix: format lint-fix
	ruff check . --unsafe-fixes

# Run all tests with coverage (after ensuring code quality)
test: check
	@echo "Running all tests with coverage..."
	@if [ ! -d "tests" ]; then echo "Error: tests/ directory not found. Create tests/ directory and add test files."; exit 1; fi
	$(UV_RUN) pytest tests/ -v --cov=ccproxy --cov-report=term-missing

# Run fast unit tests only (exclude tests marked with 'real_api')
test-unit: check
	@echo "Running fast unit tests (excluding real API calls)..."
	@if [ ! -d "tests" ]; then echo "Error: tests/ directory not found. Create tests/ directory and add test files."; exit 1; fi
	$(UV_RUN) pytest tests/ -v -m "not real_api" --tb=short

# Run tests with real API calls (marked with 'real_api')
test-real-api: check
	@echo "Running tests with real API calls (slow)..."
	@if [ ! -d "tests" ]; then echo "Error: tests/ directory not found. Create tests/ directory and add test files."; exit 1; fi
	$(UV_RUN) pytest tests/ -v -m "real_api" --tb=short

# Auto-run tests on file changes (requires entr or similar tool)
test-watch:
	@echo "Watching for file changes and running unit tests..."
	@echo "Note: Runs unit tests only (no real API calls) for faster feedback"
	@echo "Requires 'entr' tool: install with 'apt install entr' or 'brew install entr'"
	@echo "Use Ctrl+C to stop watching"
	@if command -v entr >/dev/null 2>&1; then \
		find ccproxy tests -name "*.py" | entr -c sh -c 'make check && $(UV_RUN) pytest tests/ -v -m "not real_api" --tb=short'; \
	else \
		echo "Error: 'entr' not found. Install with 'apt install entr' or 'brew install entr'"; \
		echo "Alternatively, use 'make test-unit' to run tests once"; \
		exit 1; \
	fi

# Quick test run (no coverage, but with quality checks)
test-fast: check
	@echo "Running fast tests without coverage..."
	@if [ ! -d "tests" ]; then echo "Error: tests/ directory not found. Create tests/ directory and add test files."; exit 1; fi
	$(UV_RUN) pytest tests/ -v --tb=short

# Run tests with detailed coverage report (HTML + terminal)
test-coverage: check
	@echo "Running tests with detailed coverage report..."
	@if [ ! -d "tests" ]; then echo "Error: tests/ directory not found. Create tests/ directory and add test files."; exit 1; fi
	$(UV_RUN) pytest tests/ -v --cov=ccproxy --cov-report=term-missing --cov-report=html
	@echo "HTML coverage report generated in htmlcov/"

# Run specific test file (with quality checks)
test-file: check
	@echo "Running specific test file: tests/$(FILE)"
	@if [ ! -d "tests" ]; then echo "Error: tests/ directory not found. Create tests/ directory and add test files."; exit 1; fi
	$(UV_RUN) pytest tests/$(FILE) -v

# Run tests matching a pattern (with quality checks)
test-match: check
	@echo "Running tests matching pattern: $(MATCH)"
	@if [ ! -d "tests" ]; then echo "Error: tests/ directory not found. Create tests/ directory and add test files."; exit 1; fi
	$(UV_RUN) pytest tests/ -k "$(MATCH)" -v

# Code quality
lint:
	uv run ruff check .

lint-fix: format
	uv run ruff check --fix .
	uv run ruff check --select I --fix .

typecheck:
	uv run mypy .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

# Combined checks (individual targets for granular control)
check: lint typecheck format-check

# Pre-commit hooks (comprehensive checks + auto-fixes)
pre-commit:
	uv run pre-commit run --all-files

# Full CI pipeline (comprehensive: pre-commit does more checks + auto-fixes)
ci:
	uv run pre-commit run --all-files
	$(MAKE) test

# Build targets
build:
	uv build

# Dashboard targets
dashboard-install:
	@echo "Installing dashboard dependencies..."
	@if [ ! -d "dashboard" ]; then echo "Error: dashboard/ directory not found."; exit 1; fi
	cd dashboard && bun install

dashboard-check:
	@echo "Running dashboard quality checks..."
	@if [ ! -d "dashboard" ]; then echo "Error: dashboard/ directory not found."; exit 1; fi
	cd dashboard && bun run check

dashboard-format:
	@echo "Formatting dashboard code..."
	@if [ ! -d "dashboard" ]; then echo "Error: dashboard/ directory not found."; exit 1; fi
	cd dashboard && bun run format

dashboard-build: dashboard-check
	@echo "Building dashboard for production..."
	@if [ ! -d "dashboard" ]; then echo "Error: dashboard/ directory not found."; exit 1; fi
	cd dashboard && bun run build:prod
	@echo "✅ Dashboard built and copied to ccproxy/static/"

dashboard-clean:
	@echo "Cleaning dashboard build artifacts..."
	@if [ -d "dashboard/build" ]; then rm -rf dashboard/build; fi
	@if [ -d "dashboard/node_modules/.cache" ]; then rm -rf dashboard/node_modules/.cache; fi
	@if [ -d "ccproxy/static/dashboard" ]; then rm -rf ccproxy/static/dashboard; fi

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
	uv run fastapi dev ccproxy/main.py

# Documentation targets
docs-install:
	uv sync --group docs

docs-build: docs-install
	./scripts/build-docs.sh

docs-serve: docs-install
	./scripts/serve-docs.sh

docs-clean:
	rm -rf site/
	rm -rf docs/.cache/

docs-deploy: docs-build
	@echo "Documentation built and ready for deployment"
	@echo "Upload the 'site/' directory to your web server"

# Quick development setup
setup: dev-install
	@echo "Development environment ready!"
	@echo "Run 'make dev' to start the server"
