.PHONY: help setup install dev-install clean test lint typecheck format check pre-commit ci build docker-build docker-run docs-install docs-build docs-serve docs-clean

$(eval VERSION_DOCKER := $(shell uv run python3 scripts/format_version.py docker))

# Default target
help:
	@echo "Available targets:"
	@echo "  setup        - Full setup including Claude CLI check/install"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  typecheck    - Run type checking"
	@echo "  format       - Format code"
	@echo "  check        - Run all checks (lint + typecheck)"
	@echo "  pre-commit   - Run pre-commit hooks (comprehensive checks + auto-fixes)"
	@echo "  ci           - Run full CI pipeline (pre-commit + test)"
	@echo "  build        - Build Python package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docs-install - Install documentation dependencies"
	@echo "  docs-build   - Build documentation"
	@echo "  docs-serve   - Serve documentation locally"
	@echo "  docs-clean   - Clean documentation build files"

# Installation targets
setup:
	@bash scripts/setup.sh

install:
	uv sync --no-dev
	pnpm install --prod

dev-install:
	uv sync --all-extras --dev
	pnpm install
	uv run pre-commit install

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

# Testing
test:
	uv run pytest -v --cov=ccproxy --cov-report=xml --cov-report=term-missing

test-unit:
	uv run pytest -v -m unit

test-integration:
	uv run pytest -v -m integration

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
