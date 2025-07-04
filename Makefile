.PHONY: help install dev-install clean test lint typecheck format check ci build docker-build docker-run docs-install docs-build docs-serve docs-clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  typecheck    - Run type checking"
	@echo "  format       - Format code"
	@echo "  check        - Run all checks (lint + typecheck)"
	@echo "  ci           - Run full CI pipeline (check + test)"
	@echo "  build        - Build Python package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docs-install - Install documentation dependencies"
	@echo "  docs-build   - Build documentation"
	@echo "  docs-serve   - Serve documentation locally"
	@echo "  docs-clean   - Clean documentation build files"

# Installation targets
install:
	uv sync --no-dev

dev-install:
	uv sync --all-extras --dev

# Cleanup
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f .coverage
	rm -f coverage.xml

# Testing
test:
	uv run pytest -v --cov=claude_code_proxy --cov-report=xml --cov-report=term-missing

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

# Combined checks
check: lint typecheck format-check

# Full CI pipeline
ci: check test

# Build targets
build:
	uv build

# Docker targets
docker-build:
	docker build -t claude-code-proxy .

docker-run:
	docker run --rm -p 8000:8000 claude-code-proxy

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down

# Development server
dev:
	uv run fastapi dev claude_code_proxy/main.py

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
