# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-21

This is the initial public release of the Claude Code Proxy API.

### Added

#### Core Functionality
- **Personal Claude Access**: Enables using a personal Claude Pro, Team, or Enterprise subscription as an API endpoint, without needing separate API keys.
- **OAuth2 Authentication**: Implements the official Claude OAuth2 flow for secure, local authentication.
- **Local Proxy Server**: Runs a lightweight FastAPI server on your local machine.
- **HTTP/HTTPS Proxy Support**: Full support for routing requests through an upstream HTTP or HTTPS proxy.

#### API & Compatibility
- **Dual API Support**: Provides full compatibility with both Anthropic and OpenAI API specifications.
- **Anthropic Messages API**: Native support for the Anthropic Messages API at `/v1/chat/completions`.
- **OpenAI Chat Completions API**: Compatibility layer for the OpenAI Chat Completions API at `/openai/v1/chat/completions`.
- **Request/Response Translation**: Seamlessly translates requests and responses between OpenAI and Anthropic formats.
- **Streaming Support**: Real-time streaming for both Anthropic and OpenAI-compatible endpoints.
- **Model Endpoints**: Lists available models via `/v1/models` and `/openai/v1/models`.
- **Health Check**: A `/health` endpoint for monitoring the proxy's status.

#### Configuration & CLI
- **Unified `ccproxy` CLI**: A single, user-friendly command-line interface for managing the proxy.
- **TOML Configuration**: Configure the server using a `config.toml` file with JSON Schema validation.
- **Keyring Integration**: Securely stores and manages OAuth credentials in the system's native keyring.
- **`generate-token` Command**: A CLI command to manually generate and manage API tokens.
- **Systemd Integration**: Includes a setup script and service template for running the proxy as a systemd service in production environments.
- **Docker Support**: A `Dockerfile` and `docker-compose.yml` for running the proxy in an isolated containerized environment.

#### Security
- **Local-First Design**: All processing and authentication happens locally; no conversation data is stored or transmitted to third parties.
- **Credential Security**: OAuth tokens are stored securely in the system keyring, not in plaintext files.
- **Header Stripping**: Automatically removes client-side `Authorization` headers to prevent accidental key leakage.

#### Developer Experience
- **Comprehensive Documentation**: Includes a quick start guide, API reference, and setup instructions.
- **Pre-commit Hooks**: Configured for automated code formatting and linting to ensure code quality.
- **Modern Tooling**: Uses `uv` for package management and `devenv` for a reproducible development environment.
- **Extensive Test Suite**: Includes unit, integration, and benchmark tests to ensure reliability.
- **Rich Logging**: Structured and colorized logging for improved readability during development and debugging.
