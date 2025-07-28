# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Claude Detection Service**: Implemented automatic Claude CLI header and system prompt detection at startup:
  - Automatically detects current Claude CLI version and extracts real headers/system prompt
  - Caches detection results per version to avoid repeated startup delays
  - Falls back to hardcoded values when detection fails
- **Detection Models**: Added Pydantic models for Claude detection data:
  - `ClaudeCodeHeaders` - Structured header extraction with field aliases
  - `SystemPromptData` - System prompt content with cache control
  - `ClaudeCacheData` - Complete cached detection data with version tracking

### Changed

- **HTTP Request Transformation**: Enhanced request transformers to use detected Claude CLI headers and system prompt:
  - Dynamically uses detected headers when available, falls back to hardcoded when not
  - System prompt injection now uses detected Claude Code system prompt
  - Added app_state parameter propagation for detection data access
- **Request Transformer Architecture**: Refactored transformers to support dynamic header and prompt injection:
  - Added proxy_mode parameter to RequestTransformer base class
  - Enhanced transform methods to accept app_state for detection data access
  - Improved header creation logic to distinguish between detected and fallback headers
- **Test Organization Cleanup**: Finalized test suite migration and removed obsolete migration documentation:
  - Removed obsolete `MIGRATION_GUIDE.md` files from factories, fixtures, and auth directories
  - Cleaned up `conftest.py` by removing backward compatibility aliases for fixture names
  - Updated fixture references to use direct imports from fixture modules
  - Simplified FastAPI factory test organization by removing legacy compatibility layer
  - Modernized fixture naming convention throughout test files (internal_claude_sdk_service vs claude_service)
  - Removed unused factory fixtures and consolidated client creation patterns
- **Test Organization**: Migrated test suite from flat structure to organized hierarchy:
  - Tests now organized under `tests/unit/` and `tests/integration/` directories
  - Unit tests categorized by component: `api/`, `services/`, `auth/`, `config/`, `utils/`, `cli/`
  - Integration tests moved to dedicated `tests/integration/` directory
  - Enhanced factory pattern with `FastAPIAppFactory` for flexible test app creation
  - Improved fixture organization with dedicated `tests/fixtures/` structure
- **Configuration Cleanup**: Removed unused `ConfigLoader` class and simplified configuration management
- **Logging Optimization**: Reduced permission service log verbosity from INFO to DEBUG level for cleaner production logs

### Infrastructure

- **Test Structure**: Added `.gitignore` for test artifacts and coverage reports
- **Documentation**: Updated `TESTING.md` with new test organization and examples
- **Cache Directory**: Added automatic creation of `~/.cache/ccproxy/` for detection data persistence

## [0.1.4] - 2025-05-28

### Fixed

- **Pydantic Compatibility**: Fixed TypeError in model_dump_json() call by removing invalid separators parameter (issue #5)

## [0.1.3] - 2025-07-25

### Added

- **Version Update Checking**: Automatic version checking against GitHub releases with configurable intervals (default 12 hours) and startup checks
- **MCP Server Integration**: Added Model Context Protocol (MCP) server functionality with permission checking tools for Claude Code integration
- **Permission System**: Implemented comprehensive permission management with REST API endpoints and Server-Sent Events (SSE) streaming for real-time permission requests
- **Request/Response Logging**: Added comprehensive logging middleware with configurable verbosity levels (`CCPROXY_VERBOSE_API`, `CCPROXY_REQUEST_LOG_DIR`)
- **Claude SDK Custom Content Blocks**: Added support for `system_message`, `tool_use_sdk`, and `tool_result_sdk` content blocks with full metadata preservation
- **Model Mapping Utilities**: Centralized model provider abstraction with unified mapping logic in `ccproxy/utils/models_provider.py`
- **Terminal Permission Handler**: Interactive permission workflow handler for CLI-based permission management
- **Claude SDK Field Rendering**: Added flexible content handling with `forward`, `formatted`, and `ignore` rendering options for Claude SDK fields

### Changed

- **Claude SDK Integration**: Refactored to use native ThinkingBlock models from Claude Code SDK
- **Models Endpoint**: Centralized `/v1/models` endpoint implementation to eliminate code duplication across routes
- **OpenAI Adapter**: Enhanced with improved modularization and streaming architecture
- **Logging System**: Migrated to canonical structlog pattern for structured, consistent logging
- **SSE Streaming**: Improved Server-Sent Events format with comprehensive examples and better error handling

### Fixed

- **SDK Double Content**: Resolved duplicate content issue in Claude SDK message processing
- **Error Handling**: Enhanced error handling throughout Claude SDK message processing pipeline
- **Type Safety**: Improved type checking across permission system components
- **Permission Handler**: Fixed lazy initialization issues in terminal permission handler

## [0.1.2] - 2025-07-22

### Added

- **Permission Mode Support**: Restored `--permission-mode` option supporting default, acceptEdits, and bypassPermissions modes
- **Custom Permission Tool**: Restored `--permission-prompt-tool-name` option to specify custom permission tool names
- **Permission Response Models**: Added `PermissionToolAllowResponse` and `PermissionToolDenyResponse` models with proper JSON serialization

### Changed

- **Message Formatting**: Modified `MessageConverter.combine_text_parts()` to join text parts with newlines instead of spaces, preserving formatting in multi-line content
- **Settings Integration**: Enhanced OptionsHandler to apply default Claude options from settings before API parameters
- **Configuration**: Extended settings to persist permission_mode and permission_prompt_tool_name

### Fixed

- **Claude SDK Options**: Integrated Settings dependency into ClaudeSDKService to support configuration-based options

## [0.1.1] - 2025-07-22

### Added

- **Conditional Authentication**: API endpoints now support optional authentication - when `SECURITY__AUTH_TOKEN` is configured, authentication is enforced; when not configured, the proxy runs in open mode.
- **Startup Validation**: Added comprehensive validation checks during application startup:
  - Validates OAuth credentials and warns about expired tokens
  - Checks for Claude CLI binary availability with installation instructions
  - Logs token expiration time and subscription type when valid
- **Default Command**: The `serve` command is now the default - running `ccproxy` without subcommands automatically starts the server.
- **Alternative Entry Point**: Added `ccproxy-api` as an alternative command-line entry point.

### Changed

- **Authentication Variable**: Renamed environment variable from `AUTH_TOKEN` to `SECURITY__AUTH_TOKEN` for better namespace organization and clarity.
- **Credential Priority**: Reordered credential search paths to prioritize ccproxy-specific credentials before Claude CLI paths.
- **CLI Syntax**: Migrated all CLI parameters to modern Annotated syntax for better type safety and IDE support.
- **Pydantic v2**: Updated all models to use Pydantic v2 configuration syntax (`model_config` instead of `Config` class).
- **Documentation**: Improved Aider integration docs with correct API endpoint URLs and added installation options (uv, pipx).

### Fixed

- **Authentication Separation**: Fixed critical issue where auth token was incorrectly used for both client and upstream authentication - now client auth token is separate from OAuth credentials.
- **URL Paths**: Fixed documentation to use `/api` endpoints for Aider compatibility instead of SDK mode paths.
- **Default Values**: Fixed default values for list parameters in CLI (docker_env, docker_volume, docker_arg).

### Removed

- **Status Endpoints**: Removed redundant `/status` endpoints from both Claude SDK and proxy routes.
- **Permission Tool**: Removed Claude permission tool functionality and related CLI options (`--permission-mode`, `--permission-prompt-tool-name`) that are no longer needed.
- **Deprecated Options**: Removed references to deprecated permission_mode and permission_prompt_tool_name from documentation.

## [0.1.0] - 2025-07-21

This is the initial public release of the CCProxy API.

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
