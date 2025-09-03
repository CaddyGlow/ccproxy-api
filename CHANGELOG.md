# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2025-08-13

### Added

- **Enhanced Authentication Validation**: Implemented separate authentication validation for both Claude and Codex providers at startup
  - Added `validate_claude_authentication_startup()` function with provider-specific logging messages
  - Added `validate_codex_authentication_startup()` function for OpenAI/Codex token validation
  - Updated startup lifecycle to validate both authentication systems independently
  - Clear differentiation between Claude and Codex authentication status in logs
- **Package Data Fallback System**: Implemented robust fallback data system using package-embedded JSON files
  - Added `ccproxy/data/claude_headers_fallback.json` with Claude CLI headers and system prompts
  - Added `ccproxy/data/codex_headers_fallback.json` with Codex CLI headers and detection data
  - Updated `pyproject.toml` to include data files in package builds via `[tool.hatch.build.targets.wheel]`
  - Enhanced detection services to load fallback data from package files instead of hardcoded values
  - Removed identifying UUIDs from fallback files for privacy protection
- **Version Update Checking System**: Comprehensive version monitoring with scheduled checks and caching
  - Automatic version checking against GitHub releases with configurable intervals (default 12 hours)
  - Startup version checks with graceful degradation when GitHub is unavailable
  - Intelligent caching system to avoid rate limiting and reduce network requests
  - Integration with scheduler for periodic background version monitoring
  - Detailed logging of version check results, cache status, and update availability

### Fixed

- **Authentication Error Logging**: Significantly improved error logging and reduced log noise for authentication failures
  - HTTP 401 authentication errors now log as warnings instead of errors with stack traces
  - Removed exception chaining (`from e`) for authentication failures to eliminate verbose stack traces
  - Updated Codex route exception handling to use `from None` suppression for cleaner logs
  - Enhanced HTTP exception middleware to treat 401 errors as expected events like 404s
  - Updated metrics recording to properly categorize HTTP 401 errors as "http_401"
- **Codex CLI Detection**: Improved Codex header detection reliability and startup behavior
  - Added CLI availability check before attempting Codex header detection to prevent warnings
  - Enhanced `initialize_codex_detection_startup()` to skip detection when Codex CLI is unavailable
  - Improved error handling and fallback behavior when Codex is not installed
  - Better separation between Codex enabled/disabled states and CLI availability
- **Codex Streaming Response Handling**: Enhanced streaming behavior and format conversion for Codex endpoints
  - Fixed streaming response management in `codex_chat_completions` endpoint
  - Improved Response API to Chat Completions format conversion for both streaming and non-streaming requests
  - Enhanced error handling for malformed responses and backend communication failures
  - Better stream chunk processing with proper reasoning block handling and content type detection
  - Improved non-streaming response collection and conversion from streaming backends

### Changed

- **Startup Lifecycle Architecture**: Enhanced component-based startup system with better error handling
  - Updated `LIFECYCLE_COMPONENTS` to include separate "Claude Authentication" and "Codex Authentication" components
  - Improved startup component execution with graceful degradation when individual components fail
  - Enhanced logging for component startup and shutdown phases with clear component identification
  - Better separation of concerns between different startup validation tasks
- **Exception Handling Patterns**: Standardized exception handling across Codex routes
  - Applied consistent `from None` pattern for all authentication and proxy error handling
  - Improved error message clarity and reduced stack trace pollution
  - Enhanced logging levels appropriate for different error types (debug for auth failures, error for unexpected issues)
- **Detection Service Architecture**: Improved fallback data loading and service initialization
  - Enhanced `ClaudeDetectionService` and `CodexDetectionService` to use package data files
  - Better error handling when package data files are missing or corrupted
  - Improved service initialization with graceful fallback to embedded data

### Infrastructure

- **Package Data Management**: Added proper package data inclusion for distribution
  - Updated build configuration to include `ccproxy/data/*.json` files in wheel packages
  - Ensured fallback data is available in both development and packaged installations
  - Improved data file loading with proper path resolution for packaged deployments
- **Code Quality**: Enhanced code organization and linting compliance
  - Fixed all ruff linting issues with proper exception chaining patterns
  - Improved type safety and eliminated linting warnings
  - Enhanced code consistency across authentication and error handling modules

This release focuses on improving the reliability and maintainability of the authentication system while providing better user experience through cleaner logs and more robust startup behavior. The version checking system ensures users stay informed about updates, while the improved Codex integration provides a more stable experience when using OpenAI providers.

## [0.1.6] - 2025-08-13

## Added OpenAI Codex Provider with Full Proxy Support

### Overview

Implemented comprehensive support for OpenAI Codex CLI integration, enabling users to proxy requests through their OpenAI subscription via the ChatGPT backend API. This feature provides an alternative to the Claude provider while maintaining full compatibility with the existing proxy architecture. The implementation uses the OpenAI Responses API endpoint as documented at https://platform.openai.com/docs/api-reference/responses/get.

### Key Features

**Complete Codex API Proxy**

- Full reverse proxy to `https://chatgpt.com/backend-api/codex`
- Support for both `/codex/responses` and `/codex/{session_id}/responses` endpoints
- Compatible with Codex CLI 0.21.0 and authentication flow
- Implements OpenAI Responses API protocol

**OAuth PKCE Authentication Flow**

- Implements complete OpenAI OAuth 2.0 PKCE flow matching official Codex CLI
- Local callback server on port 1455 for authorization code exchange
- Token refresh and credential management with persistent storage
- Support for `~/.codex/auth.json` configuration file format

**Intelligent Request/Response Handling**

- Automatic detection and injection of Codex CLI instructions field
- Smart streaming behavior based on user's explicit `stream` parameter
- Session management with flexible session ID handling (auto-generated, persistent, header-forwarded)
- Request transformation preserving Codex CLI identity headers

**Advanced Configuration**

- Environment variable support: `CODEX__BASE_URL`
- Configurable via TOML: `[codex]` section in configuration files
- Debug logging with request/response capture capabilities
- Comprehensive error handling with proper HTTP status codes
- Enabled by default

### Technical Implementation

**New Components Added:**

- `ccproxy/auth/openai.py` - OAuth token management and credential storage
- `ccproxy/core/codex_transformers.py` - Request/response transformation for Codex format
- `ccproxy/api/routes/codex.py` - FastAPI routes for Codex endpoints
- `ccproxy/models/detection.py` - Codex CLI detection and header management
- `ccproxy/services/codex_detection_service.py` - Runtime detection of Codex CLI requests

**Enhanced Proxy Service:**

- Extended `ProxyService.handle_codex_request()` with full Codex support
- Intelligent streaming response conversion when user doesn't explicitly request streaming
- Comprehensive request/response logging for debugging
- Error handling with proper OpenAI-compatible error responses

### Streaming Behavior Fix

**Problem Resolved:** Fixed issue where requests without explicit `stream` field were incorrectly returning streaming responses.

**Solution Implemented:**

- When `"stream"` field is missing: Inject `"stream": true` for upstream (Codex requirement) but return JSON response to client
- When `"stream": true` explicitly set: Return streaming response to client
- When `"stream": false` explicitly set: Return JSON response to client
- Smart response conversion: collects streaming data and converts to single JSON response when user didn't request streaming

### Usage Examples

**Basic Request (JSON Response):**

```bash
curl -X POST "http://127.0.0.1:8000/codex/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
    "model": "gpt-5",
    "store": false
  }'
```

**Streaming Request:**

```bash
curl -X POST "http://127.0.0.1:8000/codex/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
    "model": "gpt-5",
    "stream": true,
    "store": false
  }'
```

### Authentication Setup

**Environment Variables:**

```bash
export CODEX__BASE_URL="https://chatgpt.com/backend-api/codex"
```

**Configuration File (`~/.ccproxy.toml`):**

```toml
[codex]
base_url = "https://chatgpt.com/backend-api/codex"
```

### Compatibility

- Codex CLI: Full compatibility with `codex-cli 0.21.0`
- OpenAI OAuth: Complete PKCE flow implementation
- Session Management: Supports persistent and auto-generated sessions
- Model Support: All Codex-supported models (`gpt-5`, `gpt-4`, etc.)
- Streaming: Both streaming and non-streaming responses
- Error Handling: Proper HTTP status codes and OpenAI-compatible errors
- API Compliance: Follows OpenAI Responses API specification

### Files Modified/Added

**New Files:**

- `ccproxy/auth/openai.py` - OpenAI authentication management
- `ccproxy/core/codex_transformers.py` - Codex request/response transformation
- `ccproxy/api/routes/codex.py` - Codex API endpoints
- `ccproxy/models/detection.py` - Codex detection models
- `ccproxy/services/codex_detection_service.py` - Codex CLI detection service

**Modified Files:**

- `ccproxy/services/proxy_service.py` - Added `handle_codex_request()` method
- `ccproxy/config/settings.py` - Added Codex configuration section
- `ccproxy/api/app.py` - Integrated Codex routes
- `ccproxy/api/routes/health.py` - Added Codex health checks

### Breaking Changes

None. This is a purely additive feature that doesn't affect existing Claude provider functionality.

### Migration Notes

For users wanting to use Codex provider:

1. Authenticate: Use existing OpenAI credentials or run Codex CLI login
2. Update endpoints: Change from `/v1/messages` to `/codex/responses`

This implementation provides a complete, production-ready OpenAI Codex proxy solution that maintains the same standards as the existing Claude provider while offering users choice in their AI provider preferences.

## Added OpenAI Codex Provider with Full Proxy Support

### Overview

Implemented comprehensive support for OpenAI Codex CLI integration, enabling users to proxy requests through their OpenAI subscription via the ChatGPT backend API. This feature provides an alternative to the Claude provider while maintaining full compatibility with the existing proxy architecture. The implementation uses the OpenAI Responses API endpoint as documented at https://platform.openai.com/docs/api-reference/responses/get.

### Key Features

**Complete Codex API Proxy**

- Full reverse proxy to `https://chatgpt.com/backend-api/codex`
- Support for both `/codex/responses` and `/codex/{session_id}/responses` endpoints
- Compatible with Codex CLI 0.21.0 and authentication flow
- Implements OpenAI Responses API protocol

**OAuth PKCE Authentication Flow**

- Implements complete OpenAI OAuth 2.0 PKCE flow matching official Codex CLI
- Local callback server on port 1455 for authorization code exchange
- Token refresh and credential management with persistent storage
- Support for `~/.codex/auth.json` configuration file format

**Intelligent Request/Response Handling**

- Automatic detection and injection of Codex CLI instructions field
- Smart streaming behavior based on user's explicit `stream` parameter
- Session management with flexible session ID handling (auto-generated, persistent, header-forwarded)
- Request transformation preserving Codex CLI identity headers

**Advanced Configuration**

- Environment variable support: `CODEX__BASE_URL`
- Configurable via TOML: `[codex]` section in configuration files
- Debug logging with request/response capture capabilities
- Comprehensive error handling with proper HTTP status codes
- Enabled by default

### Technical Implementation

**New Components Added:**

- `ccproxy/auth/openai.py` - OAuth token management and credential storage
- `ccproxy/core/codex_transformers.py` - Request/response transformation for Codex format
- `ccproxy/api/routes/codex.py` - FastAPI routes for Codex endpoints
- `ccproxy/models/detection.py` - Codex CLI detection and header management
- `ccproxy/services/codex_detection_service.py` - Runtime detection of Codex CLI requests

**Enhanced Proxy Service:**

- Extended `ProxyService.handle_codex_request()` with full Codex support
- Intelligent streaming response conversion when user doesn't explicitly request streaming
- Comprehensive request/response logging for debugging
- Error handling with proper OpenAI-compatible error responses

### Streaming Behavior Fix

**Problem Resolved:** Fixed issue where requests without explicit `stream` field were incorrectly returning streaming responses.

**Solution Implemented:**

- When `"stream"` field is missing: Inject `"stream": true` for upstream (Codex requirement) but return JSON response to client
- When `"stream": true` explicitly set: Return streaming response to client
- When `"stream": false` explicitly set: Return JSON response to client
- Smart response conversion: collects streaming data and converts to single JSON response when user didn't request streaming

### Usage Examples

**Basic Request (JSON Response):**

```bash
curl -X POST "http://127.0.0.1:8000/codex/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
    "model": "gpt-5",
    "store": false
  }'
```

**Streaming Request:**

```bash
curl -X POST "http://127.0.0.1:8000/codex/responses" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
    "model": "gpt-5",
    "stream": true,
    "store": false
  }'
```

### Authentication Setup

**Environment Variables:**

```bash
export CODEX__BASE_URL="https://chatgpt.com/backend-api/codex"
```

**Configuration File (`~/.ccproxy.toml`):**

```toml
[codex]
base_url = "https://chatgpt.com/backend-api/codex"
```

### Compatibility

- Codex CLI: Full compatibility with `codex-cli 0.21.0`
- OpenAI OAuth: Complete PKCE flow implementation
- Session Management: Supports persistent and auto-generated sessions
- Model Support: All Codex-supported models (`gpt-5`, `gpt-4`, etc.)
- Streaming: Both streaming and non-streaming responses
- Error Handling: Proper HTTP status codes and OpenAI-compatible errors
- API Compliance: Follows OpenAI Responses API specification

### Files Modified/Added

**New Files:**

- `ccproxy/auth/openai.py` - OpenAI authentication management
- `ccproxy/core/codex_transformers.py` - Codex request/response transformation
- `ccproxy/api/routes/codex.py` - Codex API endpoints
- `ccproxy/models/detection.py` - Codex detection models
- `ccproxy/services/codex_detection_service.py` - Codex CLI detection service

**Modified Files:**

- `ccproxy/services/proxy_service.py` - Added `handle_codex_request()` method
- `ccproxy/config/settings.py` - Added Codex configuration section
- `ccproxy/api/app.py` - Integrated Codex routes
- `ccproxy/api/routes/health.py` - Added Codex health checks

### Breaking Changes

None. This is a purely additive feature that doesn't affect existing Claude provider functionality.

### Migration Notes

For users wanting to use Codex provider:

1. Authenticate: Use existing OpenAI credentials or run Codex CLI login
2. Update endpoints: Change from `/v1/messages` to `/codex/responses`

This implementation provides a complete, production-ready OpenAI Codex proxy solution that maintains the same standards as the existing Claude provider while offering users choice in their AI provider preferences.

## [0.1.5] - 2025-08-03

### Added

- **Advanced Session and Pool Management**:
  - Implemented a robust session-aware pool for persistent Claude SDK connections, significantly improving performance and maintaining conversation continuity.
  - Introduced a hybrid pooling system that automatically transfers clients from a general pool to the session pool upon receiving a session ID.
  - Developed a queue-based streaming architecture to efficiently handle and broadcast messages to multiple listeners, improving session management and disconnection handling.
- **Enhanced Observability and Logging**:
  - Upgraded logging capabilities to include detailed session metadata, providing deeper insights into session lifecycle and reuse.
  - Implemented a pool monitoring system to track the health and performance of both general and session-based connection pools.
  - Reduced logging noise by adjusting log levels for operational SDK messages, focusing on essential access logs, warnings, and errors.
- **Improved Configuration and Control**:
  - Introduced a `builtin_permissions` flag to provide granular control over the built-in permission handling infrastructure (MCP and SSE).
  - Implemented configurable system prompt injection modes (`minimal` and `full`) to customize how the Claude Code identity is presented in requests.
- **Robust Streaming and Header Management**:
  - Implemented `StreamingResponseWithLogging` for unified and consistent access logging across all streaming endpoints.
  - Ensured critical upstream headers (e.g., `cf-ray`, `anthropic-ratelimit-*`) are correctly forwarded in SSE streaming responses.

### Changed

- **Default Behavior**:
  - The Claude SDK connection pool is now disabled by default, requiring an explicit opt-in for safer and more predictable behavior.
- **Architectural Improvements**:
  - Refactored the application's startup and shutdown logic into a modular, component-based architecture for better maintainability and testability.
  - Renamed `SessionContext` to `SessionClient` for improved clarity and consistency in the session pooling implementation.
- **Testing Infrastructure**:
  - Reorganized the entire test suite into a hierarchical structure (`unit` and `integration`) to improve navigation and maintainability.
  - Migrated from legacy test fixtures to a more flexible and maintainable factory pattern for creating test clients and application instances.

### Fixed

- **Session and Streaming Stability**:
  - Eliminated critical race conditions and `AttributeError` exceptions in the session pool and stream handling logic.
  - Replaced fragile `asyncio.sleep` calls with a robust, event-based synchronization mechanism to prevent timing-related failures.
  - Implemented a more accurate message-based stale detection mechanism to prevent the incorrect termination of active sessions.
- **Resource Management**:
  - Corrected several resource leak issues by improving stream interruption handling, ensuring that hanging sessions are properly cleaned up.
- **Header and Content Formatting**:
  - Resolved an issue that prevented the forwarding of upstream headers in SSE streaming responses.
  - Fixed a formatting bug in the OpenAI adapter that caused message content to be improperly concatenated.

### Added

- **Configurable Permission Infrastructure**: Added `builtin_permissions` configuration flag to control built-in permission handling infrastructure:
  - New `builtin_permissions` flag (default: true) in ClaudeSettings for granular control
  - CLI support with `--builtin-permissions/--no-builtin-permissions` options
  - When disabled: No MCP server setup, no SSE permission endpoints, no permission service initialization
  - When enabled: Full built-in permission infrastructure with smart MCP server merging
  - Users can configure custom MCP servers independently when built-in infrastructure is disabled
  - Maintains full backward compatibility with existing configurations
- **Claude SDK Pool Mode**: Implemented connection pooling for Claude Code SDK clients to improve request performance:
  - Maintains a pool of pre-initialized Claude Code instances to eliminate startup overhead
  - Reduces request latency by reusing established connections
  - Pool mode is disabled by default and can be enabled via configuration
  - **Limitations**: Pool mode does not support dynamic Claude options (max_tokens, model changes, etc.)
  - Pool instances are shared across requests with identical configurations
- **Session-Aware Connection Pooling**: Added advanced session-based pooling for persistent conversation context:
  - Session pools maintain dedicated Claude SDK clients per session ID for conversation continuity
  - Configurable session TTL (time-to-live) with automatic cleanup of idle sessions
  - Session pool settings include max sessions, idle threshold, and cleanup intervals
  - Automatic connection recovery for unhealthy sessions when enabled
  - Session interruption support for graceful handling of canceled requests
  - Separate from the general connection pool - can be used independently or together
  - Configuration via `claude.session_pool` settings with sensible defaults
- **Claude Detection Service**: Implemented automatic Claude CLI header and system prompt detection at startup:
  - Automatically detects current Claude CLI version and extracts real headers/system prompt
  - Caches detection results per version to avoid repeated startup delays
  - Falls back to hardcoded values when detection fails
- **Detection Models**: Added Pydantic models for Claude detection data:
  - `ClaudeCodeHeaders` - Structured header extraction with field aliases
  - `SystemPromptData` - System prompt content with cache control
  - `ClaudeCacheData` - Complete cached detection data with version tracking

### Changed

- **Configuration Updates**: Enhanced Claude settings with new pool configuration options:
  - Added `use_client_pool` boolean flag to enable general connection pooling
  - Added `pool_settings` for configuring general pool behavior (size, timeouts, health checks)
  - Added `session_pool` settings for session-aware pooling configuration
  - Session pool enabled by default with 1-hour TTL and automatic cleanup
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
- **Session Pool Components**: Added new modules for session management:
  - `ccproxy/claude_sdk/session_pool.py` - Core session pool implementation
  - `ccproxy/claude_sdk/session_client.py` - Session-aware client wrapper
  - `ccproxy/claude_sdk/manager.py` - Unified pool management with metrics integration
- **Test Coverage**: Added comprehensive tests for session pool functionality:
  - Unit tests for session lifecycle, cleanup, and recovery
  - Integration tests for end-to-end session pooling behavior

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
<!-- Keyring integration was planned but has been removed; credentials are stored via project-managed files. -->
- **`generate-token` Command**: A CLI command to manually generate and manage API tokens.
- **Systemd Integration**: Includes a setup script and service template for running the proxy as a systemd service in production environments.
- **Docker Support**: A `Dockerfile` and `docker-compose.yml` for running the proxy in an isolated containerized environment.

#### Security

- **Local-First Design**: All processing and authentication happens locally; no conversation data is stored or transmitted to third parties.
<!-- Credential storage now uses project-managed files; system keyring is not used. -->
- **Header Stripping**: Automatically removes client-side `Authorization` headers to prevent accidental key leakage.

#### Developer Experience

- **Comprehensive Documentation**: Includes a quick start guide, API reference, and setup instructions.
- **Pre-commit Hooks**: Configured for automated code formatting and linting to ensure code quality.
- **Modern Tooling**: Uses `uv` for package management and `devenv` for a reproducible development environment.
- **Extensive Test Suite**: Includes unit, integration, and benchmark tests to ensure reliability.
- **Rich Logging**: Structured and colorized logging for improved readability during development and debugging.
## [Unreleased]

### Removed

- Dead code: removed `ccproxy/utils/models_provider.py` (unreferenced; model listing is provided by plugins).
- Pruned root runtime dependencies no longer used directly by core:
  - `aiosqlite` (unused in repo)
  - `h2` (no direct imports; `httpx[http2]` brings HTTP/2 support transitively)

### Notes

- Plugin-owned dependencies remain in root for now (plugins are bundled): `duckdb`, `duckdb-engine`, `sqlmodel`, `prometheus-client`, `textual`. These may move to plugin-specific distributions or optional extras in a future split.
## [0.2.0] - 2025-09-02

### Changed

- Core health endpoints simplified and plugin-agnostic; provider/OAuth/SDK checks moved to plugin health under `/plugins/{name}/health`.
- Plugins CLI now uses centralized `load_plugin_system()`; discovery logic consolidated.
- Documentation updated to reflect plugin-first architecture and loader flow.

### Removed

- Legacy plugin management endpoints: `POST /plugins/{name}/reload`, `POST /plugins/discover`, `DELETE /plugins/{name}` (v2 loads at startup; restart to apply changes).
- Scheduler references to Pushgateway in core; metrics plugin fully owns push task registration.
- Core middleware reliance on `app.state.duckdb_storage` alias; storage wiring is plugin-owned.

### Added

- Configuration validation that fails fast when deprecated keys are present, with guidance to the corresponding `plugins.*` keys.
- Migration guide: `docs/migration/0.2-plugin-first.md`.

This release completes the plugin-first migration and removes transitional shims.
