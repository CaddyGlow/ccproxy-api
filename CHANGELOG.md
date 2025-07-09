# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-08

### Recent Updates

#### Code Organization & Architecture
- **Domain-Based Code Organization**: Reorganized code by domain with modern type annotations for better maintainability
- **Streaming Utilities Refactoring**: Separated streaming utilities by format with unified transformer architecture
- **OpenAI Compatibility Modernization**: Enhanced OpenAI compatibility with thinking blocks and improved type safety
- **Anthropic Messages API Alignment**: Aligned Messages API with official Anthropic specification for better compatibility

#### Enhanced Features
- **Thinking Blocks Support**: Added support for OpenAI thinking blocks in API responses
- **Systemd Integration**: Enhanced OpenAI compatibility with systemd setup for production deployments
- **Proxy Support**: Added comprehensive HTTP/HTTPS proxy support for network requests
- **Keyring Security**: Implemented secure credential storage using system keyring for enhanced security

#### Package & Documentation
- **Package Rename**: Renamed package from `claude_code_proxy` to `ccproxy` for better naming consistency
- **Documentation Consolidation**: Consolidated and streamlined documentation with dual API access modes
- **Quick Start Improvements**: Streamlined Quick Start guide with concise examples and Aider integration
- **Docker Testing**: Updated Docker image name assertions in test suite

### Added

#### Configuration & CLI Enhancements
- **TOML Configuration Support**: Full TOML configuration file support with schema validation
- **Multi-format Configuration**: Support for TOML, JSON, and YAML configuration files with auto-detection
- **Enhanced CLI Interface**: New unified `ccproxy` command with improved usability
- **Schema Validation**: JSON Schema generation for TOML configuration files with editor support
- **Token Generation**: `generate-token` command with force option for API key management
- **User Mapping**: Docker user mapping support for better security and file permissions
- **Keyring Support**: Secure credential storage using system keyring for OAuth tokens and sensitive data

#### API & Integration Features
- **Anthropic Messages API**: Native Anthropic Messages API endpoint with MCP integration
- **OpenAI Model Mapping**: Enhanced OpenAI model compatibility with increased token limits
- **Pre-commit Configuration**: Comprehensive pre-commit hooks for code quality assurance
- **OpenAI Utils**: Helper utilities for OpenAI API compatibility improvements
- **Systemd Support**:
  - Service template for running ccproxy as a system service
  - Setup script for automatic systemd configuration
  - Documentation for systemd deployment

#### Personal Claude Access
- **OAuth2 Authentication**: Use your existing Claude subscription without API costs
- **Local Proxy Server**: Personal API server running on your computer (localhost)
- **Subscription Integration**: Leverage Claude Pro, Team, or Enterprise subscriptions
- **No API Keys Required**: Uses Claude OAuth2 authentication through Claude Code SDK

#### API Compatibility
- **Dual API Support**: Full compatibility with both Anthropic and OpenAI API formats
- **Existing Tool Integration**: Drop-in replacement for applications expecting these APIs
- **Chat Completions**: Complete chat completion endpoints for both API formats
  - Anthropic endpoint: `/v1/chat/completions`
  - OpenAI endpoint: `/openai/v1/chat/completions`
- **Model Support**: Access to all Claude models available in your subscription:
  - `claude-opus-4-20250514` (most capable)
  - `claude-sonnet-4-20250514` (latest)
  - `claude-3-7-sonnet-20250219` (enhanced)
  - `claude-3-5-sonnet-20241022` (stable)
  - `claude-3-5-sonnet-20240620` (legacy)
- **Streaming Support**: Real-time response streaming for both API formats
- **Request Translation**: Seamless format conversion between OpenAI and Anthropic formats

#### Local Development Features
- **Health Check**: `/health` endpoint for proxy status monitoring
- **Models List**: `/v1/models` and `/openai/v1/models` endpoints
- **CORS Support**: Cross-origin request handling for local web applications
- **Error Handling**: Comprehensive error responses with proper HTTP status codes

#### Personal Setup & Configuration
- **Simple Configuration**: Easy setup with environment variables
- **Docker Isolation**: Optional containerized execution for enhanced security
- **CLI Interface**: Command-line interface for local management
- **Auto-detection**: Smart Claude CLI path resolution and configuration
- **Privacy-Focused Logging**: Minimal logging with configurable levels (no conversation storage)

#### Development & Code Quality
- **Type Safety**: Strict typing with mypy configuration for reliability
- **Modern Python**: Python 3.11+ compatibility with modern language features
- **Comprehensive Testing**: Unit and integration tests for personal use scenarios
- **Code Quality**: Ruff formatting and linting for maintainable code
- **Development Tools**: GitHub Actions workflows for automated testing

#### Local Proxy Architecture
- **Lightweight Design**: Minimal overhead for personal use
- **ClaudeClient**: Core service for Claude OAuth2 integration
- **OpenAITranslator**: Format conversion between OpenAI and Anthropic APIs
- **Streaming Services**: Real-time response streaming for interactive use
- **Security-First**: Local execution with optional Docker isolation

#### User Documentation
- **Setup Guide**: Simple installation and configuration instructions
- **Usage Examples**: Code examples for popular Python libraries
- **API Reference**: Documentation for both Anthropic and OpenAI endpoints
- **Troubleshooting**: Common issues and solutions for personal use
- **Docker Guide**: Optional containerization for enhanced isolation

### Changed

#### Documentation Updates
- **Streamlined Quick Start**: Concise examples and improved readability
- **Aider Integration**: Added documentation for Aider AI coding assistant integration
- **Systemd Setup Guide**: Comprehensive documentation for systemd deployment

#### CLI & Architecture Improvements
- **Major CLI Restructuring**: Moved CLI to dedicated `ccproxy/cli/` package with modular command structure
- **Rich CLI Experience**: Replaced basic output with Rich toolkit for colored, structured output
- **Docker Architecture**: Refactored from DockerCommandBuilder to new adapter-based architecture
- **FastAPI Subcommands**: Organized commands under FastAPI subcommand group for better organization
- **Version Management**: Implemented dynamic versioning using hatch-vcs and git tags

#### API & Routing Enhancements
- **URL Structure Refactoring**: Moved Claude Code SDK endpoints to `/cc/` prefix for clear separation
- **Dual Router Support**: Added dedicated Anthropic (`/v1/`) and OpenAI (`/openai/v1/`) routers
- **Legacy Path Support**: Maintained backward compatibility with deprecated warnings
- **Improved Error Handling**: More robust API error handling with proper HTTP status codes

#### Logging & Configuration
- **Enhanced Logging**: Integrated Rich toolkit with uvicorn for consistent, structured logging
- **Optimized Log Levels**: Reduced noise by moving verbose messages from INFO to DEBUG
- **Configuration Display**: Enhanced config display with API usage information
- **Simplified Examples**: Streamlined example configurations for better usability

#### Code Quality & Organization
- **Improved Type Safety**: Enhanced type annotations and mypy compliance throughout codebase
- **Modular Services**: Extracted credentials, Docker, and CLI utilities to dedicated modules
- **Removed Connection Pooling**: Simplified architecture by removing connection pooling for better stability
- **Documentation**: Updated repository references and comprehensive documentation additions
- **Enhanced Reverse Proxy**:
  - Improved request and response transformation pipeline
  - Better separation of concerns with dedicated transformer services
  - Factory pattern for cleaner proxy instantiation

### Fixed
- **Test Reliability**: Improved test stability and reliability across all test suites
- **Docker Integration**: Fixed Claude Docker home directory usage and command execution
- **Environment Variables**: Resolved nested environment variable handling for configuration
- **API Response Handling**: Better handling of unexpected API response types
- **File Standardization**: Consistent file endings and formatting across all files
- **Proxy Support**: Added proper HTTP/HTTPS proxy support for network requests
- **OpenAI Compatibility**: Improved compatibility with OpenAI API format and clients
- **Docker Settings**: Updated Docker image name assertions in tests

### Removed
- **Worker Pool Implementation**: Removed Node.js worker pool for simplified architecture
- **Unused Dependencies**: Cleaned up unused imports and dependencies
- **Rate Limiting Documentation**: Removed outdated rate limiting references

### Security
- **Enhanced GitHub Actions**: Security features added to CI/CD workflows
- **Docker Security**: Improved Docker isolation and user mapping
- **Input Validation**: Strengthened request validation and sanitization
- **Credential Management**: OAuth tokens and sensitive credentials now stored securely in system keyring instead of plain text

### Authentication & Reverse Proxy Features
- **OAuth Authentication**: Implemented OAuth2 authentication flow with credentials management
- **Credentials Service**: Comprehensive credential management with secure storage and token refresh
- **OAuth Client Integration**: Built-in OAuth client for Claude authentication flow
- **Reverse Proxy Modes**: Multiple transformation modes accessible via URL prefixes:
  - `/min/*` - Minimal transformations (OAuth headers only)
  - `/full/*` - Full transformations (system prompts, format conversion)
  - `/pt/*` - Passthrough mode (no transformations except OAuth)
  - `/unclaude/*` - Backward compatibility alias for full mode
- **Configurable Default Mode**: Set default proxy mode via `default_proxy_mode` setting
- **Enhanced Security**: Automatic stripping of client auth headers to prevent key leakage
- **Beta Parameter Support**: Automatic addition of beta=true for /v1/messages requests
- **Request/Response Transformers**: Modular transformation pipeline for flexible request handling

### Technical Details

#### Core Dependencies
- **claude-code-sdk**: Official Python SDK for Claude OAuth2 integration (>=0.0.14)
- **FastAPI**: Lightweight web framework for local API server (>=0.115.14)
- **Pydantic**: Data validation for API compatibility (>=2.8.0)
- **Typer**: CLI framework for easy local management (>=0.16.0)
- **Uvicorn**: Local ASGI server for personal use (>=0.34.0)

#### Personal Development Setup
- **uv**: Fast Python package manager for easy installation
- **devenv**: Optional Nix-based development environment
- **MIT License**: Open source for personal and educational use

#### Local Performance & Security
- **Lightweight Processing**: Minimal overhead for personal computer use
- **Secure Authentication**: Uses official Claude OAuth2 flow
- **Privacy Protection**: No conversation logging or external data sharing
- **Request Validation**: Input validation for API compatibility

### Personal Configuration

#### Environment Variables
- `PORT`: Local server port (default: 8000)
- `HOST`: Server host (default: 127.0.0.1 for security)
- `LOG_LEVEL`: Logging level (default: INFO)
- `CLAUDE_CLI_PATH`: Custom Claude CLI path (optional)

#### Simple Setup
- Automatic Claude CLI detection and authentication
- OAuth2 flow handled automatically on first run
- Minimal configuration required for personal use

### Initial Release Notes

This is the initial release of Claude Code Proxy API Server, a personal tool for accessing Claude AI models through your existing subscription. The project focuses on:

- **Personal Use**: Easy setup for individual developers on their own computers
- **Subscription Leverage**: Use your existing Claude subscription without additional API costs
- **Privacy & Security**: Local execution with OAuth2 authentication, no external data sharing
- **Developer Integration**: Compatible with existing tools expecting OpenAI or Anthropic APIs
- **Simplicity**: Minimal configuration required for personal use

The proxy enables you to use Claude AI models through familiar API interfaces while leveraging your Claude subscription and the official Claude Code SDK for secure authentication.

[0.1.0]: https://github.com/CaddyGlow/claude-code-proxy-api/releases/tag/v0.1.0
