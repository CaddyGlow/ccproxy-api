# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-04

### Added

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
