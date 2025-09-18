# CCProxy API Server

`ccproxy` is a local reverse proxy server that provides unified access to multiple AI providers through a plugin-based architecture. It supports Anthropic Claude and OpenAI Codex through dedicated provider plugins, allowing you to use your existing subscriptions without separate API key billing.

## Architecture

CCProxy uses a modern plugin system that provides:

- **Provider Plugins**: Handle specific AI providers (Claude SDK, Claude API, Codex)
- **System Plugins**: Add functionality like pricing, logging, monitoring, and permissions
- **Unified API**: Consistent interface across all providers
- **Format Translation**: Seamless conversion between Anthropic and OpenAI formats

## Provider Plugins

The server provides access through different provider plugins:

### Claude SDK Plugin (`/claude/sdk`)
Routes requests through the local `claude-code-sdk`. This enables access to tools configured in your Claude environment and includes MCP (Model Context Protocol) integration.

**Endpoints:**
- `POST /claude/sdk/v1/messages` - Anthropic messages API
- `POST /claude/sdk/v1/chat/completions` - OpenAI chat completions
- `POST /claude/sdk/{session_id}/v1/messages` - Session-based messages
- `POST /claude/sdk/{session_id}/v1/chat/completions` - Session-based completions

### Claude API Plugin (`/claude`)
Acts as a direct reverse proxy to `api.anthropic.com`, injecting the necessary authentication headers. This provides full access to the underlying API features and model settings.

**Endpoints:**
- `POST /claude/v1/messages` - Anthropic messages API
- `POST /claude/v1/chat/completions` - OpenAI chat completions
- `POST /claude/v1/responses` - OpenAI Responses API (via adapters)
- `POST /claude/{session_id}/v1/responses` - Session-based Responses API
- `GET /claude/v1/models` - List available models

### Codex Plugin (`/codex`)
Provides access to OpenAI's APIs (Responses and Chat Completions) through OAuth2 PKCE.

**Endpoints:**
- `POST /codex/v1/responses` - OpenAI Responses API
- `POST /codex/{session_id}/v1/responses` - Session-based Responses
- `POST /codex/v1/chat/completions` - OpenAI Chat Completions
- `POST /codex/{session_id}/v1/chat/completions` - Session-based Chat Completions
- `GET /codex/v1/models` - List available models
- `POST /codex/v1/messages` - Anthropic messages (converted via adapters)
- `POST /codex/{session_id}/v1/messages` - Session-based Anthropic messages

All plugins support both Anthropic and OpenAI-compatible API formats for requests and responses, including streaming.

## Installation

```bash
# Install with uv (recommended)
uv tool install ccproxy-api

# Or with pipx
pipx install ccproxy-api

# For development version
uv tool install git+https://github.com/caddyglow/ccproxy-api.git@dev

# Optional: Enable shell completion
eval "$(ccproxy --show-completion zsh)"  # For zsh
eval "$(ccproxy --show-completion bash)" # For bash
```

**Prerequisites:**
- Python 3.11+
- Claude Code SDK (for SDK mode): `npm install -g @anthropic-ai/claude-code`

## Authentication

Each provider plugin has its own authentication mechanism:

### Claude SDK Plugin
Relies on the authentication handled by the `claude-code-sdk`:
```bash
claude /login
# Or for long-lived tokens:
claude setup-token
```

### Claude API Plugin
Uses OAuth2 flow to obtain credentials for direct API access:
```bash
ccproxy auth login
```

### Codex Plugin
Uses OpenAI OAuth2 PKCE flow for Codex access:
```bash
ccproxy auth login-openai
```

Check authentication status:
```bash
ccproxy auth status  # Check all providers
```

## Usage

### Starting the Server

```bash
# Start the proxy server (default port 8000)
ccproxy serve

# With custom port
ccproxy serve --port 8080

# Development mode with reload
ccproxy serve --reload

# With debug logging
ccproxy serve --log-level debug

# Enable or disable plugins at startup
ccproxy serve --enable-plugin metrics --disable-plugin docker
```

The server will start on `http://127.0.0.1:8000` by default.

### Client Configuration

Point your existing tools and applications to the local proxy instance. Most client libraries require an API key (use any dummy value like "sk-dummy").

**For OpenAI-compatible clients:**
```bash
export OPENAI_API_KEY="sk-dummy"
export OPENAI_BASE_URL="http://localhost:8000/claude/sdk/v1"  # For Claude SDK
# Or
export OPENAI_BASE_URL="http://localhost:8000/claude/v1"      # For Claude API
# Or
export OPENAI_BASE_URL="http://localhost:8000/codex/v1"       # For Codex
```

**For Anthropic clients:**
```bash
export ANTHROPIC_API_KEY="sk-dummy"
export ANTHROPIC_BASE_URL="http://localhost:8000/claude/sdk"  # For Claude SDK
# Or
export ANTHROPIC_BASE_URL="http://localhost:8000/claude"      # For Claude API
# Optional (via adapters)
# export ANTHROPIC_BASE_URL="http://localhost:8000/codex"
```

## System Plugins

### Pricing Plugin
Tracks token usage and calculates costs based on current model pricing.

### Permissions Plugin  
Manages MCP (Model Context Protocol) permissions for tool access control.

### Raw HTTP Logger Plugin
Logs raw HTTP requests and responses for debugging (configurable via environment variables).

## Configuration

CCProxy can be configured through:
1. Command-line arguments
2. Environment variables (use `__` for nesting, e.g., `LOGGING__LEVEL=debug`)
3. TOML configuration files (`.ccproxy.toml`, `ccproxy.toml`)

### Plugin Config Quickstart

Enable plugins and configure them under the `plugins.*` namespace in TOML or env vars.

TOML example:

```toml
enable_plugins = true

# Access log plugin
[plugins.access_log]
enabled = true
client_enabled = true
client_format = "structured"
client_log_file = "/tmp/ccproxy/access.log"

# Request tracer plugin
[plugins.request_tracer]
enabled = true
json_logs_enabled = true
raw_http_enabled = true
log_dir = "/tmp/ccproxy/traces"

# DuckDB storage (used by analytics)
[plugins.duckdb_storage]
enabled = true

# Analytics (logs API)
[plugins.analytics]
enabled = true

# Metrics (Prometheus endpoints and optional Pushgateway)
[plugins.metrics]
enabled = true
# pushgateway_enabled = true
# pushgateway_url = "http://localhost:9091"
# pushgateway_job = "ccproxy"
# pushgateway_push_interval = 60
```

Environment variable equivalents (nested with `__`):

```bash
export ENABLE_PLUGINS=true
export PLUGINS__ACCESS_LOG__ENABLED=true
export PLUGINS__ACCESS_LOG__CLIENT_ENABLED=true
export PLUGINS__ACCESS_LOG__CLIENT_FORMAT=structured
export PLUGINS__ACCESS_LOG__CLIENT_LOG_FILE=/tmp/ccproxy/access.log

export PLUGINS__REQUEST_TRACER__ENABLED=true
export PLUGINS__REQUEST_TRACER__JSON_LOGS_ENABLED=true
export PLUGINS__REQUEST_TRACER__RAW_HTTP_ENABLED=true
export PLUGINS__REQUEST_TRACER__LOG_DIR=/tmp/ccproxy/traces

export PLUGINS__DUCKDB_STORAGE__ENABLED=true
export PLUGINS__ANALYTICS__ENABLED=true
export PLUGINS__METRICS__ENABLED=true
# export PLUGINS__METRICS__PUSHGATEWAY_ENABLED=true
# export PLUGINS__METRICS__PUSHGATEWAY_URL=http://localhost:9091
```

See more details in Configuration and individual plugin pages:
- `docs/getting-started/configuration.md`
- `config.example.toml`
- Plugin READMEs under `plugins/*/README.md`

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed setup instructions
- [Quick Start](getting-started/quickstart.md) - Get running in minutes
- [API Usage](user-guide/api-usage.md) - Using the API endpoints
- [Authentication](user-guide/authentication.md) - Managing credentials
