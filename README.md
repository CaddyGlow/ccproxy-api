# CCProxy API Server

`ccproxy` is a local reverse proxy server built on a plugin-based architecture that provides unified access to multiple AI providers through a single interface. It supports both Anthropic Claude and OpenAI Codex backends, allowing you to use your existing subscriptions without separate API key billing.

## Architecture

CCProxy is designed around a modular plugin system where each AI provider is implemented as a separate plugin:

- **API Layer** (`ccproxy/api/`) - FastAPI endpoints and middleware
- **Plugin System** (`plugins/`) - Provider-specific implementations:
  - `claude_api` - Direct Claude API access with OAuth2
  - `claude_sdk` - Uses claude-code-sdk with local tools
  - `codex` - OpenAI Codex Response API integration
  - `permissions` - Permission management for MCP integration
  - `pricing` - Token pricing and cost calculation
  - `raw_http_logger` - HTTP request/response logging
- **Core Services** (`ccproxy/services/`) - ProxyService for request delegation, provider context management
- **Configuration** (`ccproxy/config/`) - Settings and validation
- **Models** (`ccproxy/models/`) - Pydantic data models

### Plugin Architecture

Each provider plugin follows a consistent delegation pattern:

- **Adapter** - Main plugin interface that delegates to the ProxyService
- **Transformers** - Handle request/response header and body transformation
- **Detection Services** - Provider-specific capability detection
- **Format Adapters** - Protocol conversion (e.g., OpenAI ↔ Anthropic formats)
- **Auth Manager** - Provider-specific authentication

This architecture enables easy extension to new AI providers while maintaining consistent behavior and authentication patterns.

## Supported Providers

### Anthropic Claude

Access Claude through multiple modes:

- **SDK Mode (`/claude`):** Routes requests through the local `claude-code-sdk`. This enables access to tools configured in your Claude environment and includes an integrated MCP (Model Context Protocol) server for permission management.
- **API Mode (`/api`):** Acts as a direct reverse proxy, injecting the necessary authentication headers. This provides full access to the underlying API features and model settings.

### OpenAI Codex Response API (Experimental)

Access OpenAI's [Response API](https://platform.openai.com/docs/api-reference/responses) via your ChatGPT Plus subscription at the `/api/codex` endpoints.

- **Response API (`/api/codex/responses`):** Direct reverse proxy to ChatGPT backend for conversation responses
- **Session Management:** Supports both auto-generated and persistent session IDs for conversation continuity
- **OpenAI OAuth:** Uses the same OAuth2 PKCE authentication flow as the official Codex CLI
- **ChatGPT Plus Required:** Requires an active ChatGPT Plus subscription for API access

The server includes a translation layer to support both Anthropic and OpenAI-compatible API formats for requests and responses, including streaming.

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
- Claude Code SDK (optional, for SDK mode): `npm install -g @anthropic-ai/claude-code`

## Authentication

The proxy uses different authentication mechanisms depending on the provider and mode.

### Claude Authentication

1. **Claude SDK Mode (`/claude` endpoints):**
   This mode relies on the authentication handled by the `claude-code-sdk`.

   ```bash
   claude /login
   # Or for long-lived tokens:
   claude setup-token
   ```

2. **Claude API Mode (`/api` endpoints):**
   This mode uses its own OAuth2 flow to obtain credentials for direct API access.

   ```bash
   ccproxy auth login
   ```

   If you are already connected with Claude CLI the credentials should be found automatically.

### OpenAI Codex Authentication

The Codex Response API requires ChatGPT Plus subscription and OAuth2 authentication:

```bash
# Enable Codex provider
ccproxy config codex --enable

# Login via CCProxy CLI (opens browser)
ccproxy auth login-openai

# Check authentication status for all providers
ccproxy auth status
```

**Important Notes:**
- Credentials are stored in `$HOME/.codex/auth.json`
- CCProxy reuses existing Codex CLI credentials when available
- If credentials are expired, CCProxy attempts automatic renewal

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
```

The server will start on `http://127.0.0.1:8000` by default.

### API Endpoints

#### Claude SDK Plugin (`/claude`)
- `POST /claude/v1/messages` - Anthropic messages API
- `POST /claude/v1/chat/completions` - OpenAI chat completions
- `POST /claude/{session_id}/v1/messages` - Session-based messages
- `POST /claude/{session_id}/v1/chat/completions` - Session-based completions

#### Claude API Plugin (`/api`)
- `POST /api/v1/messages` - Anthropic messages API
- `POST /api/v1/chat/completions` - OpenAI chat completions
- `GET /api/v1/models` - List available models

#### Codex Plugin (`/api/codex`)
- `POST /api/codex/responses` - Codex response API
- `POST /api/codex/chat/completions` - OpenAI format
- `POST /api/codex/{session_id}/responses` - Session-based responses
- `POST /api/codex/{session_id}/chat/completions` - Session-based completions
- `POST /api/codex/v1/chat/completions` - Standard OpenAI v1 endpoint
- `GET /api/codex/v1/models` - List available models

#### System Endpoints
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /permissions/stream` - MCP permission events

### Client Configuration

Point your existing tools and applications to the local proxy instance. Most client libraries require an API key (use any dummy value like "sk-dummy").

**For OpenAI-compatible clients:**
```bash
export OPENAI_API_KEY="sk-dummy"
export OPENAI_BASE_URL="http://localhost:8000/claude/v1"    # For Claude SDK
# Or
export OPENAI_BASE_URL="http://localhost:8000/api/v1"        # For Claude API
# Or
export OPENAI_BASE_URL="http://localhost:8000/api/codex/v1"  # For Codex
```

**For Anthropic clients:**
```bash
export ANTHROPIC_API_KEY="sk-dummy"
export ANTHROPIC_BASE_URL="http://localhost:8000/claude"  # For Claude SDK
# Or
export ANTHROPIC_BASE_URL="http://localhost:8000/api"      # For Claude API
```

### Example Usage

```python
from openai import OpenAI

# Configure client for Claude SDK mode
client = OpenAI(
    api_key="sk-dummy",
    base_url="http://localhost:8000/claude/v1"
)

# Make a request
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

print(response.choices[0].message.content)
```

## Configuration

CCProxy can be configured through:
1. Command-line arguments
2. Environment variables (use `__` for nesting, e.g., `SERVER__LOG_LEVEL=debug`)
3. TOML configuration files (`.ccproxy.toml`, `ccproxy.toml`)

### Environment Variables

```bash
# Enable verbose API logging
CCPROXY_VERBOSE_API=true

# Set request logging directory
CCPROXY_REQUEST_LOG_DIR=/tmp/ccproxy/request

# Enable raw HTTP logging
PLUGINS__RAWHTTPLOGGERCONFIG__ENABLED=true
PLUGINS__RAWHTTPLOGGERCONFIG__LOG_DIR=/tmp/ccproxy/raw

# Set server log level
SERVER__LOG_LEVEL=debug
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/CaddyGlow/ccproxy-api.git
cd ccproxy-api

# Setup development environment
make setup  # Installs dependencies and pre-commit hooks
```

### Running Tests

```bash
make test        # All tests with coverage
make test-unit   # Fast unit tests only
make ci          # Full CI pipeline
```

### Code Quality

```bash
make pre-commit  # Run all checks with auto-fixes
make format      # Format code
make lint        # Run linting
make typecheck   # Type checking
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## Documentation

- [Installation Guide](docs/getting-started/installation.md)
- [Quick Start](docs/getting-started/quickstart.md)
- [API Usage](docs/user-guide/api-usage.md)
- [Authentication](docs/user-guide/authentication.md)
- [Plugin Development](docs/PLUGIN_SYSTEM_DOCUMENTATION.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Support

- **Issues**: [GitHub Issues](https://github.com/CaddyGlow/ccproxy-api/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CaddyGlow/ccproxy-api/discussions)