# Claude Code Proxy API Server

CCProxy is a reverse proxy to *api.anthropic.com/v1/messages* that leverages your current Claude subscription. This server uses the official claude-code-sdk to process requests locally, allowing you to access Claude AI through standard API interfaces without separate API costs.

## Why Use This Tool?

### Access to Your Claude Subscription
- **Use Your Existing Subscription**: Leverage your Claude Pro or Team subscription instead of paying for API access
- **OAuth2 Authentication**: Uses your existing Claude account authentication
- **Local Execution**: Runs securely on your computer
- **Docker Isolation**: Optional Docker support for isolated Claude Code execution

### Developer-Friendly API Access
- **Dual API Compatibility**: Full support for both Anthropic and OpenAI API formats
- **Existing Tool Integration**: Drop-in replacement for applications expecting OpenAI or Anthropic APIs
- **Streaming Support**: Real-time response streaming for both API formats
- **Local Development**: Perfect for personal projects and local development

## How It Works

This proxy server provides two main access modes:

### Claude Code Mode (`/sdk` prefix)
- Uses the official claude-code-sdk for request processing
- **Advantages**: Access to all tools configured in Claude Code
- **Limitations**: Cannot directly use ToolCall, limited model settings management

### API Mode (`/api` prefix)
- Direct reverse proxy to api.anthropic.com with header injection
- **Advantages**: Full access to all API features and model settings
- **Method**: Only injects necessary authentication headers and OAuth token

Both modes leverage your existing Claude subscription without requiring separate API costs.

## Features

### Core Capabilities
- **Reverse Proxy Architecture**: Routes requests to api.anthropic.com/v1/messages
- **Claude OAuth2 Integration**: Uses your Claude account authentication
- **Dual API Compatibility**: Supports both Anthropic and OpenAI API formats
- **Request Translation**: Seamless format conversion between API types
- **Streaming Support**: Real-time response streaming
- **Two Access Modes**: Claude Code (with tools) or API (direct access)

### Security & Privacy
- **Local Execution**: All processing happens on your computer
- **No API Keys Required**: Uses your existing Claude subscription via OAuth2
- **Secure Authentication**: Leverages Claude's official authentication system
- **Optional API Authentication**: Bearer token protection for API endpoints

## Quick Start

### Install

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Install ccproxy
pipx install git+https://github.com/caddyglow/claude-code-proxy-api.git@dev

# Optional: Enable shell completion
eval "$(ccproxy --show-completion zsh)"  # For zsh
eval "$(ccproxy --show-completion bash)" # For bash
```

### Authenticate

```bash
# Authenticate Claude Code CLI
claude /login

# For API/raw mode (uses Anthropic OAuth2)
ccproxy auth login
```

### Run

```bash
# Start the proxy server
ccproxy

# Use with your favorite tools
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000/sdk
# Or for API mode: export ANTHROPIC_BASE_URL=http://127.0.0.1:8000/api
aider --model claude-sonnet-4-20250514
```

That's it! For detailed setup, see our [full documentation](https://caddyglow.github.io/claude-code-proxy-api/getting-started/quickstart/).

## Usage

### With Your Existing Applications

Once running, you can point any application that uses OpenAI or Anthropic APIs sdk to your local proxy:

**For OpenAI-compatible applications:**
```bash
# For Claude Code mode
export OPENAI_BASE_URL="http://localhost:8000/sdk/"
# Or for API mode
export OPENAI_BASE_URL="http://localhost:8000/api/"
export OPENAI_API_KEY="dummy-key"  # Required by client libraries but not used
```

**For Anthropic-compatible applications:**
```bash
# For Claude Code mode
export ANTHROPIC_BASE_URL="http://localhost:8000/sdk"
# Or for API mode
export ANTHROPIC_BASE_URL="http://localhost:8000/api"
export ANTHROPIC_API_KEY="dummy-key"  # Required by client libraries but not used
```

### API Endpoints

The proxy provides two main access modes:

| Mode | URL Prefix | Description | Use Case |
|------|------------|-------------|----------|
| **Claude Code** | `/sdk/` | Uses claude-code-sdk with all tools | When you need Claude Code features |
| **API** | `/api/` | Direct proxy with header injection | When you need full API control |

#### Anthropic-Compatible Endpoints

**Messages (Claude Code mode):**
```http
POST /sdk/v1/messages
```

**Messages (API mode - direct proxy):**
```http
POST /api/v1/messages
```

**Example request:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 1000
}
```

#### OpenAI-Compatible Endpoints

**Chat Completions (Claude Code mode):**
```http
POST /sdk/v1/chat/completions
```

**Chat Completions (API mode - direct proxy):**
```http
POST /api/v1/chat/completions
```

Uses OpenAI format with automatic translation to Claude format.

#### Utility Endpoints

**Health Check:**
```http
GET /health
```

**Available Models:**
```http
GET /sdk/models        # Claude Code mode
GET /api/models        # API mode
```

**Status Endpoints:**
```http
GET /sdk/status        # Claude SDK status
GET /api/status        # Proxy API status
```

**OAuth Endpoints:**
```http
GET /oauth/callback    # OAuth callback endpoint
```

**Observability Endpoints (Optional):**
```http
GET /metrics           # Prometheus metrics (if enabled)
GET /logs/status       # Logs status (if enabled)
GET /logs/query        # Query logs (if enabled)
GET /dashboard         # Metrics dashboard (if enabled)
```

### Supported Models

CCProxy supports recent Claude models including Opus, Sonnet, and Haiku variants. The specific models available to you will depend on your Claude account and the features enabled for your subscription.

 * `claude-opus-4-20250514` 
 * `claude-sonnet-4-20250514`
 * `claude-3-7-sonnet-20250219`
 * `claude-3-5-sonnet-20241022` 
 * `claude-3-5-sonnet-20240620` 

## Configuration


### Configuration Methods

Settings can be configured through (in order of precedence):
1. Command-line arguments
2. Environment variables 
3. `.env` file
4. TOML configuration files (`.ccproxy.toml`, `ccproxy.toml`, or `~/.config/ccproxy/config.toml`)
5. Default values

### Nested Environment Variables

For more complex configurations, use the nested syntax with `__` delimiter:

```bash
# Server settings
SERVER__HOST=0.0.0.0
SERVER__PORT=8080
SERVER__LOG_LEVEL=DEBUG
SERVER__LOG_FORMAT=json

# Security
SECURITY__AUTH_TOKEN=your-token

# Scheduler settings
SCHEDULER__ENABLED=true
SCHEDULER__PRICING_UPDATE_INTERVAL_HOURS=24

# continue with all the field of the config
```



### Claude Authentication

CCProxy uses two separate authentication systems:

#### Claude CLI Authentication (Claude Code Mode)
```bash
# Authenticate Claude CLI (for Claude Code mode)
claude /login
claude /status
```
- Credentials stored at: `~/.claude/credentials.json` or `~/.config/claude/credentials.json`
- Managed by Claude CLI directly

#### CCProxy Authentication (API Mode)
```bash
# Initial login (opens browser for Anthropic OAuth2)
ccproxy auth login

# Validate credentials
ccproxy auth validate

# View credential details (auto-renews if expired)
ccproxy auth info
```
- Credentials stored in  `~/.config/ccproxy/credentials.json`
- Fallback to: `~/.claude/credentials.json` or `~/.config/claude/credentials.json`
- Tokens are automatically renewed when using `ccproxy auth info`

### API Authentication (Optional)

For added security, you can enable token authentication for your local API access. The proxy supports multiple authentication header formats, allowing you to use the standard Anthropic and OpenAI libraries without modification.

- **Anthropic SDK**: Uses `x-api-key` header by default
- **OpenAI SDK**: Uses `Authorization: Bearer` header by default

This means you can secure your proxy instance while still using the standard libraries exactly as documented.

#### Supported Authentication Headers

- **Anthropic Format**: `x-api-key: <token>` (takes precedence)
- **OpenAI/Bearer Format**: `Authorization: Bearer <token>`

All formats use the same configured token value.

#### Generate Authentication Token

```bash
# Generate a secure token
ccproxy generate-token
# Output: AUTH_TOKEN=abc123xyz789...
```

#### Configure Authentication

```bash
# Set environment variable
export AUTH_TOKEN=abc123xyz789...

# Or add to .env file
echo "AUTH_TOKEN=abc123xyz789..." >> .env
```

#### Using Authentication

When authentication is enabled, include the token in your API requests using any supported format:

**Anthropic Format (x-api-key):**
```bash
curl -H "x-api-key: abc123xyz789..." \
     -H "Content-Type: application/json" \
     -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello"}]}' \
     http://localhost:8000/sdk/v1/messages
```

**OpenAI/Bearer Format:**
```bash
curl -H "Authorization: Bearer abc123xyz789..." \
     -H "Content-Type: application/json" \
     -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello"}]}' \
     http://localhost:8000/sdk/v1/chat/completions
```


## Usage Examples

(Examples are available)[exmaples/]

### curl Example

```bash
# Claude Code mode
curl -X POST http://localhost:8000/sdk/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# API mode - direct proxy, with all the models settings
curl -X POST http://localhost:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```
   Error: Invalid authentication credentials
   Solution: Ensure you're using the correct mode for your authentication method
   ```

2. **Claude Authentication**
   ```
   Error: Claude not authenticated or credentials expired
   Solution: Run `ccproxy auth login` to authenticate
   Verify: Run `ccproxy auth validate` to check credential status
   ```
3. **API Authentication**
   ```
   Error: 401 Unauthorized - Missing authentication token
   Solution: Include authentication token using one of:
   - Anthropic format: curl -H "x-api-key: your-token" ...
   - Bearer format: curl -H "Authorization: Bearer your-token" ...
   ```

4. **Port Already in Use**
   ```
   Error: Port 8000 already in use
   Solution: Use a different port: PORT=8001 python main.py
   ```

5. **Subscription Access**
   ```
   Error: Model not available
   Solution: Check that your Claude subscription includes the requested model
   ```


## Contributing


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

Comprehensive documentation is available:

- **[Online Documentation](https://caddyglow.github.io/claude-code-proxy-api)** - Full documentation site
- **[API Reference](https://caddyglow.github.io/claude-code-proxy-api/api-reference/overview/)** - Complete API documentation
- **[Developer Guide](https://caddyglow.github.io/claude-code-proxy-api/developer-guide/architecture/)** - Architecture and development



## Support

- Issues: [GitHub Issues](https://github.com/CaddyGlow/claude-code-proxy-api/issues)
- Documentation: [Project Documentation](https://caddyglow.github.io/claude-code-proxy-api)

## Acknowledgments

- [Anthropic](https://anthropic.com) for Claude and the Claude Code SDK
- [claude-code-sdk](https://github.com/anthropics/claude-code-sdk) for the Python SDK
- The open-source community for inspiration and contributions
