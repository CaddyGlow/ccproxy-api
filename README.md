# CCProxy API Server

`ccproxy` is a local reverse proxy server for Anthropic Claude LLM at `api.anthropic.com/v1/messages`. It allows you to use your existing Claude Max subscription to interact with the Anthropic API, bypassing the need for separate API key billing.

The server provides two primary modes of operation:
*   **SDK Mode (`/sdk`):** Routes requests through the local `claude-code-sdk`. This enables access to tools configured in your Claude environment and includes an integrated MCP (Model Context Protocol) server for permission management.
*   **API Mode (`/api`):** Acts as a direct reverse proxy, injecting the necessary authentication headers. This provides full access to the underlying API features and model settings.

It includes a translation layer to support both Anthropic and OpenAI-compatible API formats for requests and responses, including streaming.

## Installation

```bash
# The official claude-code CLI is required for SDK mode
npm install -g @anthropic-ai/claude-code

# run it with uv
uvx ccproxy-api

# run it with pipx
pipx run ccproxy-api

# install with uv
uv tool install ccproxy-api

# Install ccproxy with pip
pipx install ccproxy-api

# Optional: Enable shell completion
eval "$(ccproxy --show-completion zsh)"  # For zsh
eval "$(ccproxy --show-completion bash)" # For bash
```


For dev version replace `ccproxy-api` with `git+https://github.com/caddyglow/ccproxy-api.git@dev`

## Authentication

The proxy uses two different authentication mechanisms depending on the mode.

1.  **Claude CLI (`sdk` mode):**
    This mode relies on the authentication handled by the `claude-code-sdk`.
    ```bash
    claude /login
    ```

    It's also possible now to get a long live token to avoid renewing issues
    using
    ```sh
    ```bash
    claude setup-token`

2.  **ccproxy (`api` mode):**
    This mode uses its own OAuth2 flow to obtain credentials for direct API access.
    ```bash
    ccproxy auth login
    ```

    If you are already connected with Claude CLI the credentials should be found automatically

You can check the status of these credentials with `ccproxy auth validate` and `ccproxy auth info`.

Warning is show on start up if no credentials are setup.

## Usage

### Running the Server

```bash
# Start the proxy server
ccproxy
```
The server will start on `http://127.0.0.1:8000` by default.

### Client Configuration

Point your existing tools and applications to the local proxy instance by setting the appropriate environment variables. A dummy API key is required by most client libraries but is not used by the proxy itself.

**For OpenAI-compatible clients:**
```bash
# For SDK mode
export OPENAI_BASE_URL="http://localhost:8000/sdk/v1"
# For API mode
export OPENAI_BASE_URL="http://localhost:8000/api/v1"

export OPENAI_API_KEY="dummy-key"
```

**For Anthropic-compatible clients:**
```bash
# For SDK mode
export ANTHROPIC_BASE_URL="http://localhost:8000/sdk"
# For API mode
export ANTHROPIC_BASE_URL="http://localhost:8000/api"

export ANTHROPIC_API_KEY="dummy-key"
```


## MCP Server Integration & Permission System

In SDK mode, CCProxy automatically configures an MCP (Model Context Protocol) server that provides permission checking tools for Claude Code. This enables interactive permission management for tool execution.

### Permission Management

**Starting the Permission Handler:**
```bash
# In a separate terminal, start the permission handler
ccproxy permission-handler

# Or with custom settings
ccproxy permission-handler --host 127.0.0.1 --port 8000
```

The permission handler provides:
- **Real-time Permission Requests**: Streams permission requests via Server-Sent Events (SSE)
- **Interactive Approval/Denial**: Command-line interface for managing tool permissions
- **Automatic MCP Integration**: Works seamlessly with Claude Code SDK tools

**Working Directory Control:**
Control which project the Claude SDK API can access using the `--cwd` flag:
```bash
# Set working directory for Claude SDK
ccproxy --claude-code-options-cwd /path/to/your/project

# Example with permission bypass and formatted output
ccproxy --claude-code-options-cwd /tmp/tmp.AZyCo5a42N \
        --claude-code-options-permission-mode bypassPermissions \
        --claude-sdk-message-mode formatted

# Alternative: Change to project directory and start ccproxy
cd /path/to/your/project
ccproxy
```

### Claude SDK Message Formatting

CCProxy supports flexible message formatting through the `sdk_message_mode` configuration:

- **`forward`** (default): Preserves original Claude SDK content blocks with full metadata
- **`formatted`**: Converts content to XML tags with pretty-printed JSON data
- **`ignore`**: Filters out Claude SDK-specific content entirely

Configure via environment variables:
```bash
# Use formatted XML output
CLAUDE__SDK_MESSAGE_MODE=formatted ccproxy

# Use compact formatting without pretty-printing
CLAUDE__PRETTY_FORMAT=false ccproxy
```

## Claude SDK Pool Mode

CCProxy supports connection pooling for Claude Code SDK clients to improve request performance by maintaining a pool of pre-initialized Claude instances.

### Benefits
- **Reduced Latency**: Eliminates Claude Code startup overhead on each request
- **Improved Performance**: Reuses established connections for faster response times
- **Resource Efficiency**: Maintains a configurable pool size to balance performance and resource usage

### Usage
Pool mode is disabled by default and can be enabled using the CLI flag:

```bash
# Enable pool mode with default settings
ccproxy --sdk-enable-pool

# Configure pool size (default: 3)
ccproxy --sdk-enable-pool --sdk-pool-size 5
```

### Limitations
- **No Dynamic Options**: Pool instances cannot change Claude options (max_tokens, model, etc.) after initialization
- **Shared Configuration**: All requests using the pool must use identical Claude configuration
- **Memory Usage**: Each pool instance consumes additional memory

Pool mode is most effective for high-frequency requests with consistent configuration requirements.

## Using with Aider

CCProxy works seamlessly with Aider and other AI coding assistants:

### Anthropic Mode
```bash
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000/api
aider --model claude-sonnet-4-20250514
```

### OpenAI Mode with Model Mapping

If your tool only supports OpenAI settings, ccproxy automatically maps OpenAI models to Claude:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/api/v1
aider --model o3-mini
```

### API Mode (Direct Proxy)

For minimal interference and direct API access:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/api/v1
aider --model o3-mini
```

### `curl` Example

```bash
# SDK mode
curl -X POST http://localhost:8000/sdk/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# API mode
curl -X POST http://localhost:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```
More examples are available in the `examples/` directory.

## Endpoints

The proxy exposes endpoints under two prefixes, corresponding to its operating modes.

| Mode | URL Prefix | Description | Use Case |
|------|------------|-------------|----------|
| **SDK** | `/sdk/` | Uses `claude-code-sdk` with its configured tools. | Accessing Claude with local tools. |
| **API** | `/api/` | Direct proxy with header injection. | Full API control, direct access. |

*   **Anthropic:**
    *   `POST /sdk/v1/messages`
    *   `POST /api/v1/messages`
*   **OpenAI-Compatible:**
    *   `POST /sdk/v1/chat/completions`
    *   `POST /api/v1/chat/completions`
*   **Utility:**
    *   `GET /health`
    *   `GET /sdk/models`, `GET /api/models`
    *   `GET /sdk/status`, `GET /api/status`
    *   `GET /oauth/callback`
*   **MCP & Permissions:**
    *   `POST /mcp/permission/check` - MCP permission checking endpoint
    *   `GET /permissions/stream` - SSE stream for permission requests
    *   `GET /permissions/{id}` - Get permission request details
    *   `POST /permissions/{id}/respond` - Respond to permission request
*   **Observability (Optional):**
    *   `GET /metrics`
    *   `GET /logs/status`, `GET /logs/query`
    *   `GET /dashboard`

## Supported Models

CCProxy supports recent Claude models including Opus, Sonnet, and Haiku variants. The specific models available to you will depend on your Claude account and the features enabled for your subscription.

 * `claude-opus-4-20250514`
 * `claude-sonnet-4-20250514`
 * `claude-3-7-sonnet-20250219`
 * `claude-3-5-sonnet-20241022`
 * `claude-3-5-sonnet-20240620`

## Configuration

Settings can be configured through (in order of precedence):
1. Command-line arguments
2. Environment variables
3. `.env` file
4. TOML configuration files (`.ccproxy.toml`, `ccproxy.toml`, or `~/.config/ccproxy/config.toml`)
5. Default values

For complex configurations, you can use a nested syntax for environment variables with `__` as a delimiter:

```bash
# Server settings
SERVER__HOST=0.0.0.0
SERVER__PORT=8080
# etc.
```

## Securing the Proxy (Optional)

You can enable token authentication for the proxy. This supports multiple header formats (`x-api-key` for Anthropic, `Authorization: Bearer` for OpenAI) for compatibility with standard client libraries.

**1. Generate a Token:**
```bash
ccproxy generate-token
# Output: SECURITY__AUTH_TOKEN=abc123xyz789...
```

**2. Configure the Token:**
```bash
# Set environment variable
export SECURITY__AUTH_TOKEN=abc123xyz789...

# Or add to .env file
echo "SECURITY__AUTH_TOKEN=abc123xyz789..." >> .env
```

**3. Use in Requests:**
When authentication is enabled, include the token in your API requests.
```bash
# Anthropic Format (x-api-key)
curl -H "x-api-key: your-token" ...

# OpenAI/Bearer Format
curl -H "Authorization: Bearer your-token" ...
```

## Observability

`ccproxy` includes an optional but powerful observability suite for monitoring and analytics. When enabled, it provides:

*   **Prometheus Metrics:** A `/metrics` endpoint for real-time operational monitoring.
*   **Access Log Storage:** Detailed request logs, including token usage and costs, are stored in a local DuckDB database.
*   **Analytics API:** Endpoints to query and analyze historical usage data.
*   **Real-time Dashboard:** A live web interface at `/dashboard` to visualize metrics and request streams.

These features are disabled by default and can be enabled via configuration. For a complete guide on setting up and using these features, see the [Observability Documentation](docs/observability.md).

## Troubleshooting

### Common Issues

1.  **Authentication Error:** Ensure you're using the correct mode (`/sdk` or `/api`) for your authentication method.
2.  **Claude Credentials Expired:** Run `ccproxy auth login` to refresh credentials for API mode. Run `claude /login` for SDK mode.
3.  **Missing API Auth Token:** If you've enabled security, include the token in your request headers.
4.  **Port Already in Use:** Start the server on a different port: `ccproxy --port 8001`.
5.  **Model Not Available:** Check that your Claude subscription includes the requested model.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

- **[Online Documentation](https://caddyglow.github.io/ccproxy-api)**
- **[API Reference](https://caddyglow.github.io/ccproxy-api/api-reference/overview/)**
- **[Developer Guide](https://caddyglow.github.io/ccproxy-api/developer-guide/architecture/)**

## Support

- Issues: [GitHub Issues](https://github.com/CaddyGlow/ccproxy-api/issues)
- Documentation: [Project Documentation](https://caddyglow.github.io/ccproxy-api)

## Acknowledgments

- [Anthropic](https://anthropic.com) for Claude and the Claude Code SDK
- The open-source community
