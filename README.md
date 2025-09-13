[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/caddyglow-ccproxy-api-badge.png)](https://mseep.ai/app/caddyglow-ccproxy-api)

# CCProxy API Server

`ccproxy` is a local reverse proxy server that provides unified access to multiple AI providers through a single interface. It supports both Anthropic Claude and OpenAI Codex backends, allowing you to use your existing subscriptions without separate API key billing.

## Supported Providers

### Anthropic Claude

Access Claude via your Claude Max subscription at `api.anthropic.com/v1/messages`.

The server provides two primary modes of operation:

- **SDK Mode (`/sdk`):** Routes requests through the local `claude-code-sdk`. This enables access to tools configured in your Claude environment and includes an integrated MCP (Model Context Protocol) server for permission management.
- **API Mode (`/api`):** Acts as a direct reverse proxy, injecting the necessary authentication headers. This provides full access to the underlying API features and model settings.

### OpenAI Codex Response API (Experimental)

Access OpenAI's [Response API](https://platform.openai.com/docs/api-reference/responses) via your ChatGPT Plus subscription. This provides programmatic access to ChatGPT models through the `chatgpt.com/backend-api/codex` endpoint.

- **Response API (`/codex/responses`):** Direct reverse proxy to ChatGPT backend for conversation responses
- **Session Management:** Supports both auto-generated and persistent session IDs for conversation continuity
- **OpenAI OAuth:** Uses the same OAuth2 PKCE authentication flow as the official Codex CLI
- **ChatGPT Plus Required:** Requires an active ChatGPT Plus subscription for API access
- **Instruction Prompt:** Automatically injects the Codex instruction prompt into conversations

The server includes a translation layer to support both Anthropic and OpenAI-compatible API formats for requests and responses, including streaming.

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

The proxy uses different authentication mechanisms depending on the provider and mode.

### Claude Authentication

1.  **Claude CLI (`sdk` mode):**
    This mode relies on the authentication handled by the `claude-code-sdk`.

    ```bash
    claude /login
    ```

    It's also possible now to get a long live token to avoid renewing issues
    using

    ```bash
    claude setup-token
    ```

2.  **ccproxy (`api` mode):**
    This mode uses its own OAuth2 flow to obtain credentials for direct API access.

    ```bash
    ccproxy auth login
    ```

    If you are already connected with Claude CLI the credentials should be found automatically

### OpenAI Codex Authentication (Experimental)

The Codex Response API requires ChatGPT Plus subscription and OAuth2 authentication:

```bash
# Enable Codex provider
ccproxy config codex --enable

# Authentication options:

# Option 1: Use existing Codex CLI credentials (if available)
# CCProxy will automatically detect and use valid credentials from:
# - $HOME/.codex/auth.json (Codex CLI credentials)
# - Automatically renews tokens if expired but refresh token is valid

# Option 2: Login via CCProxy CLI (opens browser)
ccproxy auth login-openai

# Option 3: Use the official Codex CLI
codex auth login

# Check authentication status for all providers
ccproxy auth status
```

**Important Notes:**

- Credentials are stored in `$HOME/.codex/auth.json`
- CCProxy reuses existing Codex CLI credentials when available
- If credentials are expired, CCProxy attempts automatic renewal
- Without valid credentials, users must authenticate using either CCProxy or Codex CLI

### Authentication Status

You can check the status of all credentials with:

```bash
ccproxy auth status       # All providers
ccproxy auth validate     # Claude only
ccproxy auth info         # Claude only
```

Warning is shown on startup if no credentials are setup.

## Usage

### Running the Server

```bash
# Start the proxy server
ccproxy
```

The server will start on `http://127.0.0.1:8000` by default.

### Client Configuration

Point your existing tools and applications to the local proxy instance by setting the appropriate environment variables. A dummy API key is required by most client libraries but is not used by the proxy itself.

**For Claude (OpenAI-compatible clients):**

```bash
# For SDK mode
export OPENAI_BASE_URL="http://localhost:8000/sdk/v1"
# For API mode
export OPENAI_BASE_URL="http://localhost:8000/api/v1"

export OPENAI_API_KEY="dummy-key"
```

**For Claude (Anthropic-compatible clients):**

```bash
# For SDK mode
export ANTHROPIC_BASE_URL="http://localhost:8000/sdk"
# For API mode
export ANTHROPIC_BASE_URL="http://localhost:8000/api"

export ANTHROPIC_API_KEY="dummy-key"
```

**For OpenAI Codex Response API:**

```bash
# Create a new conversation response (auto-generated session)
curl -X POST http://localhost:8000/codex/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [
      {"role": "user", "content": "Hello, can you help me with Python?"}
    ]
  }'

# Continue conversation with persistent session ID
curl -X POST http://localhost:8000/codex/my_session_123/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [
      {"role": "user", "content": "Show me an example of async/await"}
    ]
  }'

# Stream responses (SSE format)
curl -X POST http://localhost:8000/codex/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "stream": true
  }'
```

**For OpenAI-compatible clients using Codex:**

```yaml
# Example aichat configuration (~/.config/aichat/config.yaml)
clients:
  - type: claude
    api_base: http://127.0.0.1:8000/codex

# Usage
aichat --model openai:gpt-5 "hello"
```

**Important Codex Limitations:**

- Limited model support (e.g., `gpt-5` works, others may not)
- Many OpenAI parameters not supported (temperature, top_p, etc.)
- Reasoning content appears in XML tags for capable models

**Note:** The Codex instruction prompt is automatically injected into all conversations to maintain compatibility with the ChatGPT backend.

### Codex Response API Details

#### Session Management

The Codex Response API supports flexible session management for conversation continuity:

- **Auto-generated sessions**: `POST /codex/responses` - Creates a new session ID for each request
- **Persistent sessions**: `POST /codex/{session_id}/responses` - Maintains conversation context across requests
- **Header forwarding**: Optional `session_id` header for custom session tracking

#### Instruction Prompt Injection

**Important:** CCProxy automatically injects the Codex instruction prompt into every conversation. This is required for proper interaction with the ChatGPT backend but affects your token usage:

- The instruction prompt is prepended to your messages
- This consumes additional tokens in each request
- The prompt ensures compatibility with ChatGPT's response generation
- You cannot disable this injection as it's required by the backend

#### Model Differences

The Response API models differ from standard OpenAI API models:

- Uses ChatGPT Plus models (e.g., `gpt-4`, `gpt-4-turbo`)
- Model behavior matches ChatGPT web interface
- Token limits and pricing follow ChatGPT Plus subscription terms
- See [OpenAI Response API Documentation](https://platform.openai.com/docs/api-reference/responses) for details

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

### Using with OpenAI Codex

For tools that support custom API bases, you can use the Codex provider. Note that this has significant limitations compared to Claude providers.

**Example with aichat:**

```yaml
# ~/.config/aichat/config.yaml
clients:
  - type: claude
    api_base: http://127.0.0.1:8000/codex
```

```bash
# Usage with confirmed working model
aichat --model openai:gpt-5 "hello"
```

**Codex Limitations:**

- Only select models work (gpt-5 confirmed, others may fail)
- No support for temperature, top_p, or most OpenAI parameters
- When using reasoning models, reasoning appears as XML tags in output

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

The proxy exposes endpoints under multiple prefixes for different providers and modes.

### Claude Endpoints

| Mode    | URL Prefix | Description                                       | Use Case                           |
| ------- | ---------- | ------------------------------------------------- | ---------------------------------- |
| **SDK** | `/sdk/`    | Uses `claude-code-sdk` with its configured tools. | Accessing Claude with local tools. |
| **API** | `/api/`    | Direct proxy with header injection.               | Full API control, direct access.   |

- **Anthropic Format:**
  - `POST /sdk/v1/messages`
  - `POST /api/v1/messages`
- **OpenAI-Compatible Format:**
  - `POST /sdk/v1/chat/completions`
  - `POST /api/v1/chat/completions`

### OpenAI Codex Endpoints

- **Response API:**
  - `POST /codex/responses` - Create response with auto-generated session
  - `POST /codex/{session_id}/responses` - Create response with persistent session
  - `POST /codex/chat/completions` - OpenAI-compatible chat completions endpoint
  - `POST /codex/v1/chat/completions` - Alternative OpenAI-compatible endpoint
  - Supports streaming via SSE when `stream: true` is set
  - See [Response API docs](https://platform.openai.com/docs/api-reference/responses)

**Codex Chat Completions Limitations:**

- **No Tool/Function Calling Support**: Tool use and function calling are not supported (use `/codex/responses` for tool calls)
- **Limited Parameter Support**: Many OpenAI parameters (temperature, top_p, frequency_penalty, etc.) are not supported
- **Restricted Model Support**: Only certain models work (e.g., `gpt-5` confirmed working, others may fail)
- **No Custom System Prompts**: System messages and instructions are overridden by the required Codex instruction prompt
- **Reasoning Mode**: GPT models with reasoning capabilities pass reasoning content between XML tags (`<reasoning>...</reasoning>`)
- **Session Management**: Uses auto-generated sessions; persistent sessions require the `/codex/{session_id}/responses` endpoint
- **ChatGPT Plus Required**: Requires active ChatGPT Plus subscription for access

**Note**: The `/codex/responses` endpoint supports tool calling and more parameters, but specific feature availability depends on ChatGPT's backend - users should test individual capabilities.

### Utility Endpoints

- **Health & Status:**
  - `GET /health`
  - `GET /sdk/models`, `GET /api/models`
  - `GET /sdk/status`, `GET /api/status`
- **Authentication:**
  - `GET /oauth/callback` - OAuth callback for both Claude and OpenAI
- **MCP & Permissions:**
  - `POST /mcp/permission/check` - MCP permission checking endpoint
  - `GET /permissions/stream` - SSE stream for permission requests
  - `GET /permissions/{id}` - Get permission request details
  - `POST /permissions/{id}/respond` - Respond to permission request
- **Observability (Optional):**
  - `GET /metrics`
  - `GET /logs/status`, `GET /logs/query`
  - `GET /dashboard`

## Supported Models

CCProxy supports recent Claude models including Opus, Sonnet, and Haiku variants. The specific models available to you will depend on your Claude account and the features enabled for your subscription.

- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514`
- `claude-3-7-sonnet-20250219`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-sonnet-20240620`

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

- **Prometheus Metrics:** A `/metrics` endpoint for real-time operational monitoring.
- **Access Log Storage:** Detailed request logs, including token usage and costs, are stored in a local DuckDB database.
- **Analytics API:** Endpoints to query and analyze historical usage data.
- **Real-time Dashboard:** A live web interface at `/dashboard` to visualize metrics and request streams.

These features are disabled by default and can be enabled via configuration. For a complete guide on setting up and using these features, see the [Observability Documentation](docs/observability.md).

## Troubleshooting

### Common Issues

1.  **Authentication Error:** Ensure you're using the correct mode (`/sdk` or `/api`) for your authentication method.
2.  **Claude Credentials Expired:** Run `ccproxy auth login` to refresh credentials for API mode. Run `claude /login` for SDK mode.
3.  **OpenAI/Codex Authentication Failed:**
    - Check if valid credentials exist: `ccproxy auth status`
    - Ensure you have an active ChatGPT Plus subscription
    - Try re-authenticating: `ccproxy auth login-openai` or `codex auth login`
    - Verify credentials in `$HOME/.codex/auth.json`
4.  **Codex Response API Errors:**
    - "Instruction prompt injection failed": The backend requires the Codex prompt; this is automatic
    - "Session not found": Use persistent session IDs for conversation continuity
    - "Model not available": Ensure you're using ChatGPT Plus compatible models
5.  **Missing API Auth Token:** If you've enabled security, include the token in your request headers.
6.  **Port Already in Use:** Start the server on a different port: `ccproxy --port 8001`.
7.  **Model Not Available:** Check that your subscription includes the requested model.

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
