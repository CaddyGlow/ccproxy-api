# Quick Start Guide

Get up and running with Claude Code Proxy API on your local machine in minutes.

## The `ccproxy` Command

The `ccproxy` command is your unified interface for Claude Code Proxy API:

```bash
# Run Claude commands locally
ccproxy claude -- /status

# Run Claude commands in Docker (isolated environment)  
ccproxy claude --docker -- /status
```

**How it works:**
- **Unified Interface**: Same command syntax for both local and Docker execution
- **Claude CLI Passthrough**: Forwards all Claude CLI commands and flags seamlessly
- **Automatic Docker Management**: Handles container lifecycle when using `--docker` flag
- **Isolated Configuration**: Docker mode uses separate config at `~/.config/cc-proxy/home`
- **Workspace Mapping**: Working directory remains consistent between local and Docker execution

## API Server Commands

Choose the right command based on your use case:

### `ccproxy api` - Production Ready
```bash
# Production server locally
ccproxy api

# Production server with Docker
ccproxy api --docker --port 8080
```
**Use for**: Production deployments, maximum stability and performance.

### `ccproxy run` - Balanced Development  
```bash
# Development server locally
ccproxy run

# Development server with reload
ccproxy run --reload --port 8080
```
**Use for**: General development work, testing, and debugging.

### `ccproxy dev` - Full Development Features
```bash
# Full development mode
ccproxy dev

# Development with all features
ccproxy dev --reload --log-level DEBUG
```
**Use for**: Active development, hot-reload, detailed logging.

## Prerequisites

Before starting, ensure you have:

- **Python 3.11 or higher**
- **Claude Code CLI** installed and authenticated
- **Claude subscription** (personal or professional account)
- **Git** for cloning the repository
- **Docker** (optional, recommended for isolation)

### Claude Code CLI Setup

The proxy requires Claude Code CLI to be available, either installed locally or via Docker.

#### Option 1: Local Installation

Install Claude Code CLI following the [official instructions](https://docs.anthropic.com/en/docs/claude-code).

**Authentication:**

CCProxy uses two separate authentication systems:

**Claude CLI (for Claude Code mode):**
```bash
# Login to Claude CLI (opens browser)
claude /login

# Verify Claude CLI status
claude /status
```
- Credentials stored at: `~/.claude/credentials.json` or `~/.config/claude/credentials.json`

**CCProxy (for API mode):**
```bash
# For API/raw mode authentication (uses Anthropic OAuth2)
ccproxy auth login

# Check ccproxy auth status
ccproxy auth validate

# Get detailed credential info
ccproxy auth info
```
- Credentials stored in system keyring (secure)
- Fallback to: `~/.config/ccproxy/credentials.json`

**Verification:**
```bash
# Test Claude CLI integration
ccproxy claude -- /status
```

#### Option 2: Docker (Recommended)

Docker users don't need to install Claude CLI locally - it's included in the Docker image.

**Docker Volume Configuration:**
- **Claude Home**: `~/.config/cc-proxy/home` (isolated from your local Claude config)
- **Working Directory**: Current user path (same as local execution)
- **Custom Path**: Override with environment variables if needed

**Authentication:**

**Claude CLI in Docker (for Claude Code mode):**
```bash
# Authenticate Claude CLI in Docker (first time setup)
ccproxy claude --docker -- /login
```
- Docker uses isolated config at: `~/.config/cc-proxy/home`

**CCProxy (for API mode):**
```bash
# For API/raw mode authentication (uses Anthropic OAuth2)
ccproxy auth login
```
- Credentials stored in system keyring (secure)
- Fallback to: `~/.config/ccproxy/credentials.json`

**Verification:**
```bash
# Test Docker Claude CLI
ccproxy claude --docker -- /status
```

**Expected output for both options:**
```
Executing: /path/to/claude /status

╭─────────────────────────────────────────────────────────╮
│ ✻ Welcome to Claude Code!                               │
│                                                         │
│   /help for help, /status for your current setup        │
╰─────────────────────────────────────────────────────────╯

 Claude Code Status v1.0.43

 Account • /login
  L Login Method: Claude Max Account
  L Organization: your-email@example.com's Organization
  L Email: your-email@example.com

 Model • /model
  L sonnet (claude-sonnet-4-20250514)
```

If you see authentication errors, refer to the [troubleshooting section](#claude-cli-not-found) below.

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/CaddyGlow/claude-code-proxy-api.git
cd claude-code-proxy-api

# Install dependencies using uv
uv sync

# Install documentation dependencies (optional)
uv sync --group docs
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/CaddyGlow/claude-code-proxy-api.git
cd claude-code-proxy-api

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Option 3: Docker (Recommended for Security)

Docker provides isolation and security for Claude Code execution on your local machine:

```bash
# Pull the Docker image
docker pull claude-code-proxy-api

# Or build locally
docker build -t claude-code-proxy-api .
```

## Running the Server

### Local Development

```bash
# Using uv (recommended)
uv run python main.py

# Or directly with Python
python main.py

# With custom port and log level
PORT=8080 LOG_LEVEL=DEBUG uv run python main.py
```

### Docker (Isolated Execution)

Run Claude Code Proxy in a secure, isolated container with proper volume mapping:

```bash
# Run with Docker (for secure local execution)
docker run -d \
  --name claude-code-proxy-api \
  -p 8000:8000 \
  -v ~/.config/cc-proxy/home:/data/home \
  -v $(pwd):/data/workspace \
  claude-code-proxy-api

# With custom settings and working directory
docker run -d \
  --name claude-code-proxy-api \
  -p 8080:8000 \
  -e PORT=8000 \
  -e LOG_LEVEL=INFO \
  -v ~/.config/cc-proxy/home:/data/home \
  -v /path/to/your/workspace:/data/workspace \
  claude-code-proxy-api
```

## Docker Configuration Summary

### 📁 **Volume Mappings**

| Host Path | Container Path | Purpose | Required |
|-----------|---------------|---------|----------|
| `~/.config/cc-proxy/home` | `/data/home` | **Claude Home**: Isolated Claude config & cache | **Required** |
| `$(pwd)` or custom path | `/data/workspace` | **Workspace**: Working directory for Claude operations | **Required** |

**Volume Details:**

- **`/data/home`** (CLAUDE_HOME):
  - Stores Claude CLI configuration, authentication, and cache
  - **Isolated** from your local `~/.claude` directory
  - Contains: `.config/`, `.cache/`, `.local/` subdirectories
  - **Persists** authentication between container restarts

- **`/data/workspace`** (CLAUDE_WORKSPACE):
  - Active working directory where Claude operates
  - **Maps to** your project directory or any custom path
  - Claude reads/writes files relative to this directory
  - Should contain your code projects

### 🔧 **Environment Variables**

| Variable | Default | Purpose | Docker Support |
|----------|---------|---------|----------------|
| `HOST` | `0.0.0.0` | Server bind address | ✅ Built-in |
| `PORT` | `8000` | Server port | ✅ Built-in |
| `LOG_LEVEL` | `INFO` | Logging verbosity | ✅ Built-in |
| `PUID` | `1000` | User ID for file permissions | ✅ Docker only |
| `PGID` | `1000` | Group ID for file permissions | ✅ Docker only |
| `CLAUDE_HOME` | `/data/home` | Claude config directory | ✅ Docker only |
| `CLAUDE_WORKSPACE` | `/data/workspace` | Claude working directory | ✅ Docker only |

**Docker-Specific Variables:**

- **`PUID`/`PGID`**: Ensures files created in volumes have correct ownership
- **`CLAUDE_HOME`**: Overrides default Claude home directory
- **`CLAUDE_WORKSPACE`**: Sets Claude's working directory

### 🛡️ **Security & Isolation Benefits**

This Docker setup provides:

- **Isolated Configuration**: Docker Claude config separate from local installation
- **File Permission Management**: Proper ownership of created files via PUID/PGID
- **Working Directory Control**: Claude operates in mapped workspace only
- **Container Security**: Claude CLI runs in isolated container environment
- **No Local Installation**: Claude CLI included in Docker image

### 📋 **Quick Setup Commands**

```bash
# Create required directories
mkdir -p ~/.config/cc-proxy/home

# Run with automatic volume setup
docker run -d \
  --name claude-code-proxy \
  -p 8000:8000 \
  -e PUID=$(id -u) \
  -e PGID=$(id -g) \
  -v ~/.config/cc-proxy/home:/data/home \
  -v $(pwd):/data/workspace \
  ghcr.io/caddyglow/claude-code-proxy-api

# First-time authentication
docker exec -it claude-code-proxy ccproxy claude -- auth login

# Verify setup
docker exec -it claude-code-proxy ccproxy claude -- /status
```

### Docker Compose (Recommended)

Complete Docker Compose setup with proper configuration:

```yaml
version: '3.8'
services:
  claude-code-proxy:
    image: ghcr.io/caddyglow/claude-code-proxy-api:latest
    container_name: claude-code-proxy
    ports:
      - "8000:8000"
    environment:
      # Server Configuration
      - HOST=0.0.0.0
      - PORT=8000
      - LOG_LEVEL=INFO

      # File Permissions (matches your user)
      - PUID=${PUID:-1000}
      - PGID=${PGID:-1000}

      # Docker Paths (pre-configured)
      - CLAUDE_HOME=/data/home
      - CLAUDE_WORKSPACE=/data/workspace
    volumes:
      # Claude config & auth (isolated)
      - ~/.config/cc-proxy/home:/data/home
      # Your workspace (current directory)
      - .:/data/workspace
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
```

**Setup Commands:**
```bash
# Create Docker Compose file (save as docker-compose.yml)
# Set your user ID (optional, defaults to 1000)
export PUID=$(id -u)
export PGID=$(id -g)

# Start the service
docker-compose up -d

# First-time authentication
docker-compose exec claude-code-proxy ccproxy claude -- auth login

# Verify setup
docker-compose exec claude-code-proxy ccproxy claude -- /status

# View logs
docker-compose logs -f claude-code-proxy
```

## First API Call

Once the server is running, test it with a simple API call:

### OAuth Users (Claude Subscription)

OAuth users must use full mode (default):

```bash
# Using curl
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {
        "role": "user",
        "content": "Hello! Can you help me test this API?"
      }
    ],
    "max_tokens": 100
  }'
```

### API Key Users

API key users can use any mode:

```bash
# Full mode (with Claude Code features)
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-..." \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Minimal mode (lightweight)
curl -X POST http://localhost:8000/min/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-..." \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Using Python

```python
from anthropic import Anthropic

# OAuth users (Claude subscription) - full mode
client = Anthropic(
    base_url="http://localhost:8000",
    api_key="dummy"  # Ignored with OAuth
)

# API key users - any mode
client = Anthropic(
    base_url="http://localhost:8000/min",  # Minimal mode
    api_key="sk-ant-api03-..."
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.content[0].text)
```

### Using OpenAI Python Client

```python
from openai import OpenAI

# OAuth users - must use full mode
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy"  # Ignored with OAuth
)

# API key users - can use any mode
client = OpenAI(
    base_url="http://localhost:8000/min/openai/v1",  # Minimal mode
    api_key="sk-ant-api03-..."
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

## Health Check

Verify the server is running properly:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "claude_cli_available": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Available Models

Check available models:

```bash
curl http://localhost:8000/v1/models
```

## Proxy Modes

The proxy supports three transformation modes:

| Mode | URL Prefix | Authentication | Use Case |
|------|------------|----------------|----------|
| Full | `/` or `/full/` | OAuth, API Key | Claude Code features, OAuth users |
| Minimal | `/min/` | API Key only | Lightweight proxy |
| Passthrough | `/pt/` | API Key only | Direct API access |

**Important**: OAuth credentials from Claude Code only work with full mode. Using `/min` or `/pt` with OAuth will result in an authentication error.

For detailed information about proxy modes, see the [Proxy Modes Guide](../user-guide/proxy-modes.md).

## Using with Aider

CCProxy works seamlessly with Aider and other AI coding assistants:

### Anthropic Mode
```bash
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000/
aider --model claude-sonnet-4-20250514
```

### OpenAI Mode with Model Mapping
If your tool only supports OpenAI settings, ccproxy automatically maps OpenAI models to Claude:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/cc/openai/v1
aider --model o3-mini
```

**Model Mapping:**
```python
OPENAI_TO_CLAUDE_MODEL_MAPPING = {
    "gpt-4o-mini": "claude-3-5-haiku-latest",
    "o3-mini": "claude-opus-4-20250514",
    "o1-mini": "claude-sonnet-4-20250514",
    "gpt-4o": "claude-3-7-sonnet-20250219",
}
```

### API Mode (Direct Proxy)
For minimal interference and direct API access:

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8000/api/openai/v1
aider --model o3-mini
```

## Next Steps

Now that you have the server running locally:

1. **[Configure the server](configuration.md)** with your personal preferences
2. **[Explore the API](../api-reference/overview.md)** to understand all available endpoints
3. **[Try examples](../examples/python-client.md)** in different programming languages
4. **[Set up Docker isolation](../deployment/overview.md)** for enhanced security
5. **[Learn about proxy modes](../user-guide/proxy-modes.md)** to choose the right mode for your use case

## Troubleshooting

### Server won't start

1. Check Python version: `python --version` (should be 3.11+)
2. Verify dependencies: `uv sync` or `pip install -e .`
3. Check port availability: `netstat -an | grep 8000`

### Claude CLI not found

**For Local Installation:**

1. **Install Claude CLI** following [official instructions](https://docs.anthropic.com/en/docs/claude-code)
2. **Verify installation**: `claude --version`
3. **Test authentication**: `claude auth login`
4. **Verify proxy detection**: `ccproxy claude -- /status`
5. **Set custom path** (if needed): `export CLAUDE_CLI_PATH=/path/to/claude`

**For Docker Users:**

1. **No local installation needed** - Claude CLI is included in Docker image
2. **Test Docker Claude**: `ccproxy claude --docker -- /status`
3. **Check volume mapping**: Ensure `~/.config/cc-proxy/home` directory exists
4. **Verify workspace**: Check that workspace volume is properly mounted

### Claude authentication issues

**For Local Installation:**

If `ccproxy claude -- /status` shows authentication errors:

1. **Re-authenticate**: `claude auth login`
2. **Check account status**: `claude /status`
3. **Verify subscription**: Ensure your Claude account has an active subscription
4. **Check permissions**: Ensure Claude CLI has proper permissions to access your account

**For Docker Users:**

If `ccproxy claude --docker -- /status` shows authentication errors:

1. **Authenticate in Docker**: `ccproxy claude --docker -- auth login`
2. **Check Docker volumes**: Verify `~/.config/cc-proxy/home` is properly mounted
3. **Verify isolated config**: Docker uses separate config from your local Claude installation
4. **Check container permissions**: Ensure Docker container has proper file permissions

### Expected ccproxy output

When running `ccproxy claude -- /status` or `ccproxy claude --docker -- /status`, you should see:

- **Executing**: Shows the Claude CLI path being used (local or Docker)
- **Welcome message**: Confirms Claude CLI is working
- **Account info**: Shows your authentication status
- **Model info**: Displays available model
- **Working Directory**: Shows correct workspace path

If any of these are missing, review the Claude CLI setup steps above.

### API calls fail

1. Check server logs for errors
2. Verify the server is running: `curl http://localhost:8000/health`
3. Test with simple curl command first
4. Check network connectivity

For more troubleshooting tips, see the [Developer Guide](../developer-guide/development.md#troubleshooting).
