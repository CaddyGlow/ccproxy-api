# Quick Start Guide

Get up and running with Claude Code Proxy API on your local machine in minutes.

## Prerequisites

Before starting, ensure you have:

- **Python 3.11 or higher**
- **Claude subscription** (personal or professional account)
- **Git** for cloning the repository
- **Docker** (optional, recommended for isolation)

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/claude-proxy.git
cd claude-proxy

# Install dependencies using uv
uv sync

# Install documentation dependencies (optional)
uv sync --group docs
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/claude-proxy.git
cd claude-proxy

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Option 3: Docker (Recommended for Security)

Docker provides isolation and security for Claude Code execution on your local machine:

```bash
# Pull the Docker image
docker pull claude-code-proxy

# Or build locally
docker build -t claude-code-proxy .
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

Run Claude Code Proxy in a secure, isolated container:

```bash
# Run with Docker (for secure local execution)
docker run -d \
  --name claude-proxy \
  -p 8000:8000 \
  -v ~/.config/claude:/root/.config/claude:ro \
  claude-code-proxy

# With custom settings
docker run -d \
  --name claude-proxy \
  -p 8080:8000 \
  -e PORT=8000 \
  -e LOG_LEVEL=INFO \
  -v ~/.config/claude:/root/.config/claude:ro \
  claude-code-proxy
```

### Docker Compose (Personal Setup)

```yaml
version: '3.8'
services:
  claude-proxy:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - PORT=8000
    volumes:
      - ~/.config/claude:/root/.config/claude:ro
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## First API Call

Once the server is running, test it with a simple API call:

### Using curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
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

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)

print(response.json())
```

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Not used but required by OpenAI client
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

## Next Steps

Now that you have the server running locally:

1. **[Configure the server](configuration.md)** with your personal preferences
2. **[Explore the API](../api-reference/overview.md)** to understand all available endpoints
3. **[Try examples](../examples/python-client.md)** in different programming languages
4. **[Set up Docker isolation](../deployment.md)** for enhanced security

## Troubleshooting

### Server won't start

1. Check Python version: `python --version` (should be 3.11+)
2. Verify dependencies: `uv sync` or `pip install -e .`
3. Check port availability: `netstat -an | grep 8000`

### Claude CLI not found

1. Install Claude CLI following [official instructions](https://docs.anthropic.com/en/docs/claude-code)
2. Verify installation: `claude --version`
3. Set custom path: `export CLAUDE_CLI_PATH=/path/to/claude`

### API calls fail

1. Check server logs for errors
2. Verify the server is running: `curl http://localhost:8000/health`
3. Test with simple curl command first
4. Check network connectivity

For more troubleshooting tips, see the [Developer Guide](../developer-guide/development.md#troubleshooting).