# Claude Code Proxy API Server

A high-performance API server that provides both Anthropic and OpenAI-compatible interfaces for Claude AI models. This proxy enables you to use your Claude OAuth account or API access through familiar API endpoints, making it easy to integrate Claude into existing applications.

## Features

### Core Capabilities
- **Dual API Compatibility**: Full support for both Anthropic and OpenAI API formats
- **Streaming Support**: Real-time response streaming for both API formats
- **Request Translation**: Seamless format conversion between OpenAI and Anthropic formats
- **Claude CLI Integration**: Uses the official Claude Code Python SDK for authentication
- **Auto-detection**: Smart Claude CLI path resolution and configuration

### Production Features
- **Docker Support**: Production-ready containerization with multi-stage builds
- **Health Monitoring**: Built-in health checks and metrics endpoints
- **Error Handling**: Comprehensive error handling with detailed error responses
- **Rate Limiting**: Built-in protection against API abuse
- **CORS Support**: Cross-origin request handling for web applications
- **Structured Logging**: JSON-formatted logs for monitoring and debugging

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Claude Code SDK (authentication handled automatically)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/claude-proxy.git
cd claude-proxy
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. Optional environment variables:
```bash
export PORT=8000  # Optional, defaults to 8000
export LOG_LEVEL=INFO  # Optional, defaults to INFO
```

### Running the Server

```bash
# Using uv
uv run python main.py

# Or directly with Python
python main.py
```

The server will start on `http://localhost:8000` by default.

## API Endpoints

### Anthropic-Compatible Endpoints

#### Chat Completions
```http
POST /v1/chat/completions
```

**Request Body:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "claude-3-5-sonnet-20241022",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 20,
    "total_tokens": 32
  }
}
```

#### Models List
```http
GET /v1/models
```

### OpenAI-Compatible Endpoints

#### Chat Completions
```http
POST /openai/v1/chat/completions
```

Uses OpenAI format with automatic translation to Claude format.

#### Models List
```http
GET /openai/v1/models
```

### Health and Monitoring

#### Health Check
```http
GET /health
```

Returns server health status and Claude CLI availability.

### Streaming Support

Add `"stream": true` to your request for streaming responses:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}
```

### Supported Models

| Model | Description |
|-------|-------------|
| `claude-3-5-sonnet-20241022` | Latest Claude 3.5 Sonnet (recommended) |
| `claude-3-5-haiku-20241022` | Latest Claude 3.5 Haiku (fast) |
| `claude-3-opus-20240229` | Claude 3 Opus (most capable) |

Enterprise users can run Claude Code using models in existing Amazon Bedrock or Google Cloud Vertex AI instances.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `HOST` | Server host | `0.0.0.0` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Configuration File

Create a `config.json` file for advanced configuration:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "claude": {
    "default_model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
    "timeout": 30,
    "cli_path": "/path/to/claude"
  },
  "logging": {
    "level": "INFO",
    "format": "json"
  },
  "rate_limiting": {
    "requests_per_minute": 60,
    "burst_size": 10
  }
}
```

### Claude CLI Configuration

The proxy automatically detects Claude CLI installation in common locations:
- System PATH
- `~/.claude/local/claude`
- `~/node_modules/.bin/claude`
- Package node_modules
- Common system directories

You can also specify the path explicitly:
```bash
export CLAUDE_CLI_PATH=/path/to/claude
```

## Development

### Setup Development Environment

This project uses [devenv](https://devenv.sh/) for development environment management:

```bash
# Install devenv (if not already installed)
nix profile install --accept-flake-config github:cachix/devenv

# Enter development environment
devenv shell
```

### Code Quality

The project uses several tools for code quality:

- **Ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **pytest**: Testing framework

Run quality checks:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy .

# Run tests
pytest
```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t claude-code-proxy-api .
```

### Run Container

```bash
docker run -d \
  --name claude-code-proxy-api \
  -p 8000:8000 \
  claude-code-proxy-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  claude-code-proxy-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

## Usage Examples

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)
print(response.json())
```

### curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### OpenAI Python Client

Use the OpenAI Python client with Claude models:

```python
from openai import OpenAI

# For Anthropic endpoints
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Not used but required by OpenAI client
)

# For OpenAI-compatible endpoints
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Anthropic Python Client

Use the official Anthropic client:

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="dummy-key"  # Not used but required
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

### Streaming Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"
)

stream = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Monitoring and Logging

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

Access metrics at:
```
http://localhost:8000/metrics
```

### Logs

Logs are output in JSON format by default. Configure log level with `LOG_LEVEL` environment variable.

## Performance Tuning

### Concurrent Requests

The server handles concurrent requests efficiently. For high-traffic scenarios, consider:

1. **Horizontal Scaling**: Run multiple instances behind a load balancer
2. **Connection Pooling**: The SDK automatically manages connection pooling
3. **Caching**: Implement response caching for repeated requests

### Rate Limiting

Built-in rate limiting prevents API abuse:

```python
# Configure in config.json
{
  "rate_limiting": {
    "requests_per_minute": 60,
    "burst_size": 10
  }
}
```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```
   Error: Invalid API key
   Solution: Ensure ANTHROPIC_API_KEY is set correctly
   ```

2. **Connection Timeout**
   ```
   Error: Request timeout
   Solution: Increase timeout in configuration or check network connectivity
   ```

3. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   Solution: Implement exponential backoff in your client
   ```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Run code quality checks: `ruff check . && mypy .`
6. Commit your changes: `git commit -am 'Add your feature'`
7. Push to the branch: `git push origin feature/your-feature`
8. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Issues: [GitHub Issues](https://github.com/your-username/claude-proxy/issues)
- Documentation: [Project Documentation](docs/)

## Acknowledgments

- [Anthropic](https://anthropic.com) for the Claude API
- [claude-code-sdk](https://github.com/anthropics/claude-code-sdk) for the Python SDK
- The open-source community for inspiration and contributions

