# Claude Proxy API Server

A high-performance API server that provides an Anthropic-compatible interface for Claude AI models, forwarding requests to Claude using the official Python SDK.

## Features

- 🚀 **Anthropic API Compatible**: Drop-in replacement for Anthropic's API endpoints
- 🔄 **Request Forwarding**: Seamlessly forwards requests to Claude using the official Python SDK
- 🛡️ **Type Safety**: Built with strict TypeScript-like typing using mypy
- ⚡ **High Performance**: Optimized for low latency and high throughput
- 🔧 **Easy Setup**: Simple configuration and deployment
- 📊 **Monitoring**: Built-in request/response logging and metrics
- 🔐 **Secure**: Proper API key management and request validation

## Quick Start

### Prerequisites

- Python 3.11 or higher
- An Anthropic API key

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

3. Set up your environment variables:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
export PORT=8000  # Optional, defaults to 8000
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

### Chat Completions

```http
POST /v1/chat/completions
```

**Request Body:**
```json
{
  "model": "claude-3-sonnet-20240229",
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
  "model": "claude-3-sonnet-20240229",
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

### Streaming Support

Add `"stream": true` to your request for streaming responses:

```json
{
  "model": "claude-3-sonnet-20240229",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}
```

### Supported Models

- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required |
| `PORT` | Server port | `8000` |
| `HOST` | Server host | `0.0.0.0` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_TOKENS_LIMIT` | Maximum tokens per request | `4096` |
| `RATE_LIMIT_REQUESTS` | Requests per minute limit | `60` |

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
    "default_model": "claude-3-sonnet-20240229",
    "max_tokens": 4096,
    "timeout": 30
  },
  "logging": {
    "level": "INFO",
    "format": "json"
  }
}
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
docker build -t claude-proxy .
```

### Run Container

```bash
docker run -d \
  --name claude-proxy \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your-api-key \
  claude-proxy
```

### Docker Compose

```yaml
version: '3.8'
services:
  claude-proxy:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
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

Since this proxy is Anthropic API compatible, you can use it with OpenAI's Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Not used but required by OpenAI client
)

response = client.chat.completions.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
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

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/claude-proxy/issues)
- 📖 Documentation: [Wiki](https://github.com/your-username/claude-proxy/wiki)

## Acknowledgments

- [Anthropic](https://anthropic.com) for the Claude API
- [claude-code-sdk](https://github.com/anthropics/claude-code-sdk) for the Python SDK
- The open-source community for inspiration and contributions

