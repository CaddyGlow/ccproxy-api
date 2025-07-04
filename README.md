# Claude Code Proxy API Server

A personal API proxy server that enables you to use your existing Claude subscription through familiar API interfaces. This tool runs locally on your computer and leverages Claude OAuth2 authentication, allowing you to use your Claude subscription without paying for separate API access.

## Why Use This Tool?

### Personal Access to Your Claude Subscription
- **Use Your Existing Subscription**: Leverage your Claude Pro or Team subscription instead of paying for API access
- **OAuth2 Authentication**: Uses your existing Claude account authentication
- **Local Execution**: Runs securely on your personal computer
- **Docker Isolation**: Optional Docker support for isolated Claude Code execution

### Developer-Friendly API Access
- **Dual API Compatibility**: Full support for both Anthropic and OpenAI API formats
- **Existing Tool Integration**: Drop-in replacement for applications expecting OpenAI or Anthropic APIs
- **Streaming Support**: Real-time response streaming for both API formats
- **Local Development**: Perfect for personal projects and local development

## Features

### Core Capabilities
- **Claude OAuth2 Integration**: Uses your Claude account authentication through Claude Code SDK
- **Personal API Server**: Runs locally on your computer (localhost)
- **Dual API Compatibility**: Supports both Anthropic and OpenAI API formats
- **Request Translation**: Seamless format conversion between API types
- **Streaming Support**: Real-time response streaming
- **Docker Support**: Optional containerized execution for better isolation

### Security & Privacy
- **Local Execution**: All processing happens on your computer
- **No API Keys Required**: Uses your existing Claude subscription via OAuth2
- **Secure Authentication**: Leverages Claude's official authentication system
- **Optional Isolation**: Docker support for sandboxed Claude Code execution

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Claude account with an active subscription (Pro, Team, or Enterprise)
- Claude Code CLI (will be set up automatically)

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

3. Optional: Configure environment variables:
```bash
export PORT=8000  # Optional, defaults to 8000
export LOG_LEVEL=INFO  # Optional, defaults to INFO
```

### Running Your Personal Proxy

```bash
# Using uv
uv run python main.py

# Or directly with Python
python main.py
```

The proxy will start on `http://localhost:8000` and automatically handle Claude authentication.

## Using Your Personal Proxy

### With Your Existing Applications

Once running, you can point any application that uses OpenAI or Anthropic APIs to your local proxy:

**For OpenAI-compatible applications:**
```bash
# Set base URL to your local proxy
export OPENAI_BASE_URL="http://localhost:8000/openai/v1"
export OPENAI_API_KEY="dummy-key"  # Required by client libraries but not used
```

**For Anthropic-compatible applications:**
```bash
# Set base URL to your local proxy
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="dummy-key"  # Required by client libraries but not used
```

### API Endpoints

#### Anthropic-Compatible Endpoints

**Chat Completions:**
```http
POST /v1/chat/completions
```

**Example request:**
```json
{
  "model": "claude-sonnet-4-20250514",
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

#### OpenAI-Compatible Endpoints

**Chat Completions:**
```http
POST /openai/v1/chat/completions
```

Uses OpenAI format with automatic translation to Claude format.

#### Utility Endpoints

**Health Check:**
```http
GET /health
```

**Available Models:**
```http
GET /v1/models
GET /openai/v1/models
```

### Supported Models

Your proxy supports all Claude models available to your subscription:

| Model | Description |
|-------|-------------|
| `claude-opus-4-20250514` | Claude 4 Opus (most capable) |
| `claude-sonnet-4-20250514` | Claude 4 Sonnet (latest, recommended) |
| `claude-3-7-sonnet-20250219` | Claude 3.7 Sonnet (enhanced) |
| `claude-3-5-sonnet-20241022` | Claude 3.5 Sonnet (stable) |
| `claude-3-5-sonnet-20240620` | Claude 3.5 Sonnet (legacy) |

*Available models depend on your Claude subscription level.*

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Local server port | `8000` |
| `HOST` | Server host (keep as localhost for security) | `127.0.0.1` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Claude Authentication

The proxy automatically detects and uses Claude Code CLI for authentication:
- First run will prompt for Claude login
- Authentication is cached for subsequent uses
- Works with all Claude subscription types

### Optional Docker Isolation

For enhanced security and isolation when running Claude Code:

```bash
# Build with Docker support
docker build -t claude-proxy .

# Run with Docker isolation
docker run -p 8000:8000 -v ~/.claude:/root/.claude claude-proxy
```

## Usage Examples

### Python with OpenAI Client

```python
from openai import OpenAI

# Point to your local proxy
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy-key"  # Required but not used
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Python with Anthropic Client

```python
from anthropic import Anthropic

# Point to your local proxy
client = Anthropic(
    base_url="http://localhost:8000",
    api_key="dummy-key"  # Required but not used
)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

### Streaming Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy-key"
)

stream = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### curl Example

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Development

### Setup Development Environment

```bash
# Using devenv (recommended)
devenv shell

# Or using uv
uv sync --group dev
```

### Code Quality

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

## Troubleshooting

### Common Issues

1. **Claude Authentication**
   ```
   Error: Claude not authenticated
   Solution: Run `claude auth login` or restart the proxy to trigger auth flow
   ```

2. **Port Already in Use**
   ```
   Error: Port 8000 already in use
   Solution: Use a different port: PORT=8001 python main.py
   ```

3. **Subscription Access**
   ```
   Error: Model not available
   Solution: Check that your Claude subscription includes the requested model
   ```

### Debug Mode

Enable debug logging for troubleshooting:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

## Privacy & Security

- **Local Only**: The proxy runs entirely on your computer
- **No External APIs**: Uses your existing Claude subscription, no additional API costs
- **Secure Authentication**: Uses Claude's official OAuth2 flow
- **Optional Isolation**: Docker support for sandboxed execution
- **No Data Logging**: Conversations are not stored or logged by the proxy

## Limitations

- **Personal Use Only**: Designed for individual use, not multi-user scenarios
- **Subscription Required**: Requires an active Claude subscription
- **Local Access**: Accessible only from your computer (for security)
- **Model Availability**: Limited to models available in your subscription

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

## Documentation

Comprehensive documentation is available:

- **[Online Documentation](https://your-username.github.io/claude-proxy)** - Full documentation site
- **[API Reference](https://your-username.github.io/claude-proxy/api-reference/overview/)** - Complete API documentation
- **[Developer Guide](https://your-username.github.io/claude-proxy/developer-guide/architecture/)** - Architecture and development

### Building Documentation Locally

```bash
# Install documentation dependencies
make docs-install

# Serve documentation locally with live reload
make docs-serve

# Build static documentation
make docs-build
```

## Support

- Issues: [GitHub Issues](https://github.com/your-username/claude-proxy/issues)
- Documentation: [Project Documentation](https://your-username.github.io/claude-proxy)

## Acknowledgments

- [Anthropic](https://anthropic.com) for Claude and the Claude Code SDK
- [claude-code-sdk](https://github.com/anthropics/claude-code-sdk) for the Python SDK
- The open-source community for inspiration and contributions