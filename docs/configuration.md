# Configuration Reference

## Overview

The Claude Code Proxy API Server supports flexible configuration through multiple sources:

1. **Environment Variables** (highest priority)
2. **Configuration Files** (.env files)
3. **JSON Configuration** (config.json)
4. **Default Values** (lowest priority)

Configuration is managed using Pydantic Settings with automatic validation and type conversion.

## Configuration Sources

### Priority Order

Configuration sources are loaded in the following priority order (higher number = higher priority):

1. Default values (defined in code)
2. JSON configuration file
3. .env file
4. Environment variables (highest priority)

### Environment Variables

Environment variables override all other configuration sources:

```bash
# Basic server configuration
export HOST=0.0.0.0
export PORT=8000
export LOG_LEVEL=INFO

# Claude configuration
export CLAUDE_CLI_PATH=/usr/local/bin/claude

# Security settings
export CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### .env File

Create a `.env` file in the project root:

```bash
# .env
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
WORKERS=4
RELOAD=false

# Claude configuration
CLAUDE_CLI_PATH=/usr/local/bin/claude

# Security
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Tools handling
TOOLS_HANDLING=warning
```

### JSON Configuration

Create a `config.json` file for advanced configuration:

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "log_level": "INFO",
  "workers": 4,
  "reload": false,
  "cors_origins": ["https://yourdomain.com"],
  "claude_cli_path": "/usr/local/bin/claude",
  "tools_handling": "warning",
  "docker_settings": {
    "docker_image": "claude-code-proxy:latest",
    "docker_volumes": [
      "$HOME/.config/claude:/data/home",
      "$PWD:/data/workspace"
    ],
    "docker_environment": {
      "CLAUDE_HOME": "/data/home",
      "CLAUDE_WORKSPACE": "/data/workspace"
    }
  }
}
```

## Configuration Options

### Server Configuration

#### Host and Port

```bash
# Server host address (default: 0.0.0.0)
HOST=0.0.0.0

# Server port number (default: 8000)
PORT=8000
```

**Type**: `string` / `integer`
**Default**: `0.0.0.0` / `8000`
**Description**: Server binding address and port

#### Workers and Performance

```bash
# Number of worker processes (default: 1)
WORKERS=4

# Enable auto-reload for development (default: false)
RELOAD=true
```

**Type**: `integer` / `boolean`
**Constraints**: Workers: 1-32, Reload: true/false
**Description**: Worker process configuration and development settings

### Logging Configuration

#### Log Level

```bash
# Logging level (default: INFO)
LOG_LEVEL=DEBUG
```

**Type**: `string`
**Valid Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
**Default**: `INFO`
**Description**: Application logging verbosity

#### Log Format Examples

```python
# Development logging
LOG_LEVEL=DEBUG
# Output: 2024-01-01 12:00:00 - claude_code_proxy.main - DEBUG - Detailed debug info

# Production logging
LOG_LEVEL=INFO
# Output: 2024-01-01 12:00:00 - claude_code_proxy.main - INFO - Request processed successfully
```

### Claude CLI Configuration

#### Claude CLI Path

```bash
# Explicit Claude CLI path
CLAUDE_CLI_PATH=/usr/local/bin/claude

# Auto-detection (default)
# CLAUDE_CLI_PATH=  # Leave empty for auto-detection
```

**Type**: `string` (optional)
**Default**: Auto-detection
**Description**: Path to Claude CLI executable

#### Auto-detection Paths

When `CLAUDE_CLI_PATH` is not set, the system searches these locations in order:

1. **PATH environment variable** - `which claude`
2. **User-specific installation** - `~/.claude/local/claude`
3. **Global npm installation** - `~/node_modules/.bin/claude`
4. **Package node_modules** - `./node_modules/.bin/claude`
5. **Current directory** - `./node_modules/.bin/claude`
6. **System-wide installations**:
   - `/usr/local/bin/claude`
   - `/opt/homebrew/bin/claude`

#### Claude Code SDK Options

```json
{
  "claude_code_options": {
    "cwd": "/path/to/working/directory",
    "model": "claude-3-5-sonnet-20241022",
    "max_thinking_tokens": 30000
  }
}
```

### Security Configuration

#### CORS Settings

```bash
# Single origin
CORS_ORIGINS=https://yourdomain.com

# Multiple origins (comma-separated)
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com,http://localhost:3000

# Allow all origins (development only)
CORS_ORIGINS=*
```

**Type**: `string` or `list[string]`
**Default**: `["*"]`
**Description**: Cross-Origin Resource Sharing allowed origins

#### Security Headers

The application automatically sets security headers:

- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`

#### User and Group Settings

```bash
# Security settings for subprocess execution
CLAUDE_USER=claude
CLAUDE_GROUP=claude
```

**Type**: `string` (optional)
**Default**: `claude` / `claude`
**Description**: User/group for dropping privileges when executing Claude subprocess

### Request Handling Configuration

#### Tools Handling

```bash
# How to handle tools definitions in requests
TOOLS_HANDLING=warning
```

**Type**: `string`
**Valid Values**: `error`, `warning`, `ignore`
**Default**: `warning`
**Description**: Behavior when tools are defined in requests

**Options**:
- `error`: Return error if tools are present
- `warning`: Log warning but continue processing
- `ignore`: Silently ignore tools definitions

### Docker Configuration

#### Docker Settings Structure

```json
{
  "docker_settings": {
    "docker_image": "claude-code-proxy:latest",
    "docker_volumes": [
      "$HOME/.config/claude:/data/home",
      "$PWD:/data/workspace"
    ],
    "docker_environment": {
      "CLAUDE_HOME": "/data/home",
      "CLAUDE_WORKSPACE": "/data/workspace"
    },
    "docker_additional_args": ["--network=host"],
    "docker_home_directory": "/home/user/.config/claude",
    "docker_workspace_directory": "/home/user/projects"
  }
}
```

#### Docker Image

```bash
# Docker image name (default: claude-code-proxy)
DOCKER_IMAGE=claude-code-proxy:latest
```

#### Volume Mounts

```bash
# Volume mounts (host:container[:options] format)
DOCKER_VOLUMES="$HOME/.config/claude:/data/home,$PWD:/data/workspace"
```

**Format**: `host_path:container_path[:options]`
**Example**: `/home/user/.config/claude:/data/home:ro`

#### Environment Variables for Docker

```bash
# Environment variables passed to Docker container
DOCKER_ENVIRONMENT='{"CLAUDE_HOME":"/data/home","CLAUDE_WORKSPACE":"/data/workspace"}'
```

#### Additional Docker Arguments

```bash
# Additional arguments for docker run command
DOCKER_ADDITIONAL_ARGS="--network=host,--security-opt=no-new-privileges"
```

## Configuration Validation

### Automatic Validation

All configuration values are automatically validated using Pydantic:

```python
# Port validation
port: int = Field(ge=1, le=65535)  # Must be 1-65535

# Log level validation
log_level: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

# CORS origins validation
cors_origins: list[str] = Field(min_items=1)
```

### Validation Errors

Configuration validation errors are reported with clear messages:

```bash
# Example validation error
Configuration error: Port must be between 1 and 65535, got 0

# Example file not found error
Configuration error: Claude CLI path does not exist: /invalid/path/claude
```

## Environment-specific Configuration

### Development Environment

Create `.env.development`:

```bash
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=DEBUG
RELOAD=true
WORKERS=1
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Staging Environment

Create `.env.staging`:

```bash
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
RELOAD=false
WORKERS=2
CORS_ORIGINS=https://staging.yourdomain.com
CLAUDE_CLI_PATH=/usr/local/bin/claude
```

### Production Environment

Create `.env.production`:

```bash
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
RELOAD=false
WORKERS=4
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
CLAUDE_CLI_PATH=/usr/local/bin/claude
TOOLS_HANDLING=error
```

## Configuration Management

### Loading Configuration

```python
from claude_code_proxy.config.settings import get_settings

# Get current configuration
settings = get_settings()

# Access configuration values
print(f"Server running on {settings.host}:{settings.port}")
print(f"Log level: {settings.log_level}")
print(f"Claude CLI: {settings.claude_cli_path}")
```

### Runtime Configuration

```python
# Check if running in development mode
if settings.is_development:
    print("Running in development mode")

# Get complete server URL
print(f"Server URL: {settings.server_url}")

# Get searched paths for Claude CLI
for path in settings.get_searched_paths():
    print(f"Searching: {path}")
```

### Configuration Inspection

```python
# Get safe configuration dump (sensitive data masked)
config_dict = settings.model_dump_safe()

# Print configuration in JSON format
import json
print(json.dumps(config_dict, indent=2))
```

## CLI Configuration Commands

### View Current Configuration

```bash
# Display current configuration
ccproxy config

# Output example:
# Current Configuration:
#   Host: 0.0.0.0
#   Port: 8000
#   Log Level: INFO
#   Claude CLI Path: /usr/local/bin/claude
#   Workers: 4
#   Reload: false
```

### Test Claude CLI

```bash
# Test Claude CLI integration
ccproxy claude -- --version

# Test with Docker
ccproxy claude --docker -- --version
```

## Advanced Configuration

### Custom Configuration File Path

```bash
# Specify custom configuration file
CONFIG_FILE=/path/to/custom/config.json ccproxy run
```

### Configuration Inheritance

Configuration files can reference environment variables:

```json
{
  "host": "${HOST:-0.0.0.0}",
  "port": "${PORT:-8000}",
  "claude_cli_path": "${CLAUDE_CLI_PATH}",
  "cors_origins": ["${CORS_ORIGIN:-*}"]
}
```

### Dynamic Configuration

```python
# Example: Load configuration from external source
import os
from claude_code_proxy.config.settings import Settings

# Override configuration programmatically
custom_settings = Settings(
    host=os.getenv("CUSTOM_HOST", "0.0.0.0"),
    port=int(os.getenv("CUSTOM_PORT", "8000")),
    log_level="DEBUG"
)
```

## Configuration Best Practices

### Security

1. **Never commit sensitive data** to version control
2. **Use environment variables** for secrets
3. **Validate file permissions** for configuration files
4. **Use HTTPS origins** in production CORS settings

### Performance

1. **Set appropriate worker count** (usually number of CPU cores)
2. **Disable reload** in production
3. **Use INFO log level** for production
4. **Configure proper CORS origins** (avoid wildcards in production)

### Maintenance

1. **Document environment-specific settings**
2. **Use consistent naming conventions**
3. **Validate configuration** before deployment
4. **Monitor configuration changes**

## Configuration Examples

### Minimal Configuration

```bash
# Minimal .env for local development
PORT=8000
LOG_LEVEL=DEBUG
```

### Complete Production Configuration

```bash
# .env.production
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
WORKERS=4
RELOAD=false

# Security
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com

# Claude
CLAUDE_CLI_PATH=/usr/local/bin/claude
TOOLS_HANDLING=error

# Security
CLAUDE_USER=claude
CLAUDE_GROUP=claude
```

### Docker Development Configuration

```bash
# .env.docker
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=DEBUG

# Docker settings
DOCKER_IMAGE=claude-code-proxy:dev
DOCKER_VOLUMES="$HOME/.config/claude:/data/home:ro,$PWD:/data/workspace"
DOCKER_ENVIRONMENT='{"CLAUDE_HOME":"/data/home","CLAUDE_WORKSPACE":"/data/workspace"}'
```

### Multi-environment Setup

```bash
# scripts/load-env.sh
#!/bin/bash

ENV=${1:-development}

case $ENV in
  "development")
    export $(cat .env.development | xargs)
    ;;
  "staging")
    export $(cat .env.staging | xargs)
    ;;
  "production")
    export $(cat .env.production | xargs)
    ;;
  *)
    echo "Unknown environment: $ENV"
    exit 1
    ;;
esac

echo "Loaded configuration for: $ENV"
```

## Troubleshooting Configuration

### Common Issues

1. **Invalid port number**: Check PORT value is 1-65535
2. **Claude CLI not found**: Verify CLAUDE_CLI_PATH or ensure Claude CLI is installed
3. **Permission denied**: Check file permissions for configuration files
4. **CORS errors**: Verify CORS_ORIGINS includes client domain

### Debugging Configuration

```bash
# Check current configuration
ccproxy config

# Test Claude CLI path
ccproxy claude -- --version

# Validate configuration file
python -c "from claude_code_proxy.config.settings import get_settings; print('Config valid')"
```

### Configuration Logs

Enable debug logging to see configuration loading:

```bash
LOG_LEVEL=DEBUG ccproxy run
```

Look for logs like:
```
DEBUG - Loading configuration from .env
DEBUG - Claude CLI found at: /usr/local/bin/claude
DEBUG - Configuration validation passed
```