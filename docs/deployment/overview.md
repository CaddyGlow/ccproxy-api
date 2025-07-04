# Docker Setup for Personal Use

## Overview

This guide covers setting up Claude Code Proxy with Docker for personal use, providing secure isolation for Claude Code execution on your local machine. Docker ensures that Claude Code operations run in a contained environment while maintaining easy access to your Claude subscription.

## Prerequisites

### System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Docker**: Docker Desktop or Docker Engine
- **Python**: 3.11 or higher (if building from source)
- **Memory**: Minimum 1GB RAM, recommended 2GB+
- **Disk**: Minimum 2GB free space
- **Network**: Internet access for Claude authentication and API calls

### Required Dependencies

- Docker (Docker Desktop recommended for personal use)
- Claude subscription (personal or professional account)
- Git (for cloning the repository)

## Docker Setup for Personal Use

### Using Pre-built Image

#### Basic Personal Setup

```bash
# Pull the latest image
docker pull claude-code-proxy:latest

# Run the container with your Claude configuration
docker run -d \
  --name claude-proxy \
  -p 127.0.0.1:8000:8000 \
  -e LOG_LEVEL=INFO \
  -v ~/.config/claude:/root/.config/claude:ro \
  claude-code-proxy:latest
```

#### Personal Docker Compose Setup

Create `docker-compose.yml` for local personal use:

```yaml
version: '3.8'

services:
  claude-proxy:
    image: claude-code-proxy:latest
    container_name: claude-proxy-personal
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"  # Bind to localhost only
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - LOG_LEVEL=INFO
      - WORKERS=2  # Fewer workers for personal use
    volumes:
      - ~/.config/claude:/root/.config/claude:ro  # Your Claude auth
      - ./logs:/app/logs                           # Local log storage
      - ./config:/app/config:ro                    # Optional custom config
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 256M
          cpus: '0.25'
```

### Building Custom Image for Personal Use

#### Personal Dockerfile

```dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && \
    uv sync --no-dev

FROM python:3.11-slim as runtime

# Install runtime dependencies for personal use
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user for security (personal container)
RUN useradd --create-home --shell /bin/bash claude

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY claude_code_proxy/ ./claude_code_proxy/
COPY entrypoint.sh ./

# Set permissions
RUN chmod +x entrypoint.sh && \
    chown -R claude:claude /app

# Switch to non-root user for security
USER claude

# Expose port (container internal)
EXPOSE 8000

# Health check for personal monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
ENTRYPOINT ["./entrypoint.sh"]
```

#### Personal Build Script

```bash
#!/bin/bash
# build-personal.sh

set -e

echo "Building Claude Code Proxy for personal use..."

# Build the image with personal tags
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  --build-arg VERSION=$(git describe --tags --always) \
  -t claude-code-proxy:personal \
  -t claude-code-proxy:latest \
  .

echo "Personal build completed successfully!"
echo "To run: docker-compose up -d"
```

### Container Configuration for Personal Use

#### Environment Variables for Local Setup

```bash
# Server Configuration (personal use)
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
WORKERS=2  # Fewer workers for personal machine

# Claude Configuration (your local CLI)
CLAUDE_CLI_PATH=/usr/local/bin/claude

# Security (local development)
CORS_ORIGINS=http://localhost:*,http://127.0.0.1:*

# Performance (personal use)
RELOAD=false
```

#### Volume Mounts for Personal Setup

```bash
# Your Claude CLI configuration (read-only for security)
-v ~/.config/claude:/root/.config/claude:ro

# Local application logs
-v ./logs:/app/logs

# Your custom configuration (optional)
-v ./config.json:/app/config.json:ro

# Local data directory (if needed)
-v ./data:/app/data
```

## Personal Authentication Setup

### Claude CLI Authentication

The Claude Code Proxy relies on your local Claude CLI authentication. Here's how to set it up:

#### Initial Claude CLI Setup

```bash
# Install Claude CLI (if not already installed)
# Follow official instructions at: https://docs.anthropic.com/en/docs/claude-code

# Authenticate with your Claude account
claude auth login

# Verify authentication
claude auth status
```

#### Docker Volume for Authentication

Your authentication credentials need to be available to the container:

```bash
# Check your Claude config location
ls -la ~/.config/claude/

# Mount it read-only in Docker
-v ~/.config/claude:/root/.config/claude:ro
```

## Security Considerations for Personal Use

### Local Network Security

#### Localhost Binding for Security

```bash
# Bind to localhost only (recommended for personal use)
docker run -p 127.0.0.1:8000:8000

# This restricts access to localhost only
# Prevents external network access to your proxy

# Alternative: Use host networking for development
docker run --network host  # Use with caution
```

#### Container Isolation Benefits

```bash
# Docker provides process isolation
docker run --rm -it claude-code-proxy:personal /bin/bash

# Resource limits for your container
docker run --memory=1g --cpus=1.0 claude-code-proxy:personal

# Read-only root filesystem for security
docker run --read-only claude-code-proxy:personal
```

### Data Privacy for Personal Use

#### Local Data Storage

```bash
# All data stays on your machine
-v ./local-data:/app/data

# Logs stored locally
-v ./logs:/app/logs

# No external databases or services required
```

#### Authentication Security

```bash
# Your Claude credentials stay local
-v ~/.config/claude:/root/.config/claude:ro

# No API keys stored in the container
# Uses your existing Claude subscription
```

## Personal Monitoring and Health Checks

### Health Check Endpoint

Monitor your local proxy with the built-in health endpoint:

```bash
# Quick health check
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "claude_cli_available": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Personal Monitoring Script

```bash
#!/bin/bash
# personal-health-check.sh

ENDPOINT="http://localhost:8000/health"
TIMEOUT=10

response=$(curl -s -w "%{http_code}" -o /dev/null --max-time $TIMEOUT "$ENDPOINT")

if [ "$response" = "200" ]; then
    echo "✓ Claude Proxy is healthy and running"
    exit 0
else
    echo "✗ Claude Proxy is not responding (HTTP $response)"
    exit 1
fi
```

### Container Resource Monitoring

```bash
# Monitor container resource usage
docker stats claude-proxy-personal

# View container logs
docker logs claude-proxy-personal

# Follow logs in real-time
docker logs -f claude-proxy-personal
```

## Backup and Maintenance for Personal Use

### Configuration Backup

```bash
#!/bin/bash
# backup-personal-config.sh

backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

# Backup your configuration files
cp docker-compose.yml "$backup_dir/"
cp config.json "$backup_dir/" 2>/dev/null || true
cp -r logs "$backup_dir/" 2>/dev/null || true

echo "Backup completed: $backup_dir"
```

### Regular Maintenance

```bash
# Update to latest image
docker pull claude-code-proxy:latest
docker-compose down
docker-compose up -d

# Clean up old containers and images
docker system prune -f

# View disk usage
docker system df
```

## Performance Tuning for Personal Use

### Container Resource Limits

```yaml
# In your docker-compose.yml
deploy:
  resources:
    limits:
      memory: 1G       # Adjust based on your system
      cpus: '1.0'      # Use 1 CPU core
    reservations:
      memory: 256M     # Minimum memory
      cpus: '0.25'     # Minimum CPU
```

### Environment Optimization

```bash
# Personal use environment variables
WORKERS=2                    # Fewer workers for personal machine
LOG_LEVEL=INFO              # Appropriate logging for personal use
RELOAD=false                # Disable auto-reload for stability
```

## Troubleshooting Personal Setup

### Common Personal Use Issues

1. **Container won't start**
   ```bash
   # Check Docker is running
   docker info
   
   # Check for port conflicts
   netstat -an | grep :8000
   
   # Check container logs
   docker logs claude-proxy-personal
   ```

2. **Claude authentication issues**
   ```bash
   # Check Claude CLI authentication
   claude auth status
   
   # Re-authenticate if needed
   claude auth login
   
   # Verify config directory
   ls -la ~/.config/claude/
   ```

3. **Performance issues**
   ```bash
   # Monitor resource usage
   docker stats claude-proxy-personal
   
   # Adjust resource limits in docker-compose.yml
   # Reduce workers if needed
   ```

### Log Analysis for Personal Use

```bash
# View recent logs
docker logs --tail 50 claude-proxy-personal

# Search for specific errors
docker logs claude-proxy-personal 2>&1 | grep -i error

# Monitor logs in real-time
docker logs -f claude-proxy-personal
```

## Getting Help

For personal use support:

1. **Check logs** first for error messages
2. **Verify Claude CLI** authentication and status
3. **Test health endpoint** to confirm service status
4. **Review configuration** for any local customizations
5. **Check Docker** resources and container status

## Security Best Practices Summary

- **Bind to localhost only** (`127.0.0.1:8000:8000`)
- **Use read-only volumes** for Claude configuration
- **Regular container updates** for security patches
- **Monitor resource usage** to prevent system overload
- **Keep Claude CLI updated** for latest security features
- **Backup configurations** regularly for quick recovery