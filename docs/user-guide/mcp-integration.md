# MCP Server Integration

## Overview

MCP (Model Context Protocol) servers allow you to extend Claude's capabilities with custom tools and functions. While the Claude Code Proxy API doesn't directly support function calling, you can leverage MCP servers through Claude Code's built-in MCP support.

## Setup Requirements

MCP servers must be registered using the Claude CLI in the environment where Claude Code is running:

```bash
# Add an MCP server
claude mcp add <server-name> <server-config>

# List registered MCP servers
claude mcp list

# Remove an MCP server
claude mcp remove <server-name>
```

**Configuration Locations:**
- Current directory: `.claude/mcp.json`
- Global configuration: `~/.claude/mcp.json`
- Git repository root: `<repo-root>/.claude/mcp.json`

## Local Machine Deployment

When running the proxy on your local machine, you can use any MCP server deployment method:

### NPX/Node.js Servers
```bash
claude mcp add filesystem '{"command": "npx", "args": ["@anthropic/mcp-server-filesystem", "/path/to/directory"]}'
```

### UVX/Python Servers
```bash
claude mcp add python-server '{"command": "uvx", "args": ["mcp-server-package", "--config", "config.json"]}'
```

### Docker-based Servers
```bash
claude mcp add docker-server '{"command": "docker", "args": ["run", "--rm", "-i", "your-mcp-server-image"]}'
```

### Native Binaries
```bash
claude mcp add native-server '{"command": "/path/to/mcp-server-binary", "args": ["--config", "config.json"]}'
```

## Docker Container Deployment

When Claude Code runs inside a Docker container, MCP server options are more limited.

### Supported Methods

**NPX/Node.js**: Install Node.js in the container
```dockerfile
RUN apt-get update && apt-get install -y nodejs npm
```

**UVX/Python**: Install Python and uv in the container
```dockerfile
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install uv
```

**Native binaries**: Include binaries in the container image

### Docker Socket Method (Advanced)

```bash
# Mount Docker socket for container access
docker run -v /var/run/docker.sock:/var/run/docker.sock claude-code-proxy
```

**Security Warning:** Mounting `/var/run/docker.sock` breaks container isolation and allows Docker escape. This creates significant security risks and should only be used in trusted environments with full understanding of the implications.

## Common MCP Server Examples

### Filesystem Access
```bash
claude mcp add filesystem '{"command": "npx", "args": ["@anthropic/mcp-server-filesystem", "/workspace"]}'
```

### Database Integration
```bash
claude mcp add database '{"command": "uvx", "args": ["mcp-server-sqlite", "--db-path", "/data/app.db"]}'
```

### Web Scraping
```bash
claude mcp add web-scraper '{"command": "npx", "args": ["@anthropic/mcp-server-puppeteer"]}'
```

### Custom API Client
```bash
claude mcp add api-client '{"command": "./mcp-servers/api-client", "args": ["--config", "api-config.json"]}'
```

## Usage Flow

1. **Register MCP servers** using `claude mcp add` in the environment where Claude Code runs
2. **Start the Claude Code Proxy API**
3. **Make API requests** - Claude automatically uses registered MCP servers for relevant requests
4. **Function calls are handled internally** by Claude Code, not exposed through the API

## Best Practices

- **Test MCP servers independently** before integrating with the proxy
- **Use specific server configurations** rather than broad access permissions
- **Monitor resource usage** of MCP servers in production
- **Keep MCP server configurations** in version control when possible
- **Document custom MCP servers** for team members

## Troubleshooting

### Common Issues

**MCP server not responding:**
- Check if the server command is accessible in the environment
- Verify configuration JSON syntax
- Review Claude logs for connection errors

**Docker container can't access local services:**
- Use host networking or container-to-container communication
- Avoid mounting Docker socket unless absolutely necessary

**Permission errors:**
- Ensure Claude has proper permissions to execute MCP server commands
- Check file system permissions for server binaries and configs

### Debugging

```bash
# Test MCP server connection
claude mcp test <server-name>

# View detailed logs
claude --log-level debug

# List active MCP connections
claude mcp status
```

This integration approach provides powerful extensibility while maintaining the simplicity of the proxy API interface.