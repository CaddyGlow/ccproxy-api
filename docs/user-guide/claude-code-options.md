# Claude Code Options

Advanced Claude Code SDK options can be passed through API requests using unofficial extension parameters.

!!! warning "Unofficial Extensions"
    The parameters documented on this page are **not part of the official Anthropic or OpenAI APIs**. They are Claude Code SDK-specific extensions that provide additional control over Claude's behavior.

## Overview

The Claude Code Proxy API supports passing all ClaudeCodeOptions parameters through API requests. This allows you to configure advanced Claude Code features like tool permissions, thinking tokens, MCP integrations, and more directly through API calls.

## Available Parameters

### Tool Management

#### `allowed_tools`
- **Type**: `array` of strings
- **Description**: List of tools that Claude is allowed to use
- **Example**: `["Read", "Write", "Bash", "Edit"]`

#### `disallowed_tools`
- **Type**: `array` of strings  
- **Description**: List of tools that Claude is explicitly forbidden from using
- **Example**: `["Bash", "Write"]`

#### `permission_mode`
- **Type**: `string`
- **Options**: `"default"`, `"acceptEdits"`, `"bypassPermissions"`
- **Description**: Controls how Claude handles tool permissions
  - `default`: Normal permission checking
  - `acceptEdits`: Auto-accept edit operations
  - `bypassPermissions`: Skip permission checks entirely

#### `permission_prompt_tool_name`
- **Type**: `string`
- **Description**: Specific tool name for permission prompts

### Thinking and Processing

#### `max_thinking_tokens`
- **Type**: `integer`
- **Default**: `8000`
- **Description**: Maximum number of thinking tokens Claude can use for internal reasoning

#### `max_turns`
- **Type**: `integer`
- **Description**: Maximum number of conversation turns to allow

### System Prompts

#### `append_system_prompt`
- **Type**: `string`
- **Description**: Additional system prompt text to append to the main system prompt

### Conversation Management

#### `continue_conversation`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Whether to continue a previous conversation

#### `resume`
- **Type**: `string`
- **Description**: Conversation ID to resume from

### MCP Integration

#### `mcp_tools`
- **Type**: `array` of strings
- **Description**: List of MCP (Model Context Protocol) tools to enable
- **Example**: `["filesystem", "database", "web_search"]`

#### `mcp_servers`
- **Type**: `object`
- **Description**: MCP server configurations with connection details
- **Example**:
  ```json
  {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem"],
      "env": {"NODE_ENV": "production"}
    }
  }
  ```

### Environment

#### `cwd`
- **Type**: `string`
- **Description**: Working directory path for Claude's operations
- **Example**: `"/path/to/project"`

## Usage Examples

### Basic Tool Configuration

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Help me review this code file"
    }
  ],
  "max_tokens": 2000,
  "allowed_tools": ["Read", "Edit"],
  "permission_mode": "acceptEdits",
  "cwd": "/home/user/project"
}
```

### Advanced Configuration with MCP

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Search for recent commits and analyze the codebase"
    }
  ],
  "max_tokens": 3000,
  "max_thinking_tokens": 10000,
  "allowed_tools": ["Read", "Bash", "mcp_git"],
  "mcp_tools": ["git", "filesystem"],
  "mcp_servers": {
    "git": {
      "command": "python",
      "args": ["-m", "mcp_git"],
      "env": {"GIT_REPO": "/home/user/project"}
    }
  },
  "cwd": "/home/user/project"
}
```

### Conversation Continuation

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Continue where we left off"
    }
  ],
  "max_tokens": 1000,
  "continue_conversation": true,
  "resume": "conversation_id_12345",
  "append_system_prompt": "Remember our previous discussion about the React components."
}
```

## API Compatibility

These extended parameters work with all API endpoints:

- **Anthropic API**: `/v1/chat/completions` and `/v1/messages`
- **OpenAI API**: `/openai/v1/chat/completions`

The parameters are passed alongside standard API parameters and are processed by the Claude Code SDK internally.

## CLI Configuration

Some of these options can also be configured via command-line arguments when starting the API server:

```bash
ccproxy api --allowed-tools Read,Write,Bash --permission-mode acceptEdits --max-thinking-tokens 10000
```

## Best Practices

### Security Considerations

1. **Tool Restrictions**: Use `allowed_tools` and `disallowed_tools` to limit Claude's capabilities based on your security requirements
2. **Permission Modes**: Be cautious with `bypassPermissions` - only use in trusted environments
3. **Working Directory**: Set `cwd` to restrict file operations to specific directories

### Performance Optimization

1. **Thinking Tokens**: Adjust `max_thinking_tokens` based on task complexity
2. **Tool Selection**: Only enable necessary tools to reduce overhead
3. **Conversation Limits**: Use `max_turns` to prevent runaway conversations

### MCP Integration

1. **Server Configuration**: Properly configure MCP servers with appropriate environment variables
2. **Tool Coordination**: Use `mcp_tools` to enable specific MCP capabilities
3. **Resource Management**: Consider resource usage when enabling multiple MCP servers

## Troubleshooting

### Common Issues

1. **Unknown Tool Names**: Ensure tool names in `allowed_tools` match exactly with available tools
2. **MCP Connection Failures**: Check MCP server configurations and network connectivity  
3. **Permission Errors**: Verify `permission_mode` settings match your security requirements

### Validation Errors

The API will return validation errors for:
- Invalid `permission_mode` values
- Malformed `mcp_servers` configurations  
- Non-existent tool names in `allowed_tools`/`disallowed_tools`

## Related Documentation

- [API Usage Guide](api-usage.md)
- [MCP Integration](mcp-integration.md)
- [Development Setup](../developer-guide/development.md)
