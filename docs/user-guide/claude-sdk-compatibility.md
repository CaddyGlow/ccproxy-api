# Claude SDK Compatibility

Understanding the differences between Claude Code SDK and standard AI APIs.

!!! important "Key Limitation"
    This proxy uses the **Claude Code SDK** internally, which has fundamentally different capabilities and limitations compared to the official Anthropic or OpenAI APIs. Many standard API parameters are ignored because they cannot be passed to the Claude SDK.

## Overview

The Claude Code Proxy acts as a bridge between standard API formats (Anthropic/OpenAI) and the Claude Code SDK. However, the Claude SDK was designed for interactive development workflows, not general API usage, which creates significant compatibility limitations.

## Parameter Support Matrix

### Fully Supported Parameters

These parameters work exactly as expected:

| Parameter | Anthropic API | OpenAI API | Claude SDK | Notes |
|-----------|---------------|------------|------------|-------|
| `model` | ✅ | ✅ | ✅ | Passed directly to SDK |
| `messages` | ✅ | ✅ | ✅ | Converted to SDK format |
| `max_tokens` | ✅ | ✅ | ✅ | Controls response length |
| `system` | ✅ | ✅ | ✅ | Passed as system prompt |
| `stream` | ✅ | ✅ | ✅ | Handled by proxy logic |

### Ignored Parameters

These parameters are **accepted but completely ignored**:

| Parameter | Anthropic API | OpenAI API | Reason for Ignoring |
|-----------|---------------|------------|---------------------|
| `temperature` | ⚠️ | ⚠️ | Claude SDK doesn't support sampling control |
| `top_p` | ⚠️ | ⚠️ | Claude SDK doesn't support nucleus sampling |
| `top_k` | ⚠️ | ❌ | Claude SDK doesn't support top-k sampling |
| `stop_sequences` | ⚠️ | ❌ | Claude SDK doesn't support custom stop sequences |
| `stop` | ❌ | ⚠️ | Claude SDK doesn't support custom stop sequences |
| `presence_penalty` | ❌ | ⚠️ | Claude SDK doesn't support penalty parameters |
| `frequency_penalty` | ❌ | ⚠️ | Claude SDK doesn't support penalty parameters |
| `logit_bias` | ❌ | ⚠️ | Claude SDK doesn't support logit manipulation |
| `user` | ❌ | ⚠️ | Claude SDK doesn't track user identifiers |
| `seed` | ❌ | ⚠️ | Claude SDK doesn't support deterministic generation |
| `logprobs` | ❌ | ⚠️ | Claude SDK doesn't provide log probabilities |
| `top_logprobs` | ❌ | ⚠️ | Claude SDK doesn't provide log probabilities |
| `response_format` | ❌ | ⚠️ | Claude SDK doesn't support structured output |
| `n` | ❌ | ⚠️ | Claude SDK only generates single responses |

**Legend**: ✅ Supported, ⚠️ Ignored, ❌ Not applicable

### Tool Parameters (Limited Support)

Tool-related parameters have very limited compatibility:

| Parameter | Support Level | Alternative |
|-----------|---------------|-------------|
| `tools` | Very Limited | Use `allowed_tools` in ClaudeCodeOptions |
| `tool_choice` | Very Limited | Use `permission_mode` in ClaudeCodeOptions |
| `parallel_tool_calls` | Ignored | Claude SDK controls tool execution |

## Why These Limitations Exist

### Claude Code SDK Design Philosophy

The Claude Code SDK was designed for:

1. **Interactive Development**: Real-time coding assistance with built-in tools
2. **Local Context**: Working with local files, directories, and projects
3. **Permission-Based Access**: Controlled tool usage with user approval
4. **Session Management**: Maintaining context across development sessions

### Standard API Design Philosophy

Traditional AI APIs (OpenAI/Anthropic) focus on:

1. **Text Generation Control**: Fine-grained sampling parameters
2. **Custom Functions**: User-defined tool/function calling
3. **Batch Processing**: Multiple responses and deterministic generation
4. **Structured Output**: JSON, formatted responses

These different design goals create fundamental incompatibilities.

## Practical Implications

### For Basic Chat Applications

**✅ Works Well:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 1000,
  "stream": true
}
```

**❌ Parameters Ignored:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 1000,
  "temperature": 0.7,          // Ignored
  "top_p": 0.9,               // Ignored
  "presence_penalty": 0.1     // Ignored
}
```

### For Coding Applications

**✅ Use ClaudeCodeOptions Instead:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Help me debug this code"}],
  "max_tokens": 2000,

  // Claude SDK specific options
  "max_thinking_tokens": 10000,
  "allowed_tools": ["Read", "Write", "Bash", "Edit"],
  "permission_mode": "acceptEdits",
  "cwd": "/path/to/project"
}
```

### For Production Applications

Consider whether Claude SDK limitations are acceptable:

**Unsuitable for:**
- Applications requiring precise temperature control
- Custom function/tool definitions
- Deterministic generation (seed-based)
- Multiple response variants
- Structured JSON output

**Suitable for:**
- Development assistance
- Code analysis and generation
- File manipulation tasks
- Interactive coding sessions

## Migration Strategies

### From OpenAI API

1. **Remove unsupported parameters**:
   ```python
   # Before
   response = client.chat.completions.create(
       model="gpt-4",
       messages=messages,
       temperature=0.7,      # Remove
       top_p=0.9,           # Remove
       presence_penalty=0.1  # Remove
   )

   # After
   response = client.chat.completions.create(
       model="claude-3-5-sonnet-20241022",
       messages=messages
   )
   ```

2. **Replace function calling with Claude tools**:
   ```python
   # Before: Custom function definitions
   tools = [{"type": "function", "function": {...}}]

   # After: Use built-in Claude tools
   extra_params = {
       "allowed_tools": ["Read", "Write", "Bash"],
       "permission_mode": "acceptEdits"
   }
   ```

### From Anthropic API

1. **Remove sampling parameters**:
   ```python
   # Before
   response = client.messages.create(
       model="claude-3-5-sonnet-20241022",
       messages=messages,
       temperature=0.7,  # Remove
       top_p=0.9,       # Remove
       top_k=40         # Remove
   )

   # After
   response = client.messages.create(
       model="claude-3-5-sonnet-20241022",
       messages=messages
   )
   ```

2. **Add ClaudeCodeOptions for advanced features**:
   ```python
   # After: Enhanced with Claude-specific options
   response = client.messages.create(
       model="claude-3-5-sonnet-20241022",
       messages=messages,
       max_thinking_tokens=15000,
       allowed_tools=["Read", "Write"],
       cwd="/project/path"
   )
   ```

## Testing Compatibility

Use the provided test script to verify parameter behavior:

```bash
./test-claude-options.sh
```

This will show which parameters are working and which are being ignored.

## Best Practices

1. **Don't rely on ignored parameters**: Remove them from your code to avoid confusion
2. **Use ClaudeCodeOptions**: Leverage Claude-specific features for better results
3. **Test thoroughly**: Behavior may differ significantly from other APIs
4. **Monitor logs**: Enable debug logging to see what's actually being passed to the SDK
5. **Understand the trade-offs**: Accept limitations in exchange for powerful coding features

## Conclusion

The Claude Code Proxy provides powerful development capabilities through the Claude SDK, but with significant API compatibility limitations. Understanding these limitations is crucial for successful integration and avoiding unexpected behavior.

For basic chat applications, the proxy works reasonably well with standard API formats. For advanced development workflows, ClaudeCodeOptions provide capabilities that exceed traditional APIs, but require learning Claude-specific parameters.
