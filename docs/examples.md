# Examples

This guide provides practical examples of using CCProxy API in different scenarios.

## Quick Start

CCProxy can be run instantly without installation:

```bash
# Run with uv
uvx ccproxy-api

# Run with pipx
pipx run ccproxy-api
```

Or install it for regular use:

```bash
# Install with pipx
pipx install ccproxy-api

# Install with uv
uv tool install ccproxy-api
```

Note: Claude Code SDK (`npm install -g @anthropic-ai/claude-code`) is required for SDK mode but not for API mode.

## API Mode Demo

API mode provides direct proxy access to Claude without requiring the Claude Code SDK.

### Starting the Server

```bash
$ uvx ccproxy-api
2025-07-22 20:24:19 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 20:24:19 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=None docker_image=None docker_mode=False duckdb_enabled=True host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 20:24:19 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 20:24:19 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8752 subscription_type=None
2025-07-22 20:24:19 [warning  ] claude_binary_not_found        install_command='npm install -g @anthropic-ai/claude-code' message='Claude CLI binary not found. Please install Claude CLI to use SDK features.' searched_paths=['PATH environment variable', '/home/rick/.claude/local/claude', '/home/rick/node_modules/.bin/claude', '/home/rick/.cache/uv/archive-v0/-l4GqN2esEE9n92CfK2fP/lib/python3.11/site-packages/node_modules/.bin/claude', '/home/rick/node_modules/.bin/claude', '/usr/local/bin/claude', '/opt/homebrew/bin/claude']
2025-07-22 20:24:19 [info     ] scheduler_starting             max_concurrent_tasks=10 registered_tasks=['version_update_check', 'pool_stats']
2025-07-22 20:24:19 [info     ] scheduler_started              active_tasks=0 running_tasks=[]
2025-07-22 20:24:19 [info     ] task_added_and_started         task_name=pricing_cache_update task_type=pricing_cache_update
2025-07-22 20:24:19 [info     ] pricing_update_task_added      force_refresh_on_startup=False interval_hours=24
2025-07-22 20:24:19 [info     ] scheduler_started              active_tasks=1 max_concurrent_tasks=10 running_tasks=1
2025-07-22 20:24:19 [info     ] pricing_loaded_from_external   cache_age_hours=0.37 model_count=15
```

### Using with Aider (Anthropic Format)

```bash
$ OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8000/api/ \
    ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL=http://127.0.0.1:8000/api \
    aider --model claude-sonnet-4-20250514
───────────────────────────────────────────────────────────────────────────────
Aider v0.85.2
Main model: claude-sonnet-4-20250514 with diff edit format, infinite output
Weak model: claude-3-5-haiku-20241022
Git repo: .git with 0 files
Repo-map: using 4096 tokens, auto refresh
Multiline mode: Enabled. Enter inserts newline, Alt-Enter submits text
──────────────────────────────────────────────────────────────────────────────
multi> Hello claude

Hello! I'm Claude Code, ready to help you with your software development tasks. Since you haven't shared any files yet, I'm waiting for you to add files to the chat that you'd like me to help modify or work with.

Just let me know what you'd like to work on - whether it's creating new files, modifying existing code, or any other development tasks!


Tokens: 2.5k sent, 80 received. Cost: $0.0088 message, $0.0088 session.
```

### Using with Aider (OpenAI Format)

```bash
$ OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8000/api/v1 ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL=http://127.0.0.1:8000/api aider --model openai/claude-sonnet-4-20250514

──────────────────────────────────────────────────────────────────────────────
Aider v0.85.2
Model: openai/claude-sonnet-4-20250514 with whole edit format
Git repo: .git with 0 files
Repo-map: using 1024 tokens, auto refresh
Multiline mode: Enabled. Enter inserts newline, Alt-Enter submits text
──────────────────────────────────────────────────────────────────────────────
multi> What model are you?

I am Claude, an AI assistant created by Anthropic. I'm designed to help with software development tasks, including reviewing code, suggesting changes, and creating new files when you provide them to me.

Since you haven't shared any code files yet, I'm ready to help once you do. Just share your code and let me know what changes you'd like me to make!


Tokens: 603 sent, 76 received.
──────────────────────────────────────────────────────────────────────────────
multi>
```

### OpenAI Model Mapping

CCProxy automatically maps OpenAI model names to Claude models:

```bash
$ OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8000/api/v1 ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL=http://127.0.0.1:8000/api aider --model openai/o3
────────────────────────────────────────────────────────────────────────────────
Warning: Streaming is not supported by openai/o3. Disabling streaming.
Aider v0.85.2
Main model: openai/o3 with diff edit format
Weak model: openai/gpt-4.1-mini
Git repo: .git with 0 files
Repo-map: using 4096 tokens, auto refresh
Multiline mode: Enabled. Enter inserts newline, Alt-Enter submits text
───────────────────────────────────────────────────────────────────────
multi> What model are you ?



The user is asking what model I am. According to my instructions, I am Claude Code, Anthropic's official CLI for Claude. I should respond to this question directly and clearly.I am Claude Code, Anthropic's official CLI for
Claude. I'm an AI assistant specifically designed to help with software development tasks, code editing, and technical questions. I'm built to work with the SEARCH/REPLACE block format for making code changes and can help
you with various programming tasks across different languages and frameworks.

Is there a coding task or project you'd like help with today?

Tokens: 2.7k sent, 132 received. Cost: $0.0064 message, $0.0064 session.
───────────────────────────────────────────────────────────────────────
multi>
```

## SDK Mode Demo

SDK mode provides access to Claude Code's full tool capabilities including file operations and MCP tools.

### Installation and Setup

```bash
# Install Claude Code SDK
$ bun install --global @anthropic-ai/claude-code
bun add v1.2.18 (0d4089ea)

installed @anthropic-ai/claude-code@1.0.57 with binaries:
 - claude

1 package installed [1.74s]

# Login to Claude
$ claude /login
...

# Start CCProxy with a working directory
$ uvx ccproxy-api serve --cwd /tmp/tmp.AZyCo5a42N
2025-07-22 20:48:49 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 20:48:49 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=/home/rick/.cache/.bun/bin/claude docker_image=None docker_mode=False duckdb_enabled=True host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 20:48:49 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 20:48:49 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8751 subscription_type=None
2025-07-22 20:48:49 [info     ] claude_binary_found            found_in_path=False message='Claude CLI binary found at: /home/rick/.cache/.bun/bin/claude' path=/home/rick/.cache/.bun/bin/claude
2025-07-22 20:48:49 [info     ] scheduler_starting             max_concurrent_tasks=10 registered_tasks=['version_update_check', 'pool_stats']
2025-07-22 20:48:49 [info     ] scheduler_started              active_tasks=0 running_tasks=[]
2025-07-22 20:48:49 [info     ] task_added_and_started         task_name=pricing_cache_update task_type=pricing_cache_update
2025-07-22 20:48:49 [info     ] pricing_update_task_added      force_refresh_on_startup=False interval_hours=24
2025-07-22 20:48:49 [info     ] scheduler_started              active_tasks=1 max_concurrent_tasks=10 running_tasks=1
2025-07-22 20:48:49 [info     ] pricing_loaded_from_external   cache_age_hours=0.78 model_count=15
```

### Using with AIChat

Configure AIChat in `~/.config/aichat/config.yaml`:

```yaml
model: claude:claude-sonnet-4-20250514
clients:
  - type: claude
    api_base: http://127.0.0.1:8000/api/v1
```

Start the server with specific permissions:

```bash
$ uv --project ~/projects-caddy/claude-code-proxy-api run ccproxy serve --cwd /tmp/tmp.AZyCo5a42N --allowed-tools Read,Write --permission-mode acceptEdits
2025-07-22 21:49:05 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 21:49:05 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=/home/rick/.cache/.bun/bin/claude docker_image=None docker_mode=False duckdb_enabled=True host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 21:49:05 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 21:49:05 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8750 subscription_type=None
2025-07-22 21:49:05 [info     ] claude_binary_found            found_in_path=False message='Claude CLI binary found at: /home/rick/.cache/.bun/bin/claude' path=/home/rick/.cache/.bun/bin/claude
```

### Creating Files with Tools

```bash
$ cd /tmp/tmp.AZyCo5a42N
$ ls
$ aichat "Hello claude, write me an hello world in test.c"
<system>{"subtype": "init", "data": {"type": "system", "subtype": "init", "cwd": "/tmp/tmp.AZyCo5a42N", "session_id": "c68ceefd-27ca-4ecf-a690-bd1b18cfeb91", "tools": ["Task", "Bash", "Glob", "Grep", "LS", "ExitPlanMode",
"Read", "Edit", "MultiEdit", "Write", "NotebookRead", "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch"], "mcp_servers": [], "model": "claude-sonnet-4-20250514", "permissionMode": "acceptEdits", "apiKeySource": "none"}}</
system><assistant>I'll create a simple "Hello, World!" program in C for you.</assistant><assistant><tooluseblock id="toolu_01TdwMXQKE2kq3Ctg1h9qxNm" name="Write">{"file_path": "/tmp/tmp.AZyCo5a42N/test.c", "content":
"#include &lt;stdio.h&gt;\n\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}"}</tooluseblock></assistant><assistant>Created test.c with a basic Hello World program. You can compile it with `gcc test.c -o
test` and run it with `./test`.</assistant>
```

Verify the file was created:

```bash
$ ls
test.c
$ cat test.c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
```

### Permission Handling

When tools require permissions not granted:

```bash
$ aichat "build test.c"
<system>{"subtype": "init", "data": {"type": "system", "subtype": "init", "cwd": "/tmp/tmp.AZyCo5a42N", "session_id": "ac949424-f68e-4b47-a6ff-b1242401c1cd", "tools": ["Task", "Bash", "Glob", "Grep", "LS", "ExitPlanMode",
"Read", "Edit", "MultiEdit", "Write", "NotebookRead", "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch"], "mcp_servers": [], "model": "claude-sonnet-4-20250514", "permissionMode": "acceptEdits", "apiKeySource": "none"}}</
system><assistant>I'll build the test.c file for you. Let me first check if it exists and then compile it.</assistant><assistant><tooluseblock id="toolu_01H9tgfy4LE8wVqzYYDLwsdP" name="LS">{"path": "/tmp/tmp.AZyCo5a42N"}</
tooluseblock></assistant><assistant><tooluseblock id="toolu_01Q9FdDC2GknHGwCbQgdCgkN" name="Bash">{"command": "gcc test.c -o test", "description": "Compile test.c using gcc"}</tooluseblock></assistant><assistant>I need
permission to run bash commands to compile the file. Please grant bash permissions so I can compile test.c using gcc.</assistant>
```

## Python SDK Examples

### Using Anthropic SDK

```python
from anthropic import Anthropic

# SDK mode - with Claude Code tools
client = Anthropic(
    base_url="http://localhost:8000/sdk",
    api_key="dummy"  # Required by SDK but ignored
)

# API mode - direct proxy
client = Anthropic(
    base_url="http://localhost:8000/api",
    api_key="dummy"
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.content[0].text)
```

### Using OpenAI SDK

```python
from openai import OpenAI

# SDK mode
client = OpenAI(
    base_url="http://localhost:8000/sdk/v1",
    api_key="dummy"
)

# API mode
client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming Responses

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000/api",
    api_key="dummy"
)

stream = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Count to 10"}],
    max_tokens=100,
    stream=True
)

for event in stream:
    if event.type == "content_block_delta":
        print(event.delta.text, end="", flush=True)
```

## cURL Examples

### Basic Message Request

```bash
# SDK mode
curl -X POST http://localhost:8000/sdk/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# API mode
curl -X POST http://localhost:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### OpenAI Format

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### With Authentication Token

If you've enabled authentication:

```bash
# Anthropic format
curl -X POST http://localhost:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-token-here" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# OpenAI format
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token-here" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Advanced Configuration

### Permission Modes

- `default`: Standard permission handling
- `acceptEdits`: Automatically accept file edits
- `bypassPermissions`: Bypass all permission checks

### Tool Restrictions

Use `--allowed-tools` to limit available tools:

```bash
ccproxy serve --allowed-tools Read,Write,Edit
```

### MCP Server Integration

Enable MCP servers with the `--permission-prompt-tool` flag for custom permission handling.

## More Examples

For additional examples, check the `examples/` directory in the repository:

- `anthropic_tools_demo.py` - Tool calling with Anthropic SDK
- `openai_tools_demo.py` - Tool calling with OpenAI SDK
- `textual_chat_agent.py` - Interactive terminal chat application
