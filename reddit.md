# ccproxy-api

Available at [https://github.com/CaddyGlow/ccproxy-api/](https://github.com/CaddyGlow/ccproxy-api/)

`ccproxy` is a local reverse proxy server for Anthropic Claude LLM at `api.anthropic.com/v1/messages`. It allows you to use your existing Claude Max subscription to interact with the Anthropic API, bypassing the need for separate API key billing.

The server provides two primary modes of operation:

* **SDK Mode (**`/sdk`**):** Routes requests through the local Claude Code using the `claude-code-sdk`. This enables access to tools configured in your Claude environment.
* **API Mode (**`/api`**):** Acts as a direct reverse proxy, injecting the necessary authentication headers. This provides full access to the underlying API features and model settings.

It includes a translation layer to support both Anthropic and OpenAI-compatible API formats for requests and responses, including streaming.

Can be run with `pipx run ccproxy-api` or `uvx ccproxy-api`.

Or installed with `pipx install ccproxy-api` or `uv tool install ccproxy-api`

Need Claude Code installed to used the `sdk` mode.

API mode can be run without it. You can login with `ccproxy-api auth login`

## API mode demo

```sh
$ uvx ccproxy-api
2025-07-22 20:24:19 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 20:24:19 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=None docker_image=None docker_mode=False duckdb_enabled=True duckdb_path=/home/rick/.local/share/ccproxy/metrics.duckdb host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 20:24:19 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 20:24:19 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8752 subscription_type=None
2025-07-22 20:24:19 [warning  ] claude_binary_not_found        install_command='npm install -g @anthropic-ai/claude-code' message='Claude CLI binary not found. Please install Claude CLI to use SDK features.' searched_paths=['PATH environment variable', '/home/rick/.claude/local/claude', '/home/rick/node_modules/.bin/claude', '/home/rick/.cache/uv/archive-v0/-l4GqN2esEE9n92CfK2fP/lib/python3.11/site-packages/node_modules/.bin/claude', '/home/rick/node_modules/.bin/claude', '/usr/local/bin/claude', '/opt/homebrew/bin/claude']
2025-07-22 20:24:19 [info     ] scheduler_starting             max_concurrent_tasks=10 registered_tasks=['pushgateway', 'stats_printing', 'pricing_cache_update']
2025-07-22 20:24:19 [info     ] scheduler_started              active_tasks=0 running_tasks=[]
2025-07-22 20:24:19 [info     ] task_added_and_started         task_name=pricing_cache_update task_type=pricing_cache_update
2025-07-22 20:24:19 [info     ] pricing_update_task_added      force_refresh_on_startup=False interval_hours=24
2025-07-22 20:24:19 [info     ] scheduler_started              active_tasks=1 max_concurrent_tasks=10 running_tasks=1
2025-07-22 20:24:19 [info     ] pricing_loaded_from_external   cache_age_hours=0.37 model_count=15
```


Starting aider with sonnet 4. We have to set the environment variable to use the proxy

```sh
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

Using the OpenAI compatible mode with Sonnet for model



```sh
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

Using OpenAI model will be map to Anthorpic one

```sh
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
───────────────────────────────────────────────────────────────────────────
multi>
```


## SDK mode demo

Install `claude-code`, if you are not login `claude /login` then run `ccproxy-api` and specify a working directory.

```sh
$ bun install --global @anthropic-ai/claude-code
bun add v1.2.18 (0d4089ea)

installed @anthropic-ai/claude-code@1.0.57 with binaries:
 - claude

1 package installed [1.74s]
$ claude /login
...
$ uvx ccproxy-api serve --cwd /tmp/tmp.AZyCo5a42N
2025-07-22 20:48:49 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 20:48:49 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=/home/rick/.cache/.bun/bin/claude docker_image=None docker_mode=False duckdb_enabled=True duckdb_path=/home/rick/.local/share/ccproxy/metrics.duckdb host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 20:48:49 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 20:48:49 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8751 subscription_type=None
2025-07-22 20:48:49 [info     ] claude_binary_found            found_in_path=False message='Claude CLI binary found at: /home/rick/.cache/.bun/bin/claude' path=/home/rick/.cache/.bun/bin/claude
2025-07-22 20:48:49 [info     ] scheduler_starting             max_concurrent_tasks=10 registered_tasks=['pushgateway', 'stats_printing', 'pricing_cache_update']
2025-07-22 20:48:49 [info     ] scheduler_started              active_tasks=0 running_tasks=[]
2025-07-22 20:48:49 [info     ] task_added_and_started         task_name=pricing_cache_update task_type=pricing_cache_update
2025-07-22 20:48:49 [info     ] pricing_update_task_added      force_refresh_on_startup=False interval_hours=24
2025-07-22 20:48:49 [info     ] scheduler_started              active_tasks=1 max_concurrent_tasks=10 running_tasks=1
2025-07-22 20:48:49 [info     ] pricing_loaded_from_external   cache_age_hours=0.78 model_count=15
```
```
```

We will use `aichat` for the demo

In `~/.config/aichat/config.yaml` you will need that

```yml
model: claude:claude-sonnet-4-20250514
clients:
  - type: claude
    api_base: http://127.0.0.1:8000/api/v1
```


Start the server, we will used `--cwd` to set the working path for claude `--allowed-tools` to allows  `Read,Write` MCP function of claude
and allow the `--permission-mode` to allow `acceptEdits`

```sh
$ uv --project ~/projects-caddy/claude-code-proxy-api run ccproxy serve --cwd /tmp/tmp.AZyCo5a42N --allowed-tools Read,Write --permission-mode acceptEdits
2025-07-22 21:49:05 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 21:49:05 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=/home/rick/.cache/.bun/bin/claude docker_image=None docker_mode=False duckdb_enabled=True duckdb_path=/home/rick/.local/share/ccproxy/metrics.duckdb host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 21:49:05 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 21:49:05 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8750 subscription_type=None
2025-07-22 21:49:05 [info     ] claude_binary_found            found_in_path=False message='Claude CLI binary found at: /home/rick/.cache/.bun/bin/claude' path=/home/rick/.cache/.bun/bin/claude
2025-07-22 21:49:05 [info     ] scheduler_starting             max_concurrent_tasks=10 registered_tasks=['pushgateway', 'stats_printing', 'pricing_cache_update']
2025-07-22 21:49:05 [info     ] scheduler_started              active_tasks=0 running_tasks=[]
2025-07-22 21:49:05 [info     ] task_added_and_started         task_name=pricing_cache_update task_type=pricing_cache_update
2025-07-22 21:49:05 [info     ] pricing_update_task_added      force_refresh_on_startup=False interval_hours=24
2025-07-22 21:49:05 [info     ] scheduler_started              active_tasks=1 max_concurrent_tasks=10 running_tasks=1
2025-07-22 21:49:05 [info     ] pricing_loaded_from_external   cache_age_hours=1.78 model_count=15
2025-07-22 21:49:11 [info     ] access_log                     client_ip=127.0.0.1 duration_ms=34.15608499199152 duration_seconds=0.034158787995693274 method=POST path=/sdk/v1/messages query=None request_id=337bea1a-5450-4c16-8f74-43b036d0c7cd service_type=access_log user_agent=unknown
2025-07-22 21:49:17 [info     ] access_log                     cache_read_tokens=13824 cache_write_tokens=14151 cost_usd=0.05958045 duration_ms=5322.508850003942 duration_seconds=5.322511695994763 endpoint=messages event_type=streaming_complete method=POST model=claude-sonnet-4-20250514 path=/sdk/v1/messages request_id=643d4a04-cbc6-4f57-bd23-514b3103fd5c service_type=claude_sdk_service status_code=200 streaming=True tokens_input=9 tokens_output=156
kan2025-07-22 21:50:36 [info     ] access_log                     client_ip=127.0.0.1 duration_ms=27.328205993399024 duration_seconds=0.027331121993483976 method=POST path=/sdk/v1/messages query=None request_id=72c4787d-9500-44ed-9607-e1b8138ee55e service_type=access_log user_agent=unknown
2025-07-22 21:50:40 [info     ] access_log                     client_ip=127.0.0.1 duration_ms=31.03382099652663 duration_seconds=0.031035719002829865 method=POST path=/sdk/v1/messages query=None request_id=0a898525-a37e-43eb-bb1e-713d498636e7 service_type=access_log user_agent=unknown
2025-07-22 21:50:46 [info     ] access_log                     cache_read_tokens=27975 cache_write_tokens=0 cost_usd=0.011219899999999998 duration_ms=5422.509380994597 duration_seconds=5.422511569006019 endpoint=messages event_type=streaming_complete method=POST model=claude-sonnet-4-20250514 path=/sdk/v1/messages request_id=2107855f-ba1b-4c23-a7a7-bfa0be304924 service_type=claude_sdk_service status_code=200 streaming=True tokens_input=9 tokens_output=178
2025-07-22 21:51:04 [info     ] access_log                     client_ip=127.0.0.1 duration_ms=31.980050000129268 duration_seconds=0.0319822429883061 method=POST path=/sdk/v1/messages query=None request_id=b22cab2f-09f8-4acb-804a-9f0a5355ddb6 service_type=access_log user_agent=unknown
2025-07-22 21:51:12 [info     ] access_log                     cache_read_tokens=41433 cache_write_tokens=702 cost_usd=0.019110000000000002 duration_ms=8115.115786000388 duration_seconds=8.115117488996475 endpoint=messages event_type=streaming_complete method=POST model=claude-sonnet-4-20250514 path=/sdk/v1/messages request_id=b29a4066-bb91-4c29-84dd-17423d9a176b service_type=claude_sdk_service status_code=200 streaming=True tokens_input=14 tokens_output=228
```

Using aichat to ask to write a hello world. Output it's not great because we are currently showing some internal message generated by the sdk and aichat have an issue at the end of the stream.

```sh
$ cd /tmp/tmp.AZyCo5a42N
$ ls
$ aichat "Hello claude, write me an hello world in test.c"
<system>{"subtype": "init", "data": {"type": "system", "subtype": "init", "cwd": "/tmp/tmp.AZyCo5a42N", "session_id": "c68ceefd-27ca-4ecf-a690-bd1b18cfeb91", "tools": ["Task", "Bash", "Glob", "Grep", "LS", "ExitPlanMode",
"Read", "Edit", "MultiEdit", "Write", "NotebookRead", "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch"], "mcp_servers": [], "model": "claude-sonnet-4-20250514", "permissionMode": "acceptEdits", "apiKeySource": "none"}}</
system><assistant>I'll create a simple "Hello, World!" program in C for you.</assistant><assistant><tooluseblock id="toolu_01TdwMXQKE2kq3Ctg1h9qxNm" name="Write">{"file_path": "/tmp/tmp.AZyCo5a42N/test.c", "content":
"#include &lt;stdio.h&gt;\n\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}"}</tooluseblock></assistant><assistant>Created test.c with a basic Hello World program. You can compile it with `gcc test.c -o
test` and run it with `./test`.</assistant>
Error: Failed to call chat-completions api

Caused by:
    expected value at line 1 column 2
```

Ask to build but it's not allowed

```sh
% aichat "build test.c"
<system>{"subtype": "init", "data": {"type": "system", "subtype": "init", "cwd": "/tmp/tmp.AZyCo5a42N", "session_id": "ac949424-f68e-4b47-a6ff-b1242401c1cd", "tools": ["Task", "Bash", "Glob", "Grep", "LS", "ExitPlanMode",
"Read", "Edit", "MultiEdit", "Write", "NotebookRead", "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch"], "mcp_servers": [], "model": "claude-sonnet-4-20250514", "permissionMode": "acceptEdits", "apiKeySource": "none"}}</
system><assistant>I'll build the test.c file for you. Let me first check if it exists and then compile it.</assistant><assistant><tooluseblock id="toolu_01H9tgfy4LE8wVqzYYDLwsdP" name="LS">{"path": "/tmp/tmp.AZyCo5a42N"}</
tooluseblock></assistant><assistant><tooluseblock id="toolu_01Q9FdDC2GknHGwCbQgdCgkN" name="Bash">{"command": "gcc test.c -o test", "description": "Compile test.c using gcc"}</tooluseblock></assistant><assistant>I need
permission to run bash commands to compile the file. Please grant bash permissions so I can compile test.c using gcc.</assistant>
Error: Failed to call chat-completions api

Caused by:
    expected value at line 1 column 2

```

File created

```sh
$ ls
test.c
$ cat test.c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
$
```
```
```

It's possible to used MCP server to ask for permission with the flag `--permission-prompt-tool`



`ccproxy` is a local reverse proxy server for Anthropic Claude LLM at `api.anthropic.com/v1/messages`. It allows you to use your existing Claude Max subscription to interact with the Anthropic API, bypassing the need for separate API key billing.

The server provides two primary modes of operation:

* **SDK Mode (**`/sdk`**):** Routes requests through the local Claude Code using the `claude-code-sdk`. This enables access to tools configured in your Claude environment.
* **API Mode (**`/api`**):** Acts as a direct reverse proxy, injecting the necessary authentication headers. This provides full access to the underlying API features and model settings.

It includes a translation layer to support both Anthropic and OpenAI-compatible API formats for requests and responses, including streaming.

Can be run with `pipx run ccproxy-api` or `uvx ccproxy-api`.

Or installed with `pipx install ccproxy-api` or `uv tool install ccproxy-api`

Need Claude Code installed to used the `sdk` mode.

API mode can be run without it. You can login with `ccproxy-api auth login`

## API mode demo

```sh
$ uvx ccproxy-api
2025-07-22 20:24:19 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 20:24:19 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=None docker_image=None docker_mode=False duckdb_enabled=True duckdb_path=/home/rick/.local/share/ccproxy/metrics.duckdb host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 20:24:19 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 20:24:19 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8752 subscription_type=None
2025-07-22 20:24:19 [warning  ] claude_binary_not_found        install_command='npm install -g @anthropic-ai/claude-code' message='Claude CLI binary not found. Please install Claude CLI to use SDK features.' searched_paths=['PATH environment variable', '/home/rick/.claude/local/claude', '/home/rick/node_modules/.bin/claude', '/home/rick/.cache/uv/archive-v0/-l4GqN2esEE9n92CfK2fP/lib/python3.11/site-packages/node_modules/.bin/claude', '/home/rick/node_modules/.bin/claude', '/usr/local/bin/claude', '/opt/homebrew/bin/claude']
2025-07-22 20:24:19 [info     ] scheduler_starting             max_concurrent_tasks=10 registered_tasks=['pushgateway', 'stats_printing', 'pricing_cache_update']
2025-07-22 20:24:19 [info     ] scheduler_started              active_tasks=0 running_tasks=[]
2025-07-22 20:24:19 [info     ] task_added_and_started         task_name=pricing_cache_update task_type=pricing_cache_update
2025-07-22 20:24:19 [info     ] pricing_update_task_added      force_refresh_on_startup=False interval_hours=24
2025-07-22 20:24:19 [info     ] scheduler_started              active_tasks=1 max_concurrent_tasks=10 running_tasks=1
2025-07-22 20:24:19 [info     ] pricing_loaded_from_external   cache_age_hours=0.37 model_count=15
```


Starting aider with sonnet 4. We have to set the environment variable to use the proxy

```sh
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

Using the OpenAI compatible mode with Sonnet for model



```sh
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

Using OpenAI model will be map to Anthorpic one

```sh
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
───────────────────────────────────────────────────────────────────────────
multi>
```


## SDK mode demo

Install `claude-code`, if you are not login `claude /login` then run `ccproxy-api` and specify a working directory.

```sh
$ bun install --global @anthropic-ai/claude-code
bun add v1.2.18 (0d4089ea)

installed @anthropic-ai/claude-code@1.0.57 with binaries:
 - claude

1 package installed [1.74s]
$ claude /login
...
$ uvx ccproxy-api serve --cwd /tmp/tmp.AZyCo5a42N
2025-07-22 20:48:49 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 20:48:49 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=/home/rick/.cache/.bun/bin/claude docker_image=None docker_mode=False duckdb_enabled=True duckdb_path=/home/rick/.local/share/ccproxy/metrics.duckdb host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 20:48:49 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 20:48:49 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8751 subscription_type=None
2025-07-22 20:48:49 [info     ] claude_binary_found            found_in_path=False message='Claude CLI binary found at: /home/rick/.cache/.bun/bin/claude' path=/home/rick/.cache/.bun/bin/claude
2025-07-22 20:48:49 [info     ] scheduler_starting             max_concurrent_tasks=10 registered_tasks=['pushgateway', 'stats_printing', 'pricing_cache_update']
2025-07-22 20:48:49 [info     ] scheduler_started              active_tasks=0 running_tasks=[]
2025-07-22 20:48:49 [info     ] task_added_and_started         task_name=pricing_cache_update task_type=pricing_cache_update
2025-07-22 20:48:49 [info     ] pricing_update_task_added      force_refresh_on_startup=False interval_hours=24
2025-07-22 20:48:49 [info     ] scheduler_started              active_tasks=1 max_concurrent_tasks=10 running_tasks=1
2025-07-22 20:48:49 [info     ] pricing_loaded_from_external   cache_age_hours=0.78 model_count=15
```
```
```

We will use `aichat` for the demo

In `~/.config/aichat/config.yaml` you will need that

```yml
model: claude:claude-sonnet-4-20250514
clients:
  - type: claude
    api_base: http://127.0.0.1:8000/api/v1
```


Start the server, we will used `--cwd` to set the working path for claude `--allowed-tools` to allows  `Read,Write` MCP function of claude
and allow the `--permission-mode` to allow `acceptEdits`

```sh
$ uv --project ~/projects-caddy/claude-code-proxy-api run ccproxy serve --cwd /tmp/tmp.AZyCo5a42N --allowed-tools Read,Write --permission-mode acceptEdits
2025-07-22 21:49:05 [info     ] cli_command_starting           command=serve config_path=None docker=False host=None port=None
2025-07-22 21:49:05 [info     ] configuration_loaded           auth_enabled=False claude_cli_path=/home/rick/.cache/.bun/bin/claude docker_image=None docker_mode=False duckdb_enabled=True duckdb_path=/home/rick/.local/share/ccproxy/metrics.duckdb host=127.0.0.1 log_file=None log_level=INFO port=8000
2025-07-22 21:49:05 [info     ] server_start                   host=127.0.0.1 port=8000 url=http://127.0.0.1:8000
2025-07-22 21:49:05 [info     ] auth_token_valid               credentials_path=/home/rick/.claude/.credentials.json expires_in_hours=8750 subscription_type=None
2025-07-22 21:49:05 [info     ] claude_binary_found            found_in_path=False message='Claude CLI binary found at: /home/rick/.cache/.bun/bin/claude' path=/home/rick/.cache/.bun/bin/claude
2025-07-22 21:49:05 [info     ] scheduler_starting             max_concurrent_tasks=10 registered_tasks=['pushgateway', 'stats_printing', 'pricing_cache_update']
2025-07-22 21:49:05 [info     ] scheduler_started              active_tasks=0 running_tasks=[]
2025-07-22 21:49:05 [info     ] task_added_and_started         task_name=pricing_cache_update task_type=pricing_cache_update
2025-07-22 21:49:05 [info     ] pricing_update_task_added      force_refresh_on_startup=False interval_hours=24
2025-07-22 21:49:05 [info     ] scheduler_started              active_tasks=1 max_concurrent_tasks=10 running_tasks=1
2025-07-22 21:49:05 [info     ] pricing_loaded_from_external   cache_age_hours=1.78 model_count=15
2025-07-22 21:49:11 [info     ] access_log                     client_ip=127.0.0.1 duration_ms=34.15608499199152 duration_seconds=0.034158787995693274 method=POST path=/sdk/v1/messages query=None request_id=337bea1a-5450-4c16-8f74-43b036d0c7cd service_type=access_log user_agent=unknown
2025-07-22 21:49:17 [info     ] access_log                     cache_read_tokens=13824 cache_write_tokens=14151 cost_usd=0.05958045 duration_ms=5322.508850003942 duration_seconds=5.322511695994763 endpoint=messages event_type=streaming_complete method=POST model=claude-sonnet-4-20250514 path=/sdk/v1/messages request_id=643d4a04-cbc6-4f57-bd23-514b3103fd5c service_type=claude_sdk_service status_code=200 streaming=True tokens_input=9 tokens_output=156
kan2025-07-22 21:50:36 [info     ] access_log                     client_ip=127.0.0.1 duration_ms=27.328205993399024 duration_seconds=0.027331121993483976 method=POST path=/sdk/v1/messages query=None request_id=72c4787d-9500-44ed-9607-e1b8138ee55e service_type=access_log user_agent=unknown
2025-07-22 21:50:40 [info     ] access_log                     client_ip=127.0.0.1 duration_ms=31.03382099652663 duration_seconds=0.031035719002829865 method=POST path=/sdk/v1/messages query=None request_id=0a898525-a37e-43eb-bb1e-713d498636e7 service_type=access_log user_agent=unknown
2025-07-22 21:50:46 [info     ] access_log                     cache_read_tokens=27975 cache_write_tokens=0 cost_usd=0.011219899999999998 duration_ms=5422.509380994597 duration_seconds=5.422511569006019 endpoint=messages event_type=streaming_complete method=POST model=claude-sonnet-4-20250514 path=/sdk/v1/messages request_id=2107855f-ba1b-4c23-a7a7-bfa0be304924 service_type=claude_sdk_service status_code=200 streaming=True tokens_input=9 tokens_output=178
2025-07-22 21:51:04 [info     ] access_log                     client_ip=127.0.0.1 duration_ms=31.980050000129268 duration_seconds=0.0319822429883061 method=POST path=/sdk/v1/messages query=None request_id=b22cab2f-09f8-4acb-804a-9f0a5355ddb6 service_type=access_log user_agent=unknown
2025-07-22 21:51:12 [info     ] access_log                     cache_read_tokens=41433 cache_write_tokens=702 cost_usd=0.019110000000000002 duration_ms=8115.115786000388 duration_seconds=8.115117488996475 endpoint=messages event_type=streaming_complete method=POST model=claude-sonnet-4-20250514 path=/sdk/v1/messages request_id=b29a4066-bb91-4c29-84dd-17423d9a176b service_type=claude_sdk_service status_code=200 streaming=True tokens_input=14 tokens_output=228
```

Using aichat to ask to write a hello world. Output it's not great because we are currently showing some internal message generated by the sdk and aichat have an issue at the end of the stream.

```sh
$ cd /tmp/tmp.AZyCo5a42N
$ ls
$ aichat "Hello claude, write me an hello world in test.c"
<system>{"subtype": "init", "data": {"type": "system", "subtype": "init", "cwd": "/tmp/tmp.AZyCo5a42N", "session_id": "c68ceefd-27ca-4ecf-a690-bd1b18cfeb91", "tools": ["Task", "Bash", "Glob", "Grep", "LS", "ExitPlanMode",
"Read", "Edit", "MultiEdit", "Write", "NotebookRead", "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch"], "mcp_servers": [], "model": "claude-sonnet-4-20250514", "permissionMode": "acceptEdits", "apiKeySource": "none"}}</
system><assistant>I'll create a simple "Hello, World!" program in C for you.</assistant><assistant><tooluseblock id="toolu_01TdwMXQKE2kq3Ctg1h9qxNm" name="Write">{"file_path": "/tmp/tmp.AZyCo5a42N/test.c", "content":
"#include &lt;stdio.h&gt;\n\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}"}</tooluseblock></assistant><assistant>Created test.c with a basic Hello World program. You can compile it with `gcc test.c -o
test` and run it with `./test`.</assistant>
Error: Failed to call chat-completions api

Caused by:
    expected value at line 1 column 2
```

Ask to build but it's not allowed

```sh
% aichat "build test.c"
<system>{"subtype": "init", "data": {"type": "system", "subtype": "init", "cwd": "/tmp/tmp.AZyCo5a42N", "session_id": "ac949424-f68e-4b47-a6ff-b1242401c1cd", "tools": ["Task", "Bash", "Glob", "Grep", "LS", "ExitPlanMode",
"Read", "Edit", "MultiEdit", "Write", "NotebookRead", "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch"], "mcp_servers": [], "model": "claude-sonnet-4-20250514", "permissionMode": "acceptEdits", "apiKeySource": "none"}}</
system><assistant>I'll build the test.c file for you. Let me first check if it exists and then compile it.</assistant><assistant><tooluseblock id="toolu_01H9tgfy4LE8wVqzYYDLwsdP" name="LS">{"path": "/tmp/tmp.AZyCo5a42N"}</
tooluseblock></assistant><assistant><tooluseblock id="toolu_01Q9FdDC2GknHGwCbQgdCgkN" name="Bash">{"command": "gcc test.c -o test", "description": "Compile test.c using gcc"}</tooluseblock></assistant><assistant>I need
permission to run bash commands to compile the file. Please grant bash permissions so I can compile test.c using gcc.</assistant>
Error: Failed to call chat-completions api

Caused by:
    expected value at line 1 column 2

```

File created

```sh
$ ls
test.c
$ cat test.c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");
    return 0;
}
$
```
```
```

It's possible to used MCP server to ask for permission with the flag `--permission-prompt-tool`
