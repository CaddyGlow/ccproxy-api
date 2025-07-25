#!/usr/bin/env python3
"""
AI Code Discussion Demo

This script demonstrates a beginner/expert code discussion between Anthropic and OpenAI clients.
Both AIs can make multiple tool requests to thoroughly explore the codebase before sharing
their findings. The OpenAI client acts as a curious beginner, while the Anthropic client
acts as an experienced expert providing explanations.

Features:
- Both AIs can make multiple tool requests per turn
- AIs explore the codebase thoroughly before responding
- OpenAI (Beginner) investigates code and asks informed questions
- Anthropic (Expert) analyzes code and provides detailed explanations
- Tool-assisted discovery and code analysis
- Rich formatting with syntax highlighting
"""

import argparse
import asyncio
import json
import logging
import os
import pathlib
from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam


try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProjectCodeAccessError(Exception):
    """Custom exception for project code access errors."""

    pass


class SecureCodeAccess:
    """Secure code access system limited to project root."""

    def __init__(self, project_root: str | None = None):
        self.project_root = pathlib.Path(project_root or pathlib.Path.cwd()).resolve()

        if not self.project_root.exists():
            raise ProjectCodeAccessError(
                f"Project root does not exist: {self.project_root}"
            )

        logger.info(f"Secure code access initialized for: {self.project_root}")

    def _validate_path(self, path: str) -> pathlib.Path:
        """Validate that a path is within the project root."""
        try:
            if pathlib.Path(path).is_absolute():
                resolved_path = pathlib.Path(path).resolve()
            else:
                resolved_path = (self.project_root / path).resolve()

            if not str(resolved_path).startswith(str(self.project_root)):
                raise ProjectCodeAccessError(
                    f"Path '{path}' is outside project root '{self.project_root}'"
                )

            return resolved_path
        except Exception as e:
            raise ProjectCodeAccessError(f"Invalid path '{path}': {e}") from e

    def read_file(self, file_path: str) -> dict[str, Any]:
        """Read a file from the project directory."""
        try:
            validated_path = self._validate_path(file_path)

            if not validated_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "path": str(validated_path),
                }

            if not validated_path.is_file():
                return {
                    "success": False,
                    "error": f"Path is not a file: {file_path}",
                    "path": str(validated_path),
                }

            try:
                with validated_path.open(encoding="utf-8") as f:
                    content = f.read()

                return {
                    "success": True,
                    "content": content,
                    "path": str(validated_path.relative_to(self.project_root)),
                    "size": len(content),
                    "type": "text",
                }
            except UnicodeDecodeError:
                return {
                    "success": False,
                    "error": f"Binary file cannot be read as text: {file_path}",
                    "path": str(validated_path.relative_to(self.project_root)),
                    "type": "binary",
                }

        except ProjectCodeAccessError as e:
            return {"success": False, "error": str(e), "path": file_path}
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error reading file: {e}",
                "path": file_path,
            }

    def list_files(
        self, directory_path: str = ".", recursive: bool = False
    ) -> dict[str, Any]:
        """List files and directories in the project directory."""
        try:
            validated_path = self._validate_path(directory_path)

            if not validated_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {directory_path}",
                    "path": str(validated_path),
                }

            if not validated_path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory_path}",
                    "path": str(validated_path),
                }

            items = []

            if recursive:
                for item in validated_path.rglob("*"):
                    if item.is_file():
                        items.append(
                            {
                                "name": item.name,
                                "path": str(item.relative_to(self.project_root)),
                                "type": "file",
                                "size": item.stat().st_size,
                            }
                        )
                    elif item.is_dir():
                        items.append(
                            {
                                "name": item.name,
                                "path": str(item.relative_to(self.project_root)),
                                "type": "directory",
                            }
                        )
            else:
                for item in validated_path.iterdir():
                    if item.is_file():
                        items.append(
                            {
                                "name": item.name,
                                "path": str(item.relative_to(self.project_root)),
                                "type": "file",
                                "size": item.stat().st_size,
                            }
                        )
                    elif item.is_dir():
                        items.append(
                            {
                                "name": item.name,
                                "path": str(item.relative_to(self.project_root)),
                                "type": "directory",
                            }
                        )

            items.sort(key=lambda x: (x["type"] == "file", str(x["name"]).lower()))

            return {
                "success": True,
                "items": items,
                "path": str(validated_path.relative_to(self.project_root)),
                "recursive": recursive,
                "total_count": len(items),
            }

        except ProjectCodeAccessError as e:
            return {"success": False, "error": str(e), "path": directory_path}
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error listing directory: {e}",
                "path": directory_path,
            }


class AICodeDiscussionManager:
    """Manages the bidirectional conversation between AI clients with code access."""

    def __init__(
        self,
        project_root: str | None = None,
        proxy_url: str = "http://127.0.0.1:8000/api",
        debug: bool = False,
        stream: bool = False,
        use_rich: bool = True,
        thinking: bool = False,
    ):
        self.project_root = project_root or str(pathlib.Path.cwd())
        self.proxy_url = proxy_url
        self.debug = debug
        self.stream = stream
        self.thinking = thinking
        self.use_rich = use_rich and RICH_AVAILABLE

        # Initialize secure code access
        self.code_access = SecureCodeAccess(project_root)

        # Initialize AI clients - both use the same unified endpoint
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=f"{proxy_url}/v1",
        )

        self.anthropic_client = openai.OpenAI(
            api_key=os.getenv("ANTHROPIC_API_KEY", "dummy"), base_url=f"{proxy_url}/v1"
        )

        # Initialize console
        self.console = Console() if self.use_rich else None

        # Code access tools definition - custom tools with proper format
        self.tools = [
            {
                "type": "custom",
                "custom": {
                    "name": "read_file",
                    "description": "Read the contents of a file from the project directory to analyze code",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read (relative to project root)",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "custom",
                "custom": {
                    "name": "list_files",
                    "description": "List files and directories in the project directory to explore the codebase",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "directory_path": {
                                "type": "string",
                                "description": "Path to directory to list (relative to project root, default: '.')",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Whether to list files recursively (default: false)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of items to return (default: 50, max: 200)",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "custom",
                "custom": {
                    "name": "run_bash",
                    "description": "Execute bash commands for code exploration (allowed: rg, fd, find, cat, xargs, head, tail, wc, grep)",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Bash command to execute (must start with allowed commands)",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
        ]

        # System prompts for beginner/expert roles
        self.beginner_system_prompt = """You are a curious beginner developer learning about code architecture. Your role is to:
- Focus on exploring only 2-3 key files relevant to the topic at hand
- Use targeted tool requests - start with 'rg' or 'find' to locate specific files, then read them
- Examine main entry points, configuration files, or core modules related to the discussion topic
- After reading specific files, ask focused questions about what you discovered
- Keep exploration concise and targeted to save tokens
- Ask specific questions about patterns, design choices, or implementation details

Available tools: read_file, list_files (use limit parameter), run_bash
Strategy: Use bash to find 2-3 relevant files → Read those specific files → Ask informed questions
Focus on quality over quantity - examine key files thoroughly rather than scanning everything.
Note: read_file shows only the first 5 lines as a preview to save tokens.

Always explore strategically first, then ask specific questions about your targeted findings."""

        self.expert_system_prompt = """You are an experienced software architect providing focused guidance. Your role is to:
- Investigate only the most relevant 2-3 files related to the specific question being asked
- Use targeted searches to locate key implementation files, then read them strategically
- Focus on answering the specific question with concrete examples from the code
- Provide concise, practical explanations backed by actual code evidence
- Keep responses brief (2-3 sentences) but complete and informative
- End with a follow-up question to encourage continued learning

Available tools: read_file, list_files (use limit parameter), run_bash
Strategy: Target specific files → Read relevant sections → Provide focused explanations with examples
Be efficient with token usage - investigate precisely what's needed to answer the question.

Always investigate strategically, then provide focused explanations with code examples."""

        # Conversation histories - both use OpenAI format now
        self.openai_messages: list[ChatCompletionMessageParam] = []
        self.anthropic_messages: list[ChatCompletionMessageParam] = []

        # Token usage tracking
        self.total_tokens_used = 0
        self.turn_tokens = []

        logger.debug(
            f"AI Code Discussion Manager initialized: project_root={self.project_root}, "
            f"proxy_url={proxy_url}, stream={stream}, use_rich={use_rich}"
        )

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a code access tool."""
        if tool_name == "read_file":
            return self.code_access.read_file(kwargs.get("file_path", ""))
        elif tool_name == "list_files":
            limit = kwargs.get("limit", 50)
            if limit > 200:
                limit = 200
            result = self.code_access.list_files(
                kwargs.get("directory_path", "."), kwargs.get("recursive", False)
            )
            # Apply limit to results
            if result.get("success") and "items" in result:
                result["items"] = result["items"][:limit]
                result["limited"] = len(result["items"]) == limit
            return result
        elif tool_name == "run_bash":
            return self._execute_bash_command(kwargs.get("command", ""))
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    async def _chat_completion_with_retry(
        self,
        client: openai.OpenAI,
        max_attempts: int | None = 10,
        backoff: float = 1,
        max_delay: float = 60,
        stream: bool = None,
        **kwargs: Any,
    ):
        """Call the chat/completions endpoint with exponential back-off retries.

        Args:
            client: The OpenAI client instance to use.
            max_attempts: Maximum number of retry attempts before giving up. If ``None`` or a
                non-positive value, the request will be retried indefinitely.
            backoff: Initial back-off delay in seconds.
            max_delay: Maximum delay between retries in seconds (caps the exponential back-off).
            stream: Whether to stream the response. If None, uses self.stream.

        Returns:
            The successful response object or full streamed content if streaming.

        Raises:
            Exception: Re-raises the last exception if all attempts fail.
        """
        attempt = 0
        delay = backoff

        # Use instance stream setting if not explicitly specified
        if stream is None:
            stream = self.stream

        while True:
            try:
                if stream:
                    # Handle streaming response
                    full_content = ""
                    thinking_content = ""
                    is_thinking = False
                    is_tool_call = False

                    # Create with stream=True - remove thinking parameter for OpenAI if it exists
                    create_kwargs = kwargs.copy()
                    if (
                        "thinking" in create_kwargs
                        and "openai" in str(client.__class__).lower()
                    ):
                        del create_kwargs["thinking"]
                    stream_resp = client.chat.completions.create(
                        stream=True, **create_kwargs
                    )

                    # For rich output
                    if self.use_rich and self.console:
                        with Live("", refresh_per_second=10) as live:
                            for chunk in stream_resp:
                                # Extract content delta from the chunk
                                delta = chunk.choices[0].delta

                                if (
                                    delta
                                    and hasattr(delta, "content")
                                    and delta.content
                                ):
                                    if is_thinking:
                                        # In thinking mode, accumulate thinking content
                                        thinking_content += delta.content
                                        live.update(
                                            f"🤔 [italic yellow]Thinking:[/italic yellow] {thinking_content}"
                                        )
                                    else:
                                        # Normal content - accumulate and display
                                        full_content += delta.content
                                        # Display markdown content
                                        live.update(Markdown(full_content))

                                # Check for thinking or tool_calls state in the delta
                                if (
                                    delta
                                    and hasattr(delta, "tool_calls")
                                    and delta.tool_calls
                                ):
                                    is_tool_call = True
                                    live.update(
                                        "🔧 [italic cyan]Using tools...[/italic cyan]"
                                    )

                                # Check for thinking or tool usage indicators in content
                                if (
                                    not is_thinking
                                    and full_content
                                    and "thinking" in full_content.lower()
                                ):
                                    is_thinking = True
                                    thinking_content = full_content
                                    live.update(
                                        f"🤔 [italic yellow]Thinking:[/italic yellow] {thinking_content}"
                                    )
                    else:
                        # Simple terminal output for streaming without rich
                        print("\nStreaming response:", end="", flush=True)
                        for chunk in stream_resp:
                            delta = chunk.choices[0].delta
                            if delta and hasattr(delta, "content") and delta.content:
                                print(delta.content, end="", flush=True)
                                full_content += delta.content
                            # Check for tool calls
                            if (
                                delta
                                and hasattr(delta, "tool_calls")
                                and delta.tool_calls
                            ):
                                print("\n[Using tools...]", end="", flush=True)
                        print()  # Final newline

                    # Create a synthetic response object with the accumulated content
                    synthetic_response = type(
                        "SyntheticResponse",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {
                                        "message": type(
                                            "Message",
                                            (),
                                            {
                                                "content": full_content,
                                                "tool_calls": None,
                                                "role": "assistant",
                                            },
                                        ),
                                        "finish_reason": "stop",
                                    },
                                )
                            ],
                            "usage": type(
                                "Usage",
                                (),
                                {
                                    "prompt_tokens": 0,
                                    "completion_tokens": len(full_content)
                                    // 4,  # Rough estimate
                                    "total_tokens": len(full_content) // 4,
                                },
                            ),
                        },
                    )

                    return synthetic_response
                else:
                    # Non-streaming request - remove thinking parameter for OpenAI if it exists
                    create_kwargs = kwargs.copy()
                    if (
                        "thinking" in create_kwargs
                        and "openai" in str(client.__class__).lower()
                    ):
                        del create_kwargs["thinking"]
                    return client.chat.completions.create(**create_kwargs)
            except Exception as e:  # Broad catch – SDKs raise various errors
                attempt += 1

                # Exit if we have exhausted the allowed attempts
                if (
                    max_attempts is not None
                    and max_attempts > 0
                    and attempt >= max_attempts
                ):
                    logger.error(
                        f"Max retry attempts reached ({max_attempts}). Raising last error."
                    )
                    raise

                logger.warning(
                    f"Request failed ({e}); retrying in {delay:.1f}s "
                    f"(attempt {attempt}{'' if max_attempts is None else f'/{max_attempts}'})"
                )
                await asyncio.sleep(delay)
                # Exponential back-off with upper bound
                delay = min(delay * 2, max_delay)

    def _execute_bash_command(self, command: str) -> dict[str, Any]:
        """Execute a bash command with security filtering."""
        import shlex
        import subprocess

        # List of allowed commands
        allowed_commands = [
            "rg",
            "fd",
            "find",
            "cat",
            "xargs",
            "head",
            "tail",
            "wc",
            "grep",
            "ls",
        ]

        if not command.strip():
            return {"success": False, "error": "Empty command"}

        # Parse command to check if it starts with allowed command
        try:
            parts = shlex.split(command)
            if not parts:
                return {"success": False, "error": "Invalid command"}

            base_command = parts[0]
            if base_command not in allowed_commands:
                return {
                    "success": False,
                    "error": f"Command '{base_command}' not allowed. Allowed: {', '.join(allowed_commands)}",
                }

            # Execute in project root
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.code_access.project_root),
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout
                )

                output = result.stdout
                if result.stderr:
                    output += f"\nStderr: {result.stderr}"

                # Limit output size
                if len(output) > 10000:
                    output = output[:10000] + "\n... (output truncated)"

                return {
                    "success": True,
                    "output": output,
                    "command": command,
                    "return_code": result.returncode,
                }

            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Command timed out (30s limit)"}
            except Exception as e:
                return {"success": False, "error": f"Command execution failed: {e}"}

        except Exception as e:
            return {"success": False, "error": f"Command parsing failed: {e}"}

    def render_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Render tool execution results."""
        if not result["success"]:
            self.render_error(result["error"])
            return

        if tool_name == "read_file":
            self.render_file_content(result)
        elif tool_name == "list_files":
            self.render_directory_listing(result)
        elif tool_name == "run_bash":
            self.render_bash_output(result)

    def render_bash_output(self, result: dict[str, Any]) -> None:
        """Render bash command output."""
        if self.use_rich and self.console:
            from rich.panel import Panel
            from rich.syntax import Syntax

            output = result.get("output", "")
            command = result.get("command", "")
            return_code = result.get("return_code", 0)

            # Try to syntax highlight based on command
            if command.startswith(("rg", "grep")):
                lexer = "text"
            elif command.startswith("cat") and any(
                ext in command for ext in [".py", ".js", ".ts", ".go"]
            ):
                lexer = "python" if ".py" in command else "javascript"
            else:
                lexer = "bash"

            syntax = Syntax(output, lexer, theme="monokai", word_wrap=True)

            title = f"$ {command}"
            if return_code != 0:
                title += f" (exit code: {return_code})"

            panel = Panel(
                syntax,
                title=title,
                title_align="left",
                border_style="yellow",
                expand=False,
            )

            self.console.print(panel)
        else:
            command = result.get("command", "")
            output = result.get("output", "")
            return_code = result.get("return_code", 0)

            print(f"\n--- Command: {command} ---")
            if return_code != 0:
                print(f"Exit code: {return_code}")
            print(output)
            print("--- End of command output ---")

    def render_file_content(self, result: dict[str, Any]) -> None:
        """Render file content with syntax highlighting (first 5 lines only)."""
        if self.use_rich and self.console:
            # Detect file type for syntax highlighting
            file_path = result["path"]
            if file_path.endswith((".py", ".pyi")):
                lexer = "python"
            elif file_path.endswith((".js", ".jsx", ".ts", ".tsx")):
                lexer = "javascript"
            elif file_path.endswith((".html", ".htm")):
                lexer = "html"
            elif file_path.endswith((".css", ".scss", ".sass")):
                lexer = "css"
            elif file_path.endswith((".json", ".jsonc")):
                lexer = "json"
            elif file_path.endswith((".md", ".markdown")):
                lexer = "markdown"
            elif file_path.endswith((".yml", ".yaml")):
                lexer = "yaml"
            elif file_path.endswith((".toml",)):
                lexer = "toml"
            elif file_path.endswith((".sh", ".bash")):
                lexer = "bash"
            else:
                lexer = "text"

            content = result["content"]
            lines = content.split("\n")
            if len(lines) > 5:
                content = "\n".join(lines[:5]) + f"\n... ({len(lines) - 5} more lines)"

            syntax = Syntax(
                content, lexer, theme="monokai", line_numbers=True, word_wrap=True
            )

            panel = Panel(
                syntax,
                title=f"📄 {result['path']} ({result['size']} bytes) - First 5 lines",
                title_align="left",
                border_style="blue",
                expand=False,
            )

            self.console.print(panel)
        else:
            print(
                f"\n--- File: {result['path']} ({result['size']} bytes) - First 5 lines ---"
            )
            content = result["content"]
            lines = content.split("\n")
            if len(lines) > 5:
                content = "\n".join(lines[:5]) + f"\n... ({len(lines) - 5} more lines)"
            print(content)
            print("--- End of preview ---")

    def render_directory_listing(self, result: dict[str, Any]) -> None:
        """Render directory listing."""
        if self.use_rich and self.console:
            table = Table(
                title=f"📁 {result['path']} ({'recursive' if result['recursive'] else 'non-recursive'})",
                show_header=True,
                header_style="bold cyan",
                expand=False,
            )

            table.add_column("Type", style="yellow", width=4)
            table.add_column("Name", style="green")
            table.add_column("Size", style="magenta", justify="right")

            for item in result["items"][:20]:  # Limit to first 20 items
                type_icon = "📄" if item["type"] == "file" else "📁"
                size_str = f"{item['size']}B" if item["type"] == "file" else ""

                table.add_row(type_icon, item["name"], size_str)

            if result["total_count"] > 20:
                table.add_row("...", f"+ {result['total_count'] - 20} more items", "")

            self.console.print(table)
        else:
            print(f"\n--- Directory: {result['path']} ---")
            for item in result["items"][:20]:
                type_indicator = "F" if item["type"] == "file" else "D"
                size_str = f" ({item['size']}B)" if item["type"] == "file" else ""
                print(f"[{type_indicator}] {item['name']}{size_str}")

            if result["total_count"] > 20:
                print(f"... + {result['total_count'] - 20} more items")
            print("--- End of directory ---")

    def render_error(self, error: str) -> None:
        """Render error messages."""
        if self.use_rich and self.console:
            panel = Panel(
                Text(error, style="bold red"),
                title="❌ Error",
                title_align="left",
                border_style="red",
            )
            self.console.print(panel)
        else:
            print(f"Error: {error}")

    def render_ai_message(
        self, turn: int, speaker: str, message: str, streaming: bool = False
    ) -> None:
        """Render AI message with rich formatting."""
        if self.use_rich and self.console:
            color = "cyan" if "Beginner" in speaker else "green"

            if streaming:
                # For streaming, we'd use Live context, but for now just show the final result
                pass

            header = f"Turn {turn}: {speaker}"
            markdown_content = Markdown(message)
            panel = Panel(
                markdown_content, title=header, border_style=color, title_align="left"
            )

            self.console.print()
            self.console.print(panel)
        else:
            print(f"\n{'=' * 60}")
            print(f"Turn {turn}: {speaker}")
            print(f"{'=' * 60}")
            print(message)
            print(f"{'=' * 60}")

    def add_initial_topic(self, topic: str) -> None:
        """Add initial topic to start the code discussion."""
        # OpenAI will be the beginner (starts the conversation)
        beginner_initial_message = f"""Let's discuss: {topic}

I want to understand the key aspects of this topic in the codebase. I'll use tools to find and examine 2-3 relevant files to understand the implementation.

Let me start by targeting specific files related to {topic}."""

        # Anthropic will be the expert (responds to questions)
        expert_initial_message = f"""I'm here to help you understand: {topic}

I'll examine the most relevant files and provide focused explanations with concrete examples from the code.

Ready to investigate and explain the key implementation details!"""

        # Add system prompts and initial messages
        self.openai_messages.append(
            {"role": "system", "content": self.beginner_system_prompt}
        )
        self.openai_messages.append(
            {"role": "user", "content": beginner_initial_message}
        )

        self.anthropic_messages.append(
            {"role": "system", "content": self.expert_system_prompt}
        )
        self.anthropic_messages.append(
            {"role": "user", "content": expert_initial_message}
        )

        logger.debug(f"Initial topic added with beginner/expert roles: {topic}")

    def track_token_usage(self, response, turn: int, speaker: str) -> None:
        """Track token usage from API response."""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(
                usage, "input_tokens", 0
            )
            output_tokens = getattr(usage, "completion_tokens", 0) or getattr(
                usage, "output_tokens", 0
            )
            total_tokens = getattr(usage, "total_tokens", 0) or (
                input_tokens + output_tokens
            )

            self.turn_tokens.append(
                {
                    "turn": turn,
                    "speaker": speaker,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )
            self.total_tokens_used += total_tokens

            if self.use_rich and self.console:
                self.console.print(
                    f"[dim yellow]💰 {speaker} - Tokens: {input_tokens} in + {output_tokens} out = {total_tokens} total[/dim yellow]"
                )
            else:
                print(
                    f"💰 {speaker} - Tokens: {input_tokens} in + {output_tokens} out = {total_tokens} total"
                )

    def print_token_summary(self) -> None:
        """Print a summary of token usage."""
        if self.use_rich and self.console:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Turn", style="yellow", width=4)
            table.add_column("Speaker", style="green")
            table.add_column("Input", style="blue", justify="right")
            table.add_column("Output", style="magenta", justify="right")
            table.add_column("Total", style="red", justify="right")

            for usage in self.turn_tokens:
                table.add_row(
                    str(usage["turn"]),
                    usage["speaker"],
                    str(usage["input_tokens"]),
                    str(usage["output_tokens"]),
                    str(usage["total_tokens"]),
                )

            table.add_section()
            table.add_row(
                "", "TOTAL", "", "", str(self.total_tokens_used), style="bold"
            )

            panel = Panel(
                table,
                title="[bold green]Token Usage Summary[/bold green]",
                border_style="green",
                title_align="left",
            )
            self.console.print()
            self.console.print(panel)
        else:
            print(f"\n{'=' * 60}")
            print("TOKEN USAGE SUMMARY")
            print(f"{'=' * 60}")
            for usage in self.turn_tokens:
                print(
                    f"Turn {usage['turn']} ({usage['speaker']}): {usage['input_tokens']} in + {usage['output_tokens']} out = {usage['total_tokens']} total"
                )
            print(f"{'=' * 60}")
            print(f"TOTAL TOKENS USED: {self.total_tokens_used}")
            print(f"{'=' * 60}")

    async def send_to_openai(
        self, messages: list[ChatCompletionMessageParam], turn: int
    ) -> str:
        """Send messages to OpenAI with tool support via unified endpoint."""
        logger.debug(f"Sending to OpenAI, turn {turn}, message count: {len(messages)}")

        try:
            # Add thinking parameter if enabled - use reasoning_effort for o1 models
            extra_params = {}
            if self.thinking:
                # Use reasoning_effort parameter for thinking mode with o1 model
                extra_params["reasoning_effort"] = "high"

            response = await self._chat_completion_with_retry(
                self.openai_client,
                model="o1-mini" if self.thinking else "gpt-4o",
                messages=messages,
                tools=self.tools,  # type: ignore
                max_tokens=1000,
                temperature=1,
                **extra_params,
            )

            if not response.choices:
                raise Exception("No choices in OpenAI response")

            # Track token usage
            self.track_token_usage(response, turn, "OpenAI Beginner")

            choice = response.choices[0]
            content = choice.message.content or ""

            # Handle tool calls - allow multiple rounds
            if choice.message.tool_calls:
                # Add assistant message with tool calls to conversation
                self.openai_messages.append(choice.message)

                # Process all tool calls and collect results
                tool_results = []
                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    if self.use_rich and self.console:
                        self.console.print(
                            f"[yellow]🔧 OpenAI Beginner using tool: {tool_name}[/yellow]"
                        )
                    else:
                        print(f"🔧 OpenAI Beginner using tool: {tool_name}")

                    result = self.execute_tool(tool_name, **tool_args)
                    self.render_tool_result(tool_name, result)

                    # Add tool result message
                    self.openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, indent=2),
                        }
                    )

                # Make another request to get the AI's response after tool usage
                follow_up_response = await self._chat_completion_with_retry(
                    self.openai_client,
                    model="o1-mini" if self.thinking else "gpt-4o",
                    messages=self.openai_messages,
                    tools=self.tools,  # type: ignore
                    max_tokens=1000,
                    temperature=1,
                    **extra_params,
                )

                # Track follow-up token usage
                self.track_token_usage(
                    follow_up_response, turn, "OpenAI Beginner (follow-up)"
                )

                if follow_up_response.choices:
                    content = follow_up_response.choices[0].message.content or ""

                    # Check if there are more tool calls (recursive handling)
                    if follow_up_response.choices[0].message.tool_calls:
                        # Recursively handle more tool calls
                        self.openai_messages.append(
                            {
                                "role": "user",
                                "content": "Please continue with your analysis and provide your findings.",
                            }
                        )
                        return await self.send_to_openai(self.openai_messages, turn)
                else:
                    content = "No response after tool usage"

            return content

        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return f"Error: {e}"

    async def send_to_anthropic(
        self, messages: list[ChatCompletionMessageParam], turn: int
    ) -> str:
        """Send messages to Anthropic with tool support via unified endpoint."""
        logger.debug(
            f"Sending to Anthropic, turn {turn}, message count: {len(messages)}"
        )

        try:
            # Add thinking parameter if enabled
            extra_params = {}
            if self.thinking:
                # Use thinking parameter that works with Claude models
                extra_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            response = await self._chat_completion_with_retry(
                self.anthropic_client,
                model="claude-sonnet-4-20250514",
                messages=messages,
                tools=self.tools,  # type: ignore
                max_tokens=1000,
                temperature=1,  # Must use temperature=1 when thinking is enabled
                **extra_params,
            )

            if not response.choices:
                raise Exception("No choices in Anthropic response")

            # Track token usage
            self.track_token_usage(response, turn, "Anthropic Expert")

            choice = response.choices[0]
            content = choice.message.content or ""

            # Handle tool calls - allow multiple rounds
            if choice.message.tool_calls:
                # Add assistant message with tool calls to conversation
                self.anthropic_messages.append(choice.message)

                # Process all tool calls and collect results
                tool_results = []
                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    if self.use_rich and self.console:
                        self.console.print(
                            f"[yellow]🔧 Anthropic Expert using tool: {tool_name}[/yellow]"
                        )
                    else:
                        print(f"🔧 Anthropic Expert using tool: {tool_name}")

                    result = self.execute_tool(tool_name, **tool_args)
                    self.render_tool_result(tool_name, result)

                    # Add tool result message
                    self.anthropic_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, indent=2),
                        }
                    )

                # Make another request to get the AI's response after tool usage
                follow_up_response = await self._chat_completion_with_retry(
                    self.anthropic_client,
                    model="claude-sonnet-4-20250514",
                    messages=self.anthropic_messages,
                    tools=self.tools,  # type: ignore
                    max_tokens=1000,
                    temperature=1,  # Must use temperature=1 when thinking is enabled
                    **extra_params,
                )

                # Track follow-up token usage
                self.track_token_usage(
                    follow_up_response, turn, "Anthropic Expert (follow-up)"
                )

                if follow_up_response.choices:
                    content = follow_up_response.choices[0].message.content or ""

                    # Check if there are more tool calls (recursive handling)
                    if follow_up_response.choices[0].message.tool_calls:
                        # Recursively handle more tool calls
                        self.anthropic_messages.append(
                            {
                                "role": "user",
                                "content": "Please continue with your analysis and provide your findings.",
                            }
                        )
                        return await self.send_to_anthropic(
                            self.anthropic_messages, turn
                        )
                else:
                    content = "No response after tool usage"

            return content

        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            return f"Error: {e}"

    def print_conversation_start(self, topic: str, max_turns: int) -> None:
        """Print conversation start information."""
        if self.use_rich and self.console:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Label", style="bold cyan")
            table.add_column("Value", style="green")

            table.add_row("Topic:", topic)
            table.add_row("Max turns:", str(max_turns))
            table.add_row("Project root:", str(self.code_access.project_root))
            table.add_row("Proxy URL:", self.proxy_url)
            table.add_row("Streaming:", "enabled" if self.stream else "disabled")
            table.add_row("Thinking mode:", "enabled" if self.thinking else "disabled")

            panel = Panel(
                table,
                title="[bold blue]AI Code Discussion - Beginner/Expert[/bold blue]",
                border_style="blue",
                title_align="left",
            )

            self.console.print()
            self.console.print(panel)
            self.console.print()
        else:
            print("AI Code Discussion")
            print("=" * 60)
            print(f"Topic: {topic}")
            print(f"Max turns: {max_turns}")
            print(f"Project root: {self.code_access.project_root}")
            print(f"Proxy URL: {self.proxy_url}")
            print(f"Thinking mode: {'enabled' if self.thinking else 'disabled'}")
            print("=" * 60)

    def print_conversation_end(self, total_turns: int) -> None:
        """Print conversation end information."""
        if self.use_rich and self.console:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Label", style="bold cyan")
            table.add_column("Value", style="green")

            table.add_row("Total turns:", str(total_turns))
            table.add_row("OpenAI messages:", str(len(self.openai_messages)))
            table.add_row("Anthropic messages:", str(len(self.anthropic_messages)))
            table.add_row("Total tokens used:", str(self.total_tokens_used))

            panel = Panel(
                table,
                title="[bold green]Discussion Completed[/bold green]",
                border_style="green",
                title_align="left",
            )

            self.console.print()
            self.console.print(panel)
        else:
            print(f"\n{'=' * 60}")
            print("Discussion completed!")
            print(f"Total turns: {total_turns}")
            print(f"Total tokens used: {self.total_tokens_used}")
            print(f"{'=' * 60}")

        # Print detailed token summary
        if self.turn_tokens:
            self.print_token_summary()

    def _share_tool_interactions(
        self,
        from_messages: list,
        to_messages: list,
        start_idx: int,
        final_response: str,
    ) -> None:
        """Share tool calls and results between AI conversations."""
        # Copy any tool interactions that happened since start_idx
        for i in range(start_idx, len(from_messages)):
            msg = from_messages[i]

            # Handle both dictionary messages and ChatCompletionMessage objects
            if isinstance(msg, dict):
                role = msg.get("role")
                tool_calls = msg.get("tool_calls")
                # Copy dictionary message directly
                if (role == "assistant" and tool_calls) or role == "tool":
                    to_messages.append(msg)
            else:
                # Handle OpenAI ChatCompletionMessage objects
                try:
                    role = getattr(msg, "role", None)
                    tool_calls = getattr(msg, "tool_calls", None)

                    if role == "assistant" and tool_calls:
                        # Convert to dict for consistency
                        to_messages.append(
                            {
                                "role": "assistant",
                                "content": msg.content,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": tc.type,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in tool_calls
                                ],
                            }
                        )
                    elif role == "tool":
                        to_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        )
                except AttributeError:
                    # Skip if we can't process this message type
                    logger.warning(f"Skipping message with unknown format: {type(msg)}")
                    continue

        # Add the final response as user message
        to_messages.append({"role": "user", "content": final_response})

    async def run_code_discussion(self, topic: str, max_turns: int = 6) -> None:
        """Run the bidirectional code discussion."""
        self.print_conversation_start(topic, max_turns)
        self.add_initial_topic(topic)

        for turn in range(1, max_turns + 1):
            try:
                if turn % 2 == 1:  # Odd turns: OpenAI speaks
                    logger.debug(f"Turn {turn}: OpenAI speaking")

                    # Remember starting point for tool interaction sharing
                    openai_start_idx = len(self.openai_messages)

                    response = await self.send_to_openai(self.openai_messages, turn)

                    # Add final response to OpenAI's history
                    self.openai_messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # Share all tool interactions and final response with Anthropic
                    self._share_tool_interactions(
                        self.openai_messages,
                        self.anthropic_messages,
                        openai_start_idx,
                        response,
                    )

                    self.render_ai_message(
                        turn, "OpenAI Beginner (via proxy)", response
                    )

                else:  # Even turns: Anthropic speaks
                    logger.debug(f"Turn {turn}: Anthropic speaking")

                    # Remember starting point for tool interaction sharing
                    anthropic_start_idx = len(self.anthropic_messages)

                    response = await self.send_to_anthropic(
                        self.anthropic_messages, turn
                    )

                    # Add final response to Anthropic's history
                    self.anthropic_messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # Share all tool interactions and final response with OpenAI
                    self._share_tool_interactions(
                        self.anthropic_messages,
                        self.openai_messages,
                        anthropic_start_idx,
                        response,
                    )

                    self.render_ai_message(
                        turn, "Anthropic Expert (via proxy)", response
                    )

                # Small delay between turns
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Turn {turn} failed: {e}")
                self.render_error(f"Turn {turn} failed: {e}")
                break

        self.print_conversation_end(min(turn, max_turns))


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Code Discussion Demo - Beginner/Expert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ai_code_discussion_demo.py
  python3 ai_code_discussion_demo.py --topic "API architecture patterns"
  python3 ai_code_discussion_demo.py --turns 10 --debug
  python3 ai_code_discussion_demo.py --project-root /path/to/project
        """,
    )

    parser.add_argument(
        "--topic",
        default="the architecture and design patterns in this codebase",
        help="Topic for the beginner/expert code discussion (default: architecture and design patterns)",
    )

    parser.add_argument(
        "--turns",
        type=int,
        default=6,
        help="Maximum number of discussion turns between beginner and expert (default: 6)",
    )

    parser.add_argument(
        "--project-root",
        default=None,
        help="Root directory of the project (default: current directory)",
    )

    parser.add_argument(
        "--proxy-url",
        default="http://127.0.0.1:8000/api",
        help="Proxy server URL (default: http://127.0.0.1:8000/api)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming mode (future feature)"
    )

    parser.add_argument("--plain", action="store_true", help="Disable rich formatting")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable AI thinking mode to see reasoning process (requires ccproxy adapter fix)",
    )

    return parser.parse_args()


async def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging(args.debug)

    # Check environment
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not args.plain and RICH_AVAILABLE:
        console = Console()

        console.print(
            "\n[bold blue]AI Code Discussion Demo - Beginner/Expert[/bold blue]"
        )
        console.print("=" * 60)

        if not openai_key:
            console.print(
                "[yellow]Warning: OPENAI_API_KEY not set, using dummy key[/yellow]"
            )
        if not anthropic_key:
            console.print(
                "[yellow]Warning: ANTHROPIC_API_KEY not set, using dummy key[/yellow]"
            )

        console.print("=" * 60)
    else:
        print("AI Code Discussion Demo - Beginner/Expert")
        print("=" * 60)

        if not openai_key:
            print("Warning: OPENAI_API_KEY not set, using dummy key")
        if not anthropic_key:
            print("Warning: ANTHROPIC_API_KEY not set, using dummy key")

        print("=" * 60)

    try:
        manager = AICodeDiscussionManager(
            project_root=args.project_root,
            proxy_url=args.proxy_url,
            debug=args.debug,
            stream=args.stream,
            use_rich=not args.plain,
            thinking=args.thinking,
        )

        await manager.run_code_discussion(args.topic, args.turns)

    except KeyboardInterrupt:
        if not args.plain and RICH_AVAILABLE:
            console = Console()
            console.print("\n[yellow]Discussion interrupted by user[/yellow]")
        else:
            print("\nDiscussion interrupted by user")
    except Exception as e:
        if not args.plain and RICH_AVAILABLE:
            console = Console()
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            console.print(
                "[yellow]Make sure your proxy server is running and accessible[/yellow]"
            )
        else:
            print(f"\nError: {e}")
            print("Make sure your proxy server is running and accessible")
        logger.error(f"main_error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
