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
from typing import Any, Optional

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
    ):
        self.project_root = project_root or str(pathlib.Path.cwd())
        self.proxy_url = proxy_url
        self.debug = debug
        self.stream = stream
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
            {
                "type": "custom",
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
            {
                "type": "custom",
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
        ]

        # System prompts for beginner/expert roles
        self.beginner_system_prompt = """You are a curious beginner developer who is learning about code architecture and patterns. Your role is to:
- Actively explore the codebase using available tools (text editor for reading files, bash for searching)
- Use multiple tool requests to thoroughly investigate areas of interest
- Use bash commands like: 'find', 'fd', 'rg', 'cat', 'xargs' to explore the codebase efficiently
- Use the text editor to read specific files you discover
- After gathering information with tools, ask thoughtful questions about what you discovered
- Show genuine curiosity about how things work
- Ask for clarification when explanations are complex
- Follow up with related questions to deepen understanding
- Keep questions focused and specific after your investigation
- Express enthusiasm for learning

Available tools: text editor (read files), bash (rg, fd, find, cat, xargs for searching/exploring)
Workflow: Use tools → Analyze findings → Ask specific questions based on your discoveries
Ask questions like "I noticed X pattern in the code, how does this work?", "Why was this approach chosen?", "What are the trade-offs?"

Always explore the code first with tools, then ask informed questions about your findings."""

        self.expert_system_prompt = """You are an experienced software architect and mentor who explains code concepts clearly and concisely. Your role is to:
- First explore the codebase using available tools (text editor for reading files, bash for searching)
- Use multiple tool requests to thoroughly investigate the areas being discussed
- Use bash commands like: 'find', 'fd', 'rg', 'cat', 'xargs' to efficiently analyze the codebase
- Use the text editor to read specific files for detailed analysis
- After gathering comprehensive information with tools, provide clear, practical explanations
- Keep responses concise but informative
- Use simple language and avoid jargon when possible
- Give concrete examples from the code you examined
- Always end with a question to encourage further discussion
- Be patient and encouraging
- Focus on practical understanding over theoretical details

Available tools: text editor (read files), bash (rg, fd, find, cat, xargs for searching/analyzing)
Workflow: Use tools to investigate → Analyze the code → Provide informed explanations with examples
Your explanations should be brief (2-3 sentences) but complete, backed by actual code findings. Always finish with a follow-up question to keep the conversation going."""

        # Conversation histories - both use OpenAI format now
        self.openai_messages: list[ChatCompletionMessageParam] = []
        self.anthropic_messages: list[ChatCompletionMessageParam] = []

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
        """Render file content with syntax highlighting."""
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

            # Truncate very long files for display
            content = result["content"]
            if len(content) > 5000:
                content = content[:5000] + "\n\n... (truncated)"

            syntax = Syntax(
                content, lexer, theme="monokai", line_numbers=True, word_wrap=True
            )

            panel = Panel(
                syntax,
                title=f"📄 {result['path']} ({result['size']} bytes)",
                title_align="left",
                border_style="blue",
                expand=False,
            )

            self.console.print(panel)
        else:
            print(f"\n--- File: {result['path']} ({result['size']} bytes) ---")
            content = result["content"]
            if len(content) > 2000:
                content = content[:2000] + "\n\n... (truncated)"
            print(content)
            print("--- End of file ---")

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
        beginner_initial_message = f"""Let's explore this codebase together: {topic}

I have access to tools to read files and explore the project structure. As a beginner, I want to understand:
- The overall architecture and structure
- How different components work together
- What patterns are being used

Let me start by exploring the codebase structure to get a better understanding of what we're working with."""

        # Anthropic will be the expert (responds to questions)
        expert_initial_message = f"""I'm here to help you understand this codebase: {topic}

As an experienced developer, I'll investigate the code thoroughly using the available tools to provide you with detailed explanations about architecture, patterns, and implementation details.

I'm ready to explore the codebase and share my findings with you!"""

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

    async def send_to_openai(
        self, messages: list[ChatCompletionMessageParam], turn: int
    ) -> str:
        """Send messages to OpenAI with tool support via unified endpoint."""
        logger.debug(f"Sending to OpenAI, turn {turn}, message count: {len(messages)}")

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,  # type: ignore
                max_tokens=1000,
                temperature=0.7,
            )

            if not response.choices:
                raise Exception("No choices in OpenAI response")

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
                follow_up_response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=self.openai_messages,
                    tools=self.tools,  # type: ignore
                    max_tokens=1000,
                    temperature=0.7,
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
            response = self.anthropic_client.chat.completions.create(
                model="claude-3-5-sonnet-20241022",
                messages=messages,
                tools=self.tools,  # type: ignore
                max_tokens=1000,
                temperature=0.7,
            )

            if not response.choices:
                raise Exception("No choices in Anthropic response")

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
                follow_up_response = self.anthropic_client.chat.completions.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=self.anthropic_messages,
                    tools=self.tools,  # type: ignore
                    max_tokens=1000,
                    temperature=0.7,
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
            print(f"{'=' * 60}")

    async def run_code_discussion(self, topic: str, max_turns: int = 6) -> None:
        """Run the bidirectional code discussion."""
        self.print_conversation_start(topic, max_turns)
        self.add_initial_topic(topic)

        for turn in range(1, max_turns + 1):
            try:
                if turn % 2 == 1:  # Odd turns: OpenAI speaks
                    logger.debug(f"Turn {turn}: OpenAI speaking")

                    response = await self.send_to_openai(self.openai_messages, turn)

                    # Add to OpenAI's history
                    self.openai_messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # Add to Anthropic's history (with role swap)
                    self.anthropic_messages.append(
                        {"role": "user", "content": response}
                    )

                    self.render_ai_message(
                        turn, "OpenAI Beginner (via proxy)", response
                    )

                else:  # Even turns: Anthropic speaks
                    logger.debug(f"Turn {turn}: Anthropic speaking")

                    response = await self.send_to_anthropic(
                        self.anthropic_messages, turn
                    )

                    # Add to Anthropic's history
                    self.anthropic_messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # Add to OpenAI's history (with role swap)
                    self.openai_messages.append({"role": "user", "content": response})

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
