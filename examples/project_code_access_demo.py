#!/usr/bin/env python3
"""
Project Code Access Demo

This script demonstrates a system that exposes tools to access project code
with security boundaries. It provides read_file and list_file tools with
recursive options, limited to the current project directory.

The system acts as a bridge between clients and project code, allowing
controlled access to read files and list directories within the project root.
"""

import argparse
import asyncio
import logging
import os
import pathlib
from typing import Any

import anthropic


try:
    from rich.console import Console
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


class ProjectCodeAccessSystem:
    """Manages secure access to project code files and directories."""

    def __init__(self, project_root: str | None = None):
        self.project_root = pathlib.Path(project_root or pathlib.Path.cwd()).resolve()
        self.console = Console() if RICH_AVAILABLE else None

        # Validate project root exists
        if not self.project_root.exists():
            raise ProjectCodeAccessError(
                f"Project root does not exist: {self.project_root}"
            )

        logger.info(f"Project code access initialized for: {self.project_root}")

    def _validate_path(self, path: str) -> pathlib.Path:
        """Validate that a path is within the project root."""
        try:
            # Resolve the path relative to project root
            if pathlib.Path(path).is_absolute():
                resolved_path = pathlib.Path(path).resolve()
            else:
                resolved_path = (self.project_root / path).resolve()

            # Ensure path is within project root
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

            # Try to read as text first
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
                # Handle binary files
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
                # Recursive listing
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
                # Non-recursive listing
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

            # Sort items: directories first, then files, alphabetically
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


class ProjectCodeRenderer:
    """Renders project code access results with rich formatting."""

    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None

    def render_file_content(self, result: dict[str, Any]) -> None:
        """Render file content with syntax highlighting."""
        if not result["success"]:
            self._render_error(result)
            return

        if self.use_rich:
            # Try to detect file type for syntax highlighting
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

            syntax = Syntax(
                result["content"],
                lexer,
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            )

            panel = Panel(
                syntax,
                title=f"📄 {result['path']} ({result['size']} bytes)",
                title_align="left",
                border_style="blue",
            )

            if self.console:
                self.console.print(panel)
        else:
            print(f"\n{'=' * 60}")
            print(f"File: {result['path']} ({result['size']} bytes)")
            print(f"{'=' * 60}")
            print(result["content"])
            print(f"{'=' * 60}")

    def render_directory_listing(self, result: dict[str, Any]) -> None:
        """Render directory listing with formatted output."""
        if not result["success"]:
            self._render_error(result)
            return

        if self.use_rich:
            # Create table
            table = Table(
                title=f"📁 Directory: {result['path']} ({'recursive' if result['recursive'] else 'non-recursive'})",
                show_header=True,
                header_style="bold cyan",
            )

            table.add_column("Name", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Size", style="magenta", justify="right")
            table.add_column("Path", style="dim")

            for item in result["items"]:
                size_str = f"{item['size']} bytes" if item["type"] == "file" else ""
                type_icon = "📄" if item["type"] == "file" else "📁"

                table.add_row(
                    f"{type_icon} {item['name']}", item["type"], size_str, item["path"]
                )

            if self.console:
                self.console.print(table)
                self.console.print(f"\nTotal items: {result['total_count']}")
        else:
            print(f"\n{'=' * 60}")
            print(
                f"Directory: {result['path']} ({'recursive' if result['recursive'] else 'non-recursive'})"
            )
            print(f"{'=' * 60}")

            for item in result["items"]:
                type_indicator = "D" if item["type"] == "directory" else "F"
                size_str = f" ({item['size']} bytes)" if item["type"] == "file" else ""
                print(f"[{type_indicator}] {item['name']}{size_str}")

            print(f"\nTotal items: {result['total_count']}")
            print(f"{'=' * 60}")

    def _render_error(self, result: dict[str, Any]) -> None:
        """Render error messages."""
        if self.use_rich and self.console:
            panel = Panel(
                Text(result["error"], style="bold red"),
                title="❌ Error",
                title_align="left",
                border_style="red",
            )
            if self.console:
                self.console.print(panel)
        else:
            print(f"\nError: {result['error']}")


class ProjectCodeAccessDemo:
    """Main demo class that integrates with AI clients."""

    def __init__(
        self,
        project_root: str | None = None,
        proxy_url: str = "http://127.0.0.1:8000/api",
        use_rich: bool = True,
    ):
        self.project_root = project_root or str(pathlib.Path.cwd())
        self.proxy_url = proxy_url
        self.use_rich = use_rich

        # Initialize components
        self.access_system = ProjectCodeAccessSystem(project_root)
        self.renderer = ProjectCodeRenderer(use_rich)

        # Initialize AI clients
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "dummy"), base_url=f"{proxy_url}"
        )

        # Available tools for AI
        self.tools = [
            {
                "name": "read_file",
                "description": "Read the contents of a file from the project directory",
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
                "name": "list_files",
                "description": "List files and directories in the project directory",
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
                    },
                    "required": [],
                },
            },
        ]

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool and return the result."""
        if tool_name == "read_file":
            return self.access_system.read_file(kwargs.get("file_path", ""))
        elif tool_name == "list_files":
            return self.access_system.list_files(
                kwargs.get("directory_path", "."), kwargs.get("recursive", False)
            )
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def print_welcome(self) -> None:
        """Print welcome message."""
        if self.use_rich:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Label", style="bold cyan")
            table.add_column("Value", style="green")

            table.add_row("Project Root:", str(self.access_system.project_root))
            table.add_row("Proxy URL:", self.proxy_url)

            panel = Panel(
                table,
                title="[bold blue]Project Code Access Demo[/bold blue]",
                border_style="blue",
                title_align="left",
            )

            if self.renderer.console:
                self.renderer.console.print(panel)
        else:
            print("Project Code Access Demo")
            print("=" * 60)
            print(f"Project Root: {self.access_system.project_root}")
            print(f"Proxy URL: {self.proxy_url}")
            print("=" * 60)

    async def interactive_demo(self) -> None:
        """Run an interactive demo."""
        self.print_welcome()

        while True:
            try:
                if self.use_rich and self.renderer.console:
                    self.renderer.console.print(
                        "\n[bold yellow]Available commands:[/bold yellow]"
                    )
                    self.renderer.console.print(
                        "  read <file_path>       - Read a file"
                    )
                    self.renderer.console.print(
                        "  list [directory] [-r]  - List directory contents"
                    )
                    self.renderer.console.print(
                        "  help                   - Show this help"
                    )
                    self.renderer.console.print("  quit                   - Exit demo")
                    self.renderer.console.print()
                else:
                    print("\nAvailable commands:")
                    print("  read <file_path>       - Read a file")
                    print("  list [directory] [-r]  - List directory contents")
                    print("  help                   - Show this help")
                    print("  quit                   - Exit demo")
                    print()

                command = input("Enter command: ").strip()

                if not command:
                    continue

                if command.lower() in ["quit", "exit", "q"]:
                    break

                if command.lower() == "help":
                    continue

                parts = command.split()
                cmd = parts[0].lower()

                if cmd == "read":
                    if len(parts) < 2:
                        print("Usage: read <file_path>")
                        continue

                    file_path = parts[1]
                    result = self.execute_tool("read_file", file_path=file_path)
                    self.renderer.render_file_content(result)

                elif cmd == "list":
                    directory = (
                        parts[1]
                        if len(parts) > 1 and not parts[1].startswith("-")
                        else "."
                    )
                    recursive = "-r" in parts or "--recursive" in parts

                    result = self.execute_tool(
                        "list_files", directory_path=directory, recursive=recursive
                    )
                    self.renderer.render_directory_listing(result)

                else:
                    print(f"Unknown command: {cmd}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    async def ai_demo(self, query: str) -> None:
        """Run AI demo with a specific query."""
        self.print_welcome()

        if self.use_rich and self.renderer.console:
            self.renderer.console.print(f"\n[bold green]AI Query:[/bold green] {query}")
        else:
            print(f"\nAI Query: {query}")

        print("AI integration demo - would send query to AI with tools")
        print("For now, you can use the interactive mode to test the tools directly")


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Project Code Access Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 project_code_access_demo.py --interactive
  python3 project_code_access_demo.py --query "Show me the main Python files"
  python3 project_code_access_demo.py --query "What's in the examples directory?"
  python3 project_code_access_demo.py --project-root /path/to/project --interactive
        """,
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

    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    parser.add_argument("--query", help="Query to send to AI")

    parser.add_argument("--plain", action="store_true", help="Disable rich formatting")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


async def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging(args.debug)

    try:
        demo = ProjectCodeAccessDemo(
            project_root=args.project_root,
            proxy_url=args.proxy_url,
            use_rich=not args.plain,
        )

        if args.interactive:
            await demo.interactive_demo()
        elif args.query:
            await demo.ai_demo(args.query)
        else:
            print("Please specify either --interactive or --query")
            return

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"main_error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
