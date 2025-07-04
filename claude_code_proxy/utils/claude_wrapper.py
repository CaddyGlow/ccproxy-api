"""Secure wrapper for claude command execution with response parsing."""

import json
import logging
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .subprocess_security import SubprocessSecurity, get_default_claude_security


logger = logging.getLogger(__name__)


class ClaudeWrapper:
    """Secure wrapper for executing claude commands with response parsing."""

    def __init__(
        self,
        security: Optional[SubprocessSecurity] = None,
        working_directory: Optional[str] = None,
        claude_path: Optional[str] = None,
    ):
        """
        Initialize the Claude wrapper.

        Args:
            security: Optional SubprocessSecurity instance for custom security settings
            working_directory: Optional working directory for claude execution
            claude_path: Optional path to claude executable (defaults to 'claude' in PATH)
        """
        self.security = security or get_default_claude_security(working_directory)
        self.claude_path = claude_path or self._find_claude_executable()
        self._process = None

    def _find_claude_executable(self) -> str:
        """Find the claude executable in the system PATH."""
        import shutil

        # Try to find claude in PATH
        claude_path = shutil.which("claude")
        if claude_path:
            return claude_path

        # Common installation paths
        common_paths = [
            "/usr/local/bin/claude",
            "/usr/bin/claude",
            Path.home() / ".local/bin/claude",
            Path.home() / "bin/claude",
            # Check current working directory node_modules
            Path.cwd() / "node_modules" / ".bin" / "claude",
        ]

        for path in common_paths:
            if Path(path).exists():
                return str(Path(path).resolve())  # Convert to absolute path

        # Default to 'claude' and let the system handle the error
        return "claude"

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful process termination."""

        def signal_handler(signum, frame):
            if self._process and self._process.poll() is None:
                logger.info("Received signal, terminating claude process...")
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Process did not terminate gracefully, killing...")
                    self._process.kill()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def execute_status(self) -> Dict[str, Any]:
        """
        Execute 'claude /status' command and parse the response.

        Returns:
            Parsed status response as a dictionary
        """
        cmd = [self.claude_path, "/status"]

        try:
            self._setup_signal_handlers()

            # Execute the command with security settings
            result = self.security.secure_run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,  # Shorter timeout for non-interactive
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Claude command failed with return code {result.returncode}: {result.stderr}"
                )

            # Parse the response
            return self._parse_status_response(result.stdout)

        except Exception as e:
            logger.error(f"Error executing claude /status: {e}")
            raise

    def execute_interactive_status(self) -> Dict[str, Any]:
        """
        Execute 'claude /status' command with interactive "Press Enter to continue" handling.

        Returns:
            Parsed status response as a dictionary
        """
        cmd = [self.claude_path, "/status"]

        try:
            self._setup_signal_handlers()

            # Create a Popen object for interactive handling
            self._process = self.security.secure_popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            output = []
            timeout_counter = 0
            max_timeout = 60  # 60 seconds max

            # Handle the interactive process
            while True:
                # Check if process has terminated
                if self._process.poll() is not None:
                    break

                # Timeout check
                timeout_counter += 1
                if timeout_counter > max_timeout:
                    print(
                        "→ Timeout waiting for claude output, terminating...",
                        flush=True,
                    )
                    self._process.terminate()
                    break

                # Read output line by line with timeout
                import select
                import sys

                # Use select to check if there's data to read (non-blocking)
                if hasattr(select, "select"):
                    ready, _, _ = select.select([self._process.stdout], [], [], 1.0)
                    if not ready:
                        continue
                else:
                    # Fallback for systems without select
                    import time

                    time.sleep(0.1)

                line = self._process.stdout.readline()
                if not line:
                    continue

                output.append(line)
                timeout_counter = 0  # Reset timeout on activity

                # Print output in real-time
                print(line.rstrip(), flush=True)

                # Check for "Press Enter to continue" prompt
                if (
                    "Press Enter to continue" in line
                    or "press enter to continue" in line.lower()
                ):
                    print(
                        "→ Detected 'Press Enter to continue' prompt. Sending Enter...",
                        flush=True,
                    )
                    self._process.stdin.write("\n")
                    self._process.stdin.flush()
                    continue

            # Wait for process to complete
            self._process.wait()

            # Get any remaining output
            remaining_output, stderr = self._process.communicate()
            if remaining_output:
                output.append(remaining_output)

            if self._process.returncode != 0:
                raise RuntimeError(
                    f"Claude command failed with return code {self._process.returncode}: {stderr}"
                )

            # Parse the combined output
            full_output = "".join(output)
            return self._parse_status_response(full_output)

        except Exception as e:
            logger.error(f"Error executing interactive claude /status: {e}")
            if self._process and self._process.poll() is None:
                self._process.terminate()
            raise
        finally:
            self._process = None

    def _parse_status_response(self, output: str) -> Dict[str, Any]:
        """
        Parse the claude /status response.

        Args:
            output: Raw output from claude /status command

        Returns:
            Parsed response as a dictionary
        """
        try:
            # Try to parse as JSON first
            if output.strip().startswith("{"):
                return json.loads(output)

            # If not JSON, parse as structured text
            parsed = {}
            lines = output.strip().split("\n")

            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for section headers
                if line.endswith(":") and not line.startswith(" "):
                    current_section = line[:-1].lower().replace(" ", "_")
                    parsed[current_section] = {}
                    continue

                # Parse key-value pairs
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip()

                    if current_section:
                        parsed[current_section][key] = value
                    else:
                        parsed[key] = value
                else:
                    # Handle list items or other formats
                    if current_section:
                        if "items" not in parsed[current_section]:
                            parsed[current_section]["items"] = []
                        parsed[current_section]["items"].append(line)
                    else:
                        if "output" not in parsed:
                            parsed["output"] = []
                        parsed["output"].append(line)

            return parsed

        except json.JSONDecodeError:
            # If parsing fails, return raw output
            return {
                "raw_output": output,
                "parsed": False,
                "error": "Failed to parse response",
            }
        except Exception as e:
            logger.error(f"Error parsing claude status response: {e}")
            return {"raw_output": output, "parsed": False, "error": str(e)}

    def execute_command(
        self, command: str, interactive: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a general claude command.

        Args:
            command: Claude command to execute (e.g., "/status", "/help")
            interactive: Whether to handle interactive prompts

        Returns:
            Parsed response as a dictionary
        """
        if command == "/status":
            return (
                self.execute_interactive_status()
                if interactive
                else self.execute_status()
            )

        cmd = [self.claude_path, command]

        try:
            self._setup_signal_handlers()

            if interactive:
                # Use interactive handling for any command
                self._process = self.security.secure_popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                output = []

                while True:
                    if self._process.poll() is not None:
                        break

                    line = self._process.stdout.readline()
                    if not line:
                        break

                    output.append(line)

                    if (
                        "Press Enter to continue" in line
                        or "press enter to continue" in line.lower()
                    ):
                        print(
                            "Detected 'Press Enter to continue' prompt. Sending Enter..."
                        )
                        self._process.stdin.write("\n")
                        self._process.stdin.flush()
                        continue

                    print(line.rstrip())

                self._process.wait()
                remaining_output, stderr = self._process.communicate()
                if remaining_output:
                    output.append(remaining_output)

                if self._process.returncode != 0:
                    raise RuntimeError(
                        f"Claude command failed with return code {self._process.returncode}: {stderr}"
                    )

                full_output = "".join(output)
                return {"output": full_output, "command": command, "interactive": True}

            else:
                # Non-interactive execution
                result = self.security.secure_run(
                    cmd, capture_output=True, text=True, timeout=30
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"Claude command failed with return code {result.returncode}: {result.stderr}"
                    )

                return {
                    "output": result.stdout,
                    "command": command,
                    "interactive": False,
                }

        except Exception as e:
            logger.error(f"Error executing claude command {command}: {e}")
            if self._process and self._process.poll() is None:
                self._process.terminate()
            raise
        finally:
            self._process = None


# Convenience function for quick usage
def create_claude_wrapper(
    claude_path: Optional[str] = None,
    working_directory: Optional[str] = None,
    user: Optional[str] = None,
    group: Optional[str] = None,
) -> ClaudeWrapper:
    """
    Create a ClaudeWrapper instance with optional custom settings.

    Args:
        claude_path: Path to claude executable
        working_directory: Working directory for execution
        user: User to drop privileges to
        group: Group to drop privileges to

    Returns:
        ClaudeWrapper instance
    """
    security = None
    if user or group or working_directory:
        security = SubprocessSecurity(
            user=user,
            group=group,
            working_directory=working_directory,
        )

    return ClaudeWrapper(
        security=security,
        working_directory=working_directory,
        claude_path=claude_path,
    )
