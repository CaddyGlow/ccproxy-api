"""File formatter for command replay output."""

import stat
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from ccproxy.core.logging import get_plugin_logger
from ccproxy.utils.command_line import generate_curl_shell_script


logger = get_plugin_logger()


class CommandFileFormatter:
    """Formats and writes command replay data to files."""

    def __init__(
        self,
        log_dir: str = "/tmp/ccproxy/command_replay",
        enabled: bool = True,
        separate_files_per_command: bool = False,
    ) -> None:
        """Initialize with configuration.

        Args:
            log_dir: Directory for command replay files
            enabled: Enable file writing
            separate_files_per_command: Create separate files for curl/xh vs combined
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.separate_files_per_command = separate_files_per_command

        if self.enabled:
            # Create log directory if it doesn't exist
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(
                    "failed_to_create_command_replay_directory",
                    log_dir=str(self.log_dir),
                    error=str(e),
                    exc_info=e,
                )
                # Disable file writing if we can't create the directory
                self.enabled = False

        # Track which files we've already created (for logging purposes only)
        self._created_files: set[str] = set()

    def _compose_file_id(self, request_id: str | None) -> str:
        """Generate base file ID from request ID.

        Args:
            request_id: Request ID for correlation

        Returns:
            Base file ID string
        """
        if request_id:
            # Clean up request ID for filesystem safety
            safe_id = "".join(
                c if c.isalnum() or c in "-_" else "_" for c in request_id
            )
            return safe_id[:50]  # Limit length
        else:
            return str(uuid.uuid4())[:8]

    def _compose_file_id_with_timestamp(self, request_id: str | None) -> str:
        """Build filename ID with timestamp suffix for better organization.

        Format: {base_id}_{timestamp}_{nanos}
        Where timestamp is in format: YYYYMMDD_HHMMSS_microseconds
        And nanos is a counter to prevent collisions
        """
        base_id = self._compose_file_id(request_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Add a high-resolution timestamp with nanoseconds for uniqueness
        nanos = time.time_ns() % 1000000  # Get nanosecond portion
        return f"{base_id}_{timestamp}_{nanos:06d}"

    def should_write_files(self) -> bool:
        """Check if file writing is enabled."""
        return bool(self.enabled)

    async def write_commands(
        self,
        request_id: str,
        curl_command: str,
        xh_command: str,
        provider: str | None = None,
        timestamp_prefix: str | None = None,
        method: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        body: Any = None,
        is_json: bool = False,
    ) -> list[str]:
        """Write command replay data to files.

        Args:
            request_id: Request ID for correlation
            curl_command: Generated curl command
            xh_command: Generated xh command
            provider: Provider name (anthropic, openai, etc.)
            timestamp_prefix: Optional timestamp prefix from RequestContext
            method: HTTP method for shell script generation
            url: Request URL for shell script generation
            headers: HTTP headers for shell script generation
            body: Request body for shell script generation
            is_json: Whether body is JSON for shell script generation

        Returns:
            List of file paths that were written
        """
        if not self.enabled:
            return []

        written_files = []

        # Use provided timestamp prefix or generate our own
        if timestamp_prefix:
            base_id = f"{self._compose_file_id(request_id)}_{timestamp_prefix}"
        else:
            base_id = self._compose_file_id_with_timestamp(request_id)

        # Add provider to filename if available
        if provider:
            base_id = f"{base_id}_{provider}"

        try:
            if self.separate_files_per_command:
                # Write separate files for curl and xh
                if curl_command:
                    curl_file = await self._write_single_command_file(
                        base_id, "curl", curl_command, request_id
                    )
                    if curl_file:
                        written_files.append(curl_file)

                if xh_command:
                    xh_file = await self._write_single_command_file(
                        base_id, "xh", xh_command, request_id
                    )
                    if xh_file:
                        written_files.append(xh_file)
            else:
                # Write combined file with both commands
                combined_file = await self._write_combined_command_file(
                    base_id, curl_command, xh_command, request_id, provider
                )
                if combined_file:
                    written_files.append(combined_file)

            # Generate executable shell script if we have raw request data
            if method and url:
                shell_script_file = await self._write_shell_script_file(
                    base_id, request_id, method, url, headers, body, is_json, provider
                )
                if shell_script_file:
                    written_files.append(shell_script_file)

            # Make files executable
            await self._make_files_executable(written_files)

        except Exception as e:
            logger.error(
                "command_replay_file_write_error",
                request_id=request_id,
                error=str(e),
                exc_info=e,
            )

        return written_files

    async def _write_single_command_file(
        self,
        base_id: str,
        command_type: str,
        command: str,
        request_id: str,
    ) -> str | None:
        """Write a single command to its own file.

        Args:
            base_id: Base filename identifier
            command_type: Command type (curl, xh)
            command: Command string to write
            request_id: Request ID for logging

        Returns:
            File path if successful, None if failed
        """
        file_path = self.log_dir / f"{base_id}_{command_type}.sh"

        # Log file creation (only once per unique file path)
        if str(file_path) not in self._created_files:
            self._created_files.add(str(file_path))
            logger.debug(
                "command_replay_file_created",
                request_id=request_id,
                command_type=command_type,
                file_path=str(file_path),
                mode="separate",
            )

        try:
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write("#!/usr/bin/env bash\n")
                await f.write(f"# {command_type.upper()} Command Replay\n")
                await f.write(f"# Request ID: {request_id}\n")
                await f.write(f"# Generated: {datetime.now().isoformat()}\n")
                await f.write("#\n")
                await f.write(
                    f"# Run this file directly: ./{base_id}_{command_type}.sh\n"
                )
                await f.write("\n")
                await f.write(command)
                await f.write("\n")

            return str(file_path)

        except Exception as e:
            logger.error(
                "command_replay_single_file_write_error",
                request_id=request_id,
                command_type=command_type,
                file_path=str(file_path),
                error=str(e),
            )
            return None

    async def _write_combined_command_file(
        self,
        base_id: str,
        curl_command: str,
        xh_command: str,
        request_id: str,
        provider: str | None = None,
    ) -> str | None:
        """Write both commands to a single combined file.

        Args:
            base_id: Base filename identifier
            curl_command: curl command string
            xh_command: xh command string
            request_id: Request ID for logging
            provider: Provider name for header

        Returns:
            File path if successful, None if failed
        """
        file_path = self.log_dir / f"{base_id}_commands.sh"

        # Log file creation (only once per unique file path)
        if str(file_path) not in self._created_files:
            self._created_files.add(str(file_path))
            logger.debug(
                "command_replay_file_created",
                request_id=request_id,
                command_type="combined",
                file_path=str(file_path),
                mode="combined",
            )

        try:
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                # Write shebang and header
                await f.write("#!/usr/bin/env bash\n")
                await f.write("# Command Replay File\n")
                await f.write(f"# Request ID: {request_id}\n")
                if provider:
                    await f.write(f"# Provider: {provider}\n")
                await f.write(f"# Generated: {datetime.now().isoformat()}\n")
                await f.write("#\n")
                await f.write("# This file contains both curl and xh commands.\n")
                await f.write("# Uncomment the command you want to run.\n")
                await f.write("\n")

                # Write curl command
                if curl_command:
                    await f.write("# CURL Command\n")
                    await f.write("# " + "=" * 50 + "\n")
                    # Comment out the command so it doesn't run accidentally
                    for line in curl_command.split("\n"):
                        if line.strip():
                            await f.write(f"# {line}\n")
                        else:
                            await f.write("#\n")
                    await f.write("\n")

                # Write xh command
                if xh_command:
                    await f.write("# XH Command\n")
                    await f.write("# " + "=" * 50 + "\n")
                    # Comment out the command so it doesn't run accidentally
                    for line in xh_command.split("\n"):
                        if line.strip():
                            await f.write(f"# {line}\n")
                        else:
                            await f.write("#\n")
                    await f.write("\n")

                # Add footer with instructions
                await f.write("# " + "=" * 60 + "\n")
                await f.write("# Instructions:\n")
                await f.write("# 1. Uncomment the command you want to use\n")
                await f.write("# 2. Make sure you have curl or xh installed\n")
                await f.write("# 3. Run: chmod +x this_file.sh && ./this_file.sh\n")
                await f.write("# " + "=" * 60 + "\n")

            return str(file_path)

        except Exception as e:
            logger.error(
                "command_replay_combined_file_write_error",
                request_id=request_id,
                file_path=str(file_path),
                error=str(e),
            )
            return None

    def get_log_dir(self) -> str:
        """Get the log directory path."""
        return str(self.log_dir)

    async def _make_files_executable(self, file_paths: list[str]) -> None:
        """Make the generated files executable.

        Args:
            file_paths: List of file paths to make executable
        """

        for file_path_str in file_paths:
            try:
                file_path = Path(file_path_str)
                # Add execute permission for owner, group, and others
                current_mode = file_path.stat().st_mode
                new_mode = current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                file_path.chmod(new_mode)

                logger.debug(
                    "command_replay_file_made_executable",
                    file_path=file_path_str,
                )
            except Exception as e:
                logger.warning(
                    "command_replay_chmod_failed",
                    file_path=file_path_str,
                    error=str(e),
                )

    async def _write_shell_script_file(
        self,
        base_id: str,
        request_id: str,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        body: Any,
        is_json: bool,
        provider: str | None = None,
    ) -> str | None:
        """Write an executable shell script file.

        Args:
            base_id: Base filename identifier
            request_id: Request ID for logging
            method: HTTP method
            url: Request URL
            headers: HTTP headers
            body: Request body
            is_json: Whether body is JSON
            provider: Provider name

        Returns:
            File path if successful, None if failed
        """
        file_path = self.log_dir / f"{base_id}_script.sh"

        # Log file creation
        if str(file_path) not in self._created_files:
            self._created_files.add(str(file_path))
            logger.debug(
                "command_replay_file_created",
                request_id=request_id,
                command_type="shell_script",
                file_path=str(file_path),
                mode="executable",
            )

        try:
            # Generate shell-safe script content
            script_content = generate_curl_shell_script(
                method=method,
                url=url,
                headers=headers,
                body=body,
                is_json=is_json,
            )

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write("#!/bin/bash\n")
                await f.write("# Executable Shell Script for Request Replay\n")
                await f.write(f"# Request ID: {request_id}\n")
                if provider:
                    await f.write(f"# Provider: {provider}\n")
                await f.write(f"# Generated: {datetime.now().isoformat()}\n")
                await f.write(f"# Usage: bash {file_path.name} or ./{file_path.name}\n")
                await f.write("\n")
                await f.write(script_content)

            return str(file_path)

        except Exception as e:
            logger.error(
                "command_replay_shell_script_write_error",
                request_id=request_id,
                file_path=str(file_path),
                error=str(e),
            )
            return None

    def cleanup(self) -> None:
        """Clean up resources (if any)."""
        self._created_files.clear()
