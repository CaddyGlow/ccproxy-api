"""Security utilities for subprocess execution with privilege dropping."""

import grp
import logging
import os
import pwd
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class SubprocessSecurity:
    """Handles secure subprocess execution with privilege dropping and CWD control."""

    def __init__(
        self,
        user: str | None = None,
        group: str | None = None,
        working_directory: str | Path | None = None,
        environment: dict[str, str] | None = None,
    ):
        """
        Initialize subprocess security configuration.

        Args:
            user: Username to drop privileges to (e.g., 'claude')
            group: Group name to drop privileges to (e.g., 'claude')
            working_directory: Working directory for subprocess
            environment: Environment variables to set
        """
        self.user = user
        self.group = group
        self.working_directory = working_directory
        self.environment = environment or {}

        # Validate user and group exist
        self._validate_user_group()

        # Validate working directory
        self._validate_working_directory()

    def _validate_user_group(self) -> None:
        """Validate that the specified user and group exist."""
        if self.user:
            try:
                pwd.getpwnam(self.user)
            except KeyError as err:
                raise ValueError(f"User '{self.user}' does not exist") from err

        if self.group:
            try:
                grp.getgrnam(self.group)
            except KeyError as err:
                raise ValueError(f"Group '{self.group}' does not exist") from err

    def _validate_working_directory(self) -> None:
        """Validate that the working directory exists and is accessible."""
        if self.working_directory:
            path = Path(self.working_directory)
            if not path.exists():
                raise ValueError(
                    f"Working directory '{self.working_directory}' does not exist"
                )
            if not path.is_dir():
                raise ValueError(
                    f"Working directory '{self.working_directory}' is not a directory"
                )
            if not os.access(self.working_directory, os.R_OK | os.X_OK):
                raise ValueError(
                    f"Working directory '{self.working_directory}' is not accessible"
                )

    def _drop_privileges(self) -> None:
        """Drop privileges to the specified user and group."""
        if self.group:
            group_info = grp.getgrnam(self.group)
            os.setgid(group_info.gr_gid)
            logger.debug(
                f"Dropped group privileges to {self.group} (gid: {group_info.gr_gid})"
            )

        if self.user:
            user_info = pwd.getpwnam(self.user)
            os.setuid(user_info.pw_uid)
            logger.debug(
                f"Dropped user privileges to {self.user} (uid: {user_info.pw_uid})"
            )

    def get_preexec_fn(self) -> Callable[[], None]:
        """Get the preexec_fn for subprocess calls."""

        def preexec_fn() -> None:
            """Function to run before exec in the child process."""
            try:
                # Change working directory first
                if self.working_directory:
                    os.chdir(self.working_directory)
                    logger.debug(
                        f"Changed working directory to {self.working_directory}"
                    )

                # Drop privileges
                self._drop_privileges()

            except Exception as e:
                logger.error(f"Error in preexec_fn: {e}")
                raise

        return preexec_fn

    def get_subprocess_kwargs(self) -> dict[str, Any]:
        """Get subprocess keyword arguments with security settings."""
        # Ensure we have the current environment with any PATH updates
        env = dict(os.environ)
        env.update(self.environment)

        kwargs = {
            "preexec_fn": self.get_preexec_fn(),
            "env": env,
        }

        # Set working directory if specified
        if self.working_directory:
            kwargs["cwd"] = self.working_directory

        return kwargs

    def secure_run(
        self, cmd: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a command with security settings applied.

        Args:
            cmd: Command to run as a list of strings
            **kwargs: Additional arguments to pass to subprocess.run

        Returns:
            CompletedProcess result
        """
        # Merge security settings with user kwargs
        secure_kwargs = self.get_subprocess_kwargs()
        secure_kwargs.update(kwargs)

        logger.info(f"Running secure subprocess: {' '.join(cmd)}")
        logger.debug(
            f"Security settings - User: {self.user}, Group: {self.group}, CWD: {self.working_directory}"
        )

        return subprocess.run(cmd, **secure_kwargs)

    def secure_popen(self, cmd: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        """
        Create a Popen object with security settings applied.

        Args:
            cmd: Command to run as a list of strings
            **kwargs: Additional arguments to pass to subprocess.Popen

        Returns:
            Popen object
        """
        # Merge security settings with user kwargs
        secure_kwargs = self.get_subprocess_kwargs()
        secure_kwargs.update(kwargs)

        logger.info(f"Creating secure Popen: {' '.join(cmd)}")
        logger.debug(
            f"Security settings - User: {self.user}, Group: {self.group}, CWD: {self.working_directory}"
        )

        return subprocess.Popen(cmd, **secure_kwargs)


def create_claude_user_if_not_exists() -> None:
    """Create a claude user for privilege dropping if it doesn't exist."""
    try:
        pwd.getpwnam("claude")
        logger.info("User 'claude' already exists")
    except KeyError:
        logger.info("Creating 'claude' user for privilege dropping")
        # Note: This would need to be run as root during container setup
        # In practice, this should be done in the Dockerfile
        subprocess.run(
            ["useradd", "--system", "--create-home", "--shell", "/bin/bash", "claude"],
            check=True,
        )


def get_default_claude_security(
    working_directory: str | Path | None = None,
) -> SubprocessSecurity:
    """
    Get default security settings for Claude subprocess execution.

    Args:
        working_directory: Optional working directory override

    Returns:
        SubprocessSecurity instance with default settings
    """
    # Default working directory for Claude operations
    default_cwd = working_directory or "/tmp/claude-workspace"

    # Ensure the working directory exists
    Path(default_cwd).mkdir(parents=True, exist_ok=True)

    # Create security instance
    # Only drop privileges if we're running as root
    if os.getuid() == 0:
        return SubprocessSecurity(
            user="claude",
            group="claude",
            working_directory=default_cwd,
            environment={
                # "HOME": str(default_cwd),
                "TMPDIR": str(default_cwd),
            },
        )
    else:
        # If not running as root, just set CWD
        return SubprocessSecurity(
            working_directory=default_cwd,
            environment={
                # "HOME": default_cwd,
                "TMPDIR": str(default_cwd),
            },
        )
