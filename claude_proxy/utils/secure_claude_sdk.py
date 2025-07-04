"""Secure wrapper for Claude Code SDK with privilege dropping."""

import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Optional

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    SystemMessage,
    UserMessage,
)
from claude_code_sdk import (
    query as original_query,
)

from .subprocess_security import get_default_claude_security


logger = logging.getLogger(__name__)


class SecureClaudeSDK:
    """Secure wrapper for Claude Code SDK with privilege dropping and CWD control."""

    def __init__(
        self,
        user: str | None = None,
        group: str | None = None,
        working_directory: str | None = None,
        environment: dict[str, str] | None = None,
    ):
        """
        Initialize secure Claude SDK wrapper.

        Args:
            user: Username to drop privileges to
            group: Group name to drop privileges to
            working_directory: Working directory for Claude operations
            environment: Environment variables to set
        """
        self.security = get_default_claude_security(working_directory)
        if user:
            self.security.user = user
        if group:
            self.security.group = group
        if environment:
            self.security.environment.update(environment)

    async def secure_query(
        self,
        prompt: str,
        options: ClaudeCodeOptions,
    ) -> AsyncIterator[UserMessage | AssistantMessage | SystemMessage | ResultMessage]:
        """
        Secure version of claude_code_sdk.query with privilege dropping.

        Args:
            prompt: The prompt to send to Claude
            options: Claude Code SDK options

        Yields:
            Messages from Claude Code SDK
        """
        # Monkey patch the subprocess execution in the Claude SDK
        # This is a bit hacky but necessary since we can't modify the SDK directly
        original_popen = None
        original_run = None

        try:
            # Store original subprocess functions
            import subprocess

            original_popen = subprocess.Popen
            original_run = subprocess.run

            # Create secure wrappers
            def secure_popen(*args, **kwargs):
                # Apply security settings
                security_kwargs = self.security.get_subprocess_kwargs()
                kwargs.update(security_kwargs)
                return original_popen(*args, **kwargs)

            def secure_run(*args, **kwargs):
                # Apply security settings
                security_kwargs = self.security.get_subprocess_kwargs()
                kwargs.update(security_kwargs)
                return original_run(*args, **kwargs)

            # Monkey patch subprocess functions
            subprocess.Popen = secure_popen
            subprocess.run = secure_run

            logger.info("Starting secure Claude query with privilege dropping")
            logger.debug(
                f"Security settings - User: {self.security.user}, Group: {self.security.group}, CWD: {self.security.working_directory}"
            )

            # Call the original query function with our security patches
            async for message in original_query(prompt=prompt, options=options):
                yield message

        except Exception as e:
            logger.error(f"Error in secure Claude query: {e}")
            raise
        finally:
            # Restore original subprocess functions
            if original_popen is not None:
                import subprocess

                subprocess.Popen = original_popen
            if original_run is not None:
                import subprocess

                subprocess.run = original_run

            logger.debug("Restored original subprocess functions")


# Global instance for easy access
_secure_claude_sdk: SecureClaudeSDK | None = None


def get_secure_claude_sdk() -> SecureClaudeSDK:
    """Get the global secure Claude SDK instance."""
    global _secure_claude_sdk
    if _secure_claude_sdk is None:
        _secure_claude_sdk = SecureClaudeSDK()
    return _secure_claude_sdk


def configure_secure_claude_sdk(
    user: str | None = None,
    group: str | None = None,
    working_directory: str | None = None,
    environment: dict[str, str] | None = None,
) -> None:
    """
    Configure the global secure Claude SDK instance.

    Args:
        user: Username to drop privileges to
        group: Group name to drop privileges to
        working_directory: Working directory for Claude operations
        environment: Environment variables to set
    """
    global _secure_claude_sdk
    _secure_claude_sdk = SecureClaudeSDK(
        user=user,
        group=group,
        working_directory=working_directory,
        environment=environment,
    )


async def secure_query(
    prompt: str,
    options: ClaudeCodeOptions,
) -> AsyncIterator[UserMessage | AssistantMessage | SystemMessage | ResultMessage]:
    """
    Secure version of claude_code_sdk.query with privilege dropping.

    Args:
        prompt: The prompt to send to Claude
        options: Claude Code SDK options

    Yields:
        Messages from Claude Code SDK
    """
    sdk = get_secure_claude_sdk()
    async for message in sdk.secure_query(prompt, options):
        yield message
