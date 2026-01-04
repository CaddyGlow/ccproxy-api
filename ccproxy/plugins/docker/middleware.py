"""Docker output middleware for processing and logging container output."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger

from .stream_process import OutputMiddleware, create_chained_middleware


logger = get_plugin_logger(__name__)


class LoggerOutputMiddleware(OutputMiddleware[str]):
    """Simple middleware that prints output with optional prefixes.

    This middleware prints each line to the console with configurable
    prefixes for stdout and stderr streams.
    """

    def __init__(self, logger: Any, stdout_prefix: str = "", stderr_prefix: str = ""):
        """Initialize middleware with custom prefixes.

        Args:
            stdout_prefix: Prefix for stdout lines (default: "")
            stderr_prefix: Prefix for stderr lines (default: "")
        """
        self.logger = logger
        self.stderr_prefix = stderr_prefix
        self.stdout_prefix = stdout_prefix

    async def process(self, line: str, stream_type: str) -> str:
        """Process and print a line with the appropriate prefix.

        Args:
            line: Output line to process
            stream_type: Either "stdout" or "stderr"

        Returns:
            The original line (unmodified)
        """
        if stream_type == "stdout":
            self.logger.info(
                "docker_stdout", prefix=self.stdout_prefix, line=line, stream="stdout"
            )
        else:
            self.logger.info(
                "docker_stderr", prefix=self.stderr_prefix, line=line, stream="stderr"
            )
        return line


def create_logger_middleware(
    logger_instance: Any | None = None,
    stdout_prefix: str = "",
    stderr_prefix: str = "",
) -> LoggerOutputMiddleware:
    """Factory function to create a LoggerOutputMiddleware instance.

    Args:
        logger_instance: Logger instance to use (defaults to module logger)
        stdout_prefix: Prefix for stdout lines
        stderr_prefix: Prefix for stderr lines

    Returns:
        Configured LoggerOutputMiddleware instance
    """
    if logger_instance is None:
        logger_instance = logger
    return LoggerOutputMiddleware(logger_instance, stdout_prefix, stderr_prefix)


def create_chained_docker_middleware(
    middleware_chain: list[OutputMiddleware[Any]],
    include_logger: bool = True,
    logger_instance: Any | None = None,
    stdout_prefix: str = "",
    stderr_prefix: str = "",
) -> OutputMiddleware[Any]:
    """Factory function to create chained middleware for Docker operations.

    Args:
        middleware_chain: List of middleware components to chain together
        include_logger: Whether to automatically add logger middleware at the end
        logger_instance: Logger instance to use (defaults to module logger)
        stdout_prefix: Prefix for stdout lines in logger middleware
        stderr_prefix: Prefix for stderr lines in logger middleware

    Returns:
        Chained middleware instance

    """
    final_chain = list(middleware_chain)

    if include_logger:
        logger_middleware = create_logger_middleware(
            logger_instance, stdout_prefix, stderr_prefix
        )
        final_chain.append(logger_middleware)

    if len(final_chain) == 1:
        return final_chain[0]

    return create_chained_middleware(final_chain)
