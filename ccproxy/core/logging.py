import inspect
import logging
import os
import re
import shutil
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Protocol, TextIO

import structlog
from rich.console import Console
from rich.traceback import Traceback
from structlog.contextvars import bind_contextvars
from structlog.stdlib import BoundLogger
from structlog.typing import ExcInfo, Processor

from ccproxy.core.id_utils import generate_short_id


# Custom protocol for BoundLogger with trace method
class TraceBoundLogger(Protocol):
    """Protocol defining BoundLogger with trace method."""

    def trace(self, msg: str, *args: Any, **kwargs: Any) -> Any:
        """Log at TRACE level."""
        ...

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> Any:
        """Log at DEBUG level."""
        ...

    def info(self, msg: str, *args: Any, **kwargs: Any) -> Any:
        """Log at INFO level."""
        ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> Any:
        """Log at WARNING level."""
        ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> Any:
        """Log at ERROR level."""
        ...

    def bind(self, **kwargs: Any) -> "TraceBoundLogger":
        """Bind additional context to logger."""
        ...

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> Any:
        """Log at specific level."""
        ...


# Import LogCategory locally to avoid circular import


# Add TRACE level below DEBUG
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

# Register TRACE level with structlog
structlog.stdlib.LEVEL_TO_NAME[TRACE_LEVEL] = "trace"  # type: ignore[attr-defined]
structlog.stdlib.NAME_TO_LEVEL["trace"] = TRACE_LEVEL  # type: ignore[attr-defined]


# Monkey-patch trace method to Logger class
def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log at TRACE level (below DEBUG)."""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)


logging.Logger.trace = trace  # type: ignore[attr-defined]


# Custom BoundLogger that includes trace method
class TraceBoundLoggerImpl(BoundLogger):
    """BoundLogger with trace method support."""

    def trace(self, msg: str, *args: Any, **kwargs: Any) -> Any:
        """Log at TRACE level."""
        return self.log(TRACE_LEVEL, msg, *args, **kwargs)


suppress_debug = [
    "ccproxy.scheduler",
]


def category_filter(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Filter logs by category based on environment configuration."""
    # Get filter settings from environment
    included_channels = os.getenv("CCPROXY_LOG_CHANNELS", "").strip()
    excluded_channels = os.getenv("CCPROXY_LOG_EXCLUDE_CHANNELS", "").strip()

    if not included_channels and not excluded_channels:
        return event_dict  # No filtering

    included = (
        [c.strip() for c in included_channels.split(",") if c.strip()]
        if included_channels
        else []
    )
    excluded = (
        [c.strip() for c in excluded_channels.split(",") if c.strip()]
        if excluded_channels
        else []
    )

    category = event_dict.get("category")

    # For foreign (stdlib) logs without category, check if logger name suggests a category
    if category is None:
        logger_name = event_dict.get("logger", "")
        # Map common logger names to categories
        if logger_name.startswith(("uvicorn", "fastapi", "starlette")):
            category = "general"  # Allow uvicorn/fastapi logs through as general
        elif logger_name.startswith("httpx"):
            category = "http"
        else:
            category = "general"  # Default fallback

        # Add the category to the event dict for consistent handling
        event_dict["category"] = category

    # Apply filters - be more permissive with foreign logs that got "general" as fallback
    # and ALWAYS allow errors and warnings through regardless of category filtering
    log_level = event_dict.get("level", "").lower()
    is_critical_message = log_level in ("error", "warning", "critical")

    if included and category not in included:
        # Always allow critical messages through regardless of category filtering
        if is_critical_message:
            return event_dict

        # If it's a foreign log with "general" fallback, and "general" is not in included channels,
        # still allow it through to prevent breaking stdlib logging
        logger_name = event_dict.get("logger", "")
        is_foreign_log = not logger_name.startswith(
            "ccproxy"
        ) and not logger_name.startswith("plugins")

        if not (is_foreign_log and category == "general"):
            raise structlog.DropEvent

    if excluded and category in excluded:
        # Always allow critical messages through even if their category is explicitly excluded
        if is_critical_message:
            return event_dict
        raise structlog.DropEvent

    return event_dict


def format_category_for_console(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Format category field for better visibility in console output."""
    if "category" in event_dict:
        category = event_dict.pop("category")  # Remove from key-value pairs
        # Prepend category to the event message
        event = event_dict.get("event", "")
        event_dict["event"] = f"[{category.upper()}] {event}"
    else:
        # Add default category if missing
        event = event_dict.get("event", "")
        event_dict["event"] = f"[GENERAL] {event}"
    return event_dict


class CategoryConsoleRenderer:
    """Custom console renderer that formats categories as a separate padded column."""

    def __init__(self, base_renderer: Any):
        self.base_renderer = base_renderer

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> str:
        # Extract category and plugin_name, remove from event dict to prevent duplicate display
        category = event_dict.pop("category", "general")
        plugin_name = event_dict.pop("plugin_name", None)

        # Get the rendered output from base renderer (without category/plugin_name in key-value pairs)
        rendered = self.base_renderer(logger, method_name, event_dict)

        # Color mapping for different categories
        category_colors = {
            "lifecycle": "\033[92m",  # bright green
            "plugin": "\033[94m",  # bright blue
            "http": "\033[95m",  # bright magenta
            "streaming": "\033[96m",  # bright cyan
            "auth": "\033[93m",  # bright yellow
            "transform": "\033[91m",  # bright red
            "cache": "\033[97m",  # bright white
            "middleware": "\033[35m",  # magenta
            "config": "\033[34m",  # blue
            "metrics": "\033[32m",  # green
            "access": "\033[33m",  # yellow
            "request": "\033[36m",  # cyan
            "general": "\033[37m",  # white
        }

        # Plugin name colors (distinct from categories)
        plugin_colors = {
            "claude_api": "\033[38;5;33m",  # blue
            "claude_sdk": "\033[38;5;39m",  # bright blue
            "codex": "\033[38;5;214m",  # orange
            "permissions": "\033[38;5;165m",  # purple
            "raw_http_logger": "\033[38;5;150m",  # light green
        }

        # Get colors
        category_color = category_colors.get(category.lower(), "\033[37m")
        plugin_color = (
            plugin_colors.get(plugin_name, "\033[38;5;242m") if plugin_name else None
        )

        # Build the display fields
        # Truncate long category names to fit the field width
        truncated_category = (
            category.lower()[:10] if len(category) > 10 else category.lower()
        )
        category_field = f"{category_color}\033[1m[{truncated_category:<10}]\033[0m"

        # Always show a plugin field - either plugin name or "core"
        if plugin_name:
            # Truncate long plugin names to fit the field width
            truncated_name = plugin_name[:12] if len(plugin_name) > 12 else plugin_name
            plugin_field = f"{plugin_color}\033[1m[{truncated_name:<12}]\033[0m "
        else:
            # Show "core" for non-plugin logs with a distinct color
            core_color = "\033[38;5;8m"  # dark gray
            plugin_field = f"{core_color}\033[1m[{'core':<12}]\033[0m "

        # Insert fields after the level field in the rendered output
        # Find the position right after the level field closes with "] "
        level_end_pattern = r"(\[[^\]]*\[[^\]]*m[^\]]*\[[^\]]*m\])\s+"
        match = re.search(level_end_pattern, rendered)

        if match:
            # Insert plugin_field and category_field after the level field
            insert_pos = match.end()
            rendered = (
                rendered[:insert_pos]
                + plugin_field
                + category_field
                + " "
                + rendered[insert_pos:]
            )
        else:
            # Fallback: prepend fields to the beginning
            rendered = plugin_field + category_field + " " + rendered

        return str(rendered)


def configure_structlog(log_level: int = logging.INFO) -> None:
    """Configure structlog with shared processors following canonical pattern."""
    # Shared processors for all structlog loggers
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,  # For request context in web apps
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        category_filter,  # Add category filtering
    ]

    # Add debug-specific processors
    if log_level < logging.INFO:
        # Dev mode (DEBUG): add callsite information
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    # Common processors for all log levels
    # First add timestamp with microseconds
    processors.append(
        structlog.processors.TimeStamper(
            fmt="%H:%M:%S.%f" if log_level < logging.INFO else "%Y-%m-%d %H:%M:%S.%f",
            key="timestamp_raw",
        )
    )

    # Then add processor to convert microseconds to milliseconds
    def format_timestamp_ms(
        logger: Any, log_method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Format timestamp with milliseconds instead of microseconds."""
        if "timestamp_raw" in event_dict:
            # Truncate microseconds to milliseconds (6 digits to 3)
            timestamp_raw = event_dict.pop("timestamp_raw")
            event_dict["timestamp"] = timestamp_raw[:-3]
        return event_dict

    processors.extend(
        [
            format_timestamp_ms,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,  # Handle exceptions properly
            # This MUST be the last processor - allows different renderers per handler
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    )

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=TraceBoundLoggerImpl,
        cache_logger_on_first_use=True,
    )


def rich_traceback(sio: TextIO, exc_info: ExcInfo) -> None:
    """Pretty-print *exc_info* to *sio* using the *Rich* package.

    Based on:
    https://github.com/hynek/structlog/blob/74cdff93af217519d4ebea05184f5e0db2972556/src/structlog/dev.py#L179-L192

    """
    term_width, _height = shutil.get_terminal_size((80, 123))
    sio.write("\n")
    # Rich docs: https://rich.readthedocs.io/en/stable/reference/traceback.html
    Console(file=sio, color_system="truecolor").print(
        Traceback.from_exception(
            *exc_info,
            # show_locals=True,  # Takes up too much vertical space
            extra_lines=1,  # Reduce amount of source code displayed
            width=term_width,  # Maximize width
            max_frames=5,  # Default is 10
            suppress=[
                "click",
                "typer",
                "uvicorn",
                "fastapi",
                "starlette",
            ],  # Suppress noise from these libraries
        ),
    )


def setup_logging(
    json_logs: bool = False,
    log_level_name: str = "DEBUG",
    log_file: str | None = None,
) -> TraceBoundLogger:
    """
    Setup logging for the entire application using canonical structlog pattern.
    Returns a structlog logger instance.
    """
    # Handle custom TRACE level explicitly
    log_level_upper = log_level_name.upper()
    if log_level_upper == "TRACE":
        log_level = TRACE_LEVEL
    else:
        log_level = getattr(logging, log_level_upper, logging.INFO)

    # Install rich traceback handler globally with frame limit
    # install_rich_traceback(
    #     show_locals=log_level <= logging.DEBUG,  # Only show locals in debug mode
    #     max_frames=max_traceback_frames,
    #     width=120,
    #     word_wrap=True,
    #     suppress=[
    #         "click",
    #         "typer",
    #         "uvicorn",
    #         "fastapi",
    #         "starlette",
    #     ],  # Suppress noise from these libraries
    # )

    # Get root logger and set level BEFORE configuring structlog
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 1. Configure structlog with shared processors
    configure_structlog(log_level=log_level)

    # 2. Setup root logger handlers
    root_logger.handlers = []  # Clear any existing handlers

    # 3. Create shared processors for foreign (stdlib) logs
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        category_filter,  # Apply category filtering to all logs
        structlog.dev.set_exc_info,
    ]

    # Add debug processors if needed
    if log_level < logging.INFO:
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(  # type: ignore[arg-type]
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    # Add appropriate timestamper for console vs file
    # Using custom lambda to truncate microseconds to milliseconds
    console_timestamper = (
        structlog.processors.TimeStamper(fmt="%H:%M:%S.%f", key="timestamp_raw")
        if log_level < logging.INFO
        else structlog.processors.TimeStamper(
            fmt="%Y-%m-%d %H:%M:%S.%f", key="timestamp_raw"
        )
    )

    # Processor to convert microseconds to milliseconds
    def format_timestamp_ms(
        logger: Any, log_method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Format timestamp with milliseconds instead of microseconds."""
        if "timestamp_raw" in event_dict:
            # Truncate microseconds to milliseconds (6 digits to 3)
            timestamp_raw = event_dict.pop("timestamp_raw")
            event_dict["timestamp"] = timestamp_raw[:-3]
        return event_dict

    file_timestamper = structlog.processors.TimeStamper(fmt="iso")

    # 4. Setup console handler with ConsoleRenderer
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    base_console_renderer = structlog.dev.ConsoleRenderer(
        exception_formatter=rich_traceback,  # Use rich for better formatting
        colors=True,
        pad_event=30,
    )

    console_renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else CategoryConsoleRenderer(base_console_renderer)
    )

    # Console gets human-readable timestamps for both structlog and stdlib logs
    # Note: format_category_for_console must come after category_filter
    console_processors = shared_processors + [
        console_timestamper,
        format_timestamp_ms,
        format_category_for_console,
    ]
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=console_processors,  # type: ignore[arg-type]
            processor=console_renderer,
        )
    )
    root_logger.addHandler(console_handler)

    # 5. Setup file handler with JSONRenderer (if log_file provided)
    if log_file:
        # Ensure parent directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
        file_handler.setLevel(log_level)

        # File gets ISO timestamps for both structlog and stdlib logs
        file_processors = shared_processors + [file_timestamper]
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=file_processors,
                processor=structlog.processors.JSONRenderer(),
            )
        )
        root_logger.addHandler(file_handler)

    # 6. Configure stdlib loggers to propagate to our handlers
    for logger_name in [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "ccproxy",
    ]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []  # Remove default handlers
        logger.propagate = True  # Use root logger's handlers

        # In DEBUG mode, let all logs through at DEBUG level
        # Otherwise, reduce uvicorn noise by setting to WARNING
        if log_level == logging.DEBUG:
            logger.setLevel(logging.DEBUG)
        elif logger_name.startswith("uvicorn"):
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(log_level)

    # Configure httpx logger separately - INFO when app is DEBUG, WARNING otherwise
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.handlers = []
    httpx_logger.propagate = True
    # httpx_logger.setLevel(logging.INFO if log_level < logging.INFO else logging.WARNING)

    # Set noisy HTTP-related loggers to WARNING
    noisy_log_level = logging.WARNING if log_level <= logging.WARNING else log_level
    for noisy_logger_name in [
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "aiohttp",
        "httpcore",
        "httpcore.http11",
        "fastapi_mcp",
        "sse_starlette",
        "mcp",
        "hpack",
    ]:
        noisy_logger = logging.getLogger(noisy_logger_name)
        noisy_logger.handlers = []
        noisy_logger.propagate = True
        noisy_logger.setLevel(noisy_log_level)

    for logger_name in suppress_debug:
        logging.getLogger(logger_name).setLevel(
            logging.INFO if log_level <= logging.DEBUG else log_level
        )

    return structlog.get_logger()  # type: ignore[no-any-return]


# Create a convenience function for getting loggers
def get_logger(name: str | None = None) -> TraceBoundLogger:
    """Get a structlog logger instance with request context automatically bound.

    This function checks for an active RequestContext and automatically binds
    the request_id to the logger if available, ensuring all logs are correlated
    with the current request.

    Args:
        name: Logger name (typically __name__)

    Returns:
        TraceBoundLogger with request_id bound if available
    """
    logger = structlog.get_logger(name)

    # Try to get request context and bind request_id if available
    try:
        from ccproxy.core.request_context import RequestContext

        context = RequestContext.get_current()
        if context and context.request_id:
            logger = logger.bind(request_id=context.request_id)
    except Exception:
        # If anything fails, just return the regular logger
        # This ensures backward compatibility
        pass

    return logger  # type: ignore[no-any-return]


def get_plugin_logger(name: str | None = None) -> TraceBoundLogger:
    """Get a plugin-aware logger with plugin_name automatically bound.

    This function auto-detects the plugin name from the caller's module path
    and binds it to the logger. Preserves all existing functionality including
    request_id binding and trace method.

    Args:
        name: Logger name (auto-detected from caller if None)

    Returns:
        TraceBoundLogger with plugin_name and request_id bound if available
    """
    if name is None:
        # Auto-detect caller's module name
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")
        else:
            name = "unknown"

    # Use existing get_logger (preserves request_id binding & trace method)
    logger = get_logger(name)

    # Extract and bind plugin name for plugin modules
    if name and name.startswith("plugins."):
        parts = name.split(".", 2)
        if len(parts) > 1:
            plugin_name = parts[1]  # e.g., "claude_api", "codex"
            logger = logger.bind(plugin_name=plugin_name)

    return logger


def info_allowed(app: Any | None = None) -> bool:
    """Whether non-summary INFO logs are allowed based on app.state.

    If `app.state.info_summaries_only` is True, return False to suppress
    granular INFO lines. Defaults to True when state not available.
    """
    try:
        if app is None:
            return True
        state = getattr(app, "state", None)
        if not state:
            return True
        return not bool(getattr(state, "info_summaries_only", False))
    except Exception:
        return True


def reduce_startup(app: Any | None = None) -> bool:
    """Whether startup noise reduction is enabled.

    Returns True when `app.state.reduce_startup_info` is True; False otherwise.
    """
    try:
        if app is None:
            return False
        state = getattr(app, "state", None)
        if not state:
            return False
        return bool(getattr(state, "reduce_startup_info", False))
    except Exception:
        return False


def _parse_arg_value(argv: list[str], flag: str) -> str | None:
    """Parse a simple CLI flag value from argv.

    Supports "--flag value" and "--flag=value" forms. Returns None if not present.
    """
    if not argv:
        return None
    try:
        for i, token in enumerate(argv):
            if token == flag and i + 1 < len(argv):
                return argv[i + 1]
            if token.startswith(flag + "="):
                return token.split("=", 1)[1]
    except Exception:
        # Be forgiving in bootstrap parsing
        return None
    return None


def bootstrap_cli_logging(argv: list[str] | None = None) -> None:
    """Best-effort early logging setup from env and CLI args.

    - Parses `--log-level` and `--log-file` from argv (if provided).
    - Honors env overrides `LOGGING__LEVEL`, `LOGGING__FILE`.
    - Enables JSON logs if explicitly requested via `LOGGING__FORMAT=json` or `CCPROXY_JSON_LOGS=true`.
    - No-op if structlog is already configured, letting later setup prevail.

    This is intentionally lightweight and is followed by a full `setup_logging`
    call after settings are loaded (e.g., in the serve command), so runtime
    changes from config are still applied.
    """
    try:
        if structlog.is_configured():
            return

        if argv is None:
            argv = sys.argv[1:]

        # Env-based defaults
        env_level = os.getenv("LOGGING__LEVEL")
        env_file = os.getenv("LOGGING__FILE")
        env_format = os.getenv("LOGGING__FORMAT")

        # CLI overrides
        arg_level = _parse_arg_value(argv, "--log-level")
        arg_file = _parse_arg_value(argv, "--log-file")

        # Decide whether to bootstrap at all: only if any override present
        any_override = any([env_level, env_file, env_format, arg_level, arg_file])
        if not any_override:
            return

        # Resolve effective values (CLI > env)
        level = (arg_level or env_level or "INFO").upper()
        log_file = arg_file or env_file

        # JSON if explicitly requested via env
        json_logs = False
        if env_format:
            json_logs = env_format.lower() == "json"

        # Apply early setup. Safe to run again later with final settings.
        setup_logging(json_logs=json_logs, log_level_name=level, log_file=log_file)
    except Exception:
        # Never break CLI due to bootstrap; final setup will run later.
        return


def set_command_context(cmd_id: str | None = None) -> str:
    """Bind a command-wide correlation ID to structlog context.

    Uses structlog.contextvars so all logs (including from plugins) will carry
    `cmd_id` once logging is configured with `merge_contextvars`.

    Args:
        cmd_id: Optional explicit command ID. If None, a UUID4 is generated.

    Returns:
        The command ID that was bound.
    """
    try:
        if not cmd_id:
            cmd_id = generate_short_id()
        # Bind only cmd_id to avoid colliding with per-request request_id fields
        bind_contextvars(cmd_id=cmd_id)
        return cmd_id
    except Exception:
        # Be defensive: never break CLI startup due to context binding
        return cmd_id or ""


# --- Lightweight test-time bootstrap ---------------------------------------
# Ensure structlog logs are capturable by pytest's caplog without requiring
# full application setup. When running under pytest (PYTEST_CURRENT_TEST),
# configure structlog to emit through stdlib logging with a simple renderer
# and set the root level to INFO so info logs are not filtered.
def _bootstrap_test_logging_if_needed() -> None:
    try:
        if os.getenv("PYTEST_CURRENT_TEST") and not structlog.is_configured():
            # Ensure INFO-level logs are visible to caplog
            logging.getLogger().setLevel(logging.INFO)

            # Configure structlog to hand off to stdlib with extra fields so that
            # pytest's caplog sees attributes like `record.category`.
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.add_logger_name,
                    category_filter,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.format_exc_info,
                    # Pass fields as LogRecord.extra for caplog
                    structlog.stdlib.render_to_log_kwargs,
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=TraceBoundLoggerImpl,
                cache_logger_on_first_use=True,
            )
    except Exception:
        # Never fail test imports due to logging bootstrap
        pass


# Invoke test bootstrap on import if appropriate
_bootstrap_test_logging_if_needed()
