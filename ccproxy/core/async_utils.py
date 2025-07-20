"""Async utilities for the Claude Code Proxy API."""

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


# Extract the typing fix from utils/helper.py
@contextmanager
def patched_typing() -> Iterator[None]:
    """Fix for typing.TypedDict not supported in older Python versions.

    This patches typing.TypedDict to use typing_extensions.TypedDict.
    """
    import typing

    import typing_extensions

    original = typing.TypedDict
    typing.TypedDict = typing_extensions.TypedDict
    try:
        yield
    finally:
        typing.TypedDict = original


def get_package_dir() -> Path:
    """Get the package directory path.

    Returns:
        Path to the package directory
    """
    try:
        import importlib.util

        # Get the path to the ccproxy package and resolve it
        spec = importlib.util.find_spec(get_root_package_name())
        if spec and spec.origin:
            package_dir = Path(spec.origin).parent.parent.resolve()
        else:
            package_dir = Path(__file__).parent.parent.parent.resolve()
    except Exception:
        package_dir = Path(__file__).parent.parent.parent.resolve()

    return package_dir


def get_root_package_name() -> str:
    """Get the root package name.

    Returns:
        The root package name
    """
    if __package__:
        return __package__.split(".")[0]
    return __name__.split(".")[0]


async def run_in_executor(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous function in an executor.

    Args:
        func: The synchronous function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call
    """
    loop = asyncio.get_event_loop()

    # Create a partial function if we have kwargs
    if kwargs:
        from functools import partial

        func = partial(func, **kwargs)

    return await loop.run_in_executor(None, func, *args)


async def safe_await(awaitable: Awaitable[T], timeout: float | None = None) -> T | None:
    """Safely await an awaitable with optional timeout.

    Args:
        awaitable: The awaitable to wait for
        timeout: Optional timeout in seconds

    Returns:
        The result of the awaitable or None if timeout/error
    """
    try:
        if timeout is not None:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        return await awaitable
    except TimeoutError:
        return None
    except Exception:
        return None


async def gather_with_concurrency(
    limit: int, *awaitables: Awaitable[T], return_exceptions: bool = False
) -> list[T | BaseException] | list[T]:
    """Gather awaitables with concurrency limit.

    Args:
        limit: Maximum number of concurrent operations
        *awaitables: Awaitables to execute
        return_exceptions: Whether to return exceptions as results

    Returns:
        List of results from the awaitables
    """
    semaphore = asyncio.Semaphore(limit)

    async def _limited_awaitable(awaitable: Awaitable[T]) -> T:
        async with semaphore:
            return await awaitable

    limited_awaitables = [_limited_awaitable(aw) for aw in awaitables]
    if return_exceptions:
        return await asyncio.gather(*limited_awaitables, return_exceptions=True)
    else:
        return await asyncio.gather(*limited_awaitables)


@asynccontextmanager
async def async_timer() -> AsyncIterator[Callable[[], float]]:
    """Context manager for timing async operations.

    Yields:
        Function that returns elapsed time in seconds
    """
    import time

    start_time = time.perf_counter()

    def get_elapsed() -> float:
        return time.perf_counter() - start_time

    yield get_elapsed


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exception types to catch and retry on
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the successful function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                raise

    # This should never be reached, but just in case
    raise last_exception if last_exception else Exception("Retry failed")


async def wait_for_condition(
    condition: Callable[[], bool | Awaitable[bool]],
    timeout: float = 30.0,
    interval: float = 0.1,
) -> bool:
    """Wait for a condition to become true.

    Args:
        condition: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds

    Returns:
        True if condition was met, False if timeout occurred
    """
    start_time = asyncio.get_event_loop().time()

    while True:
        try:
            result = condition()
            if asyncio.iscoroutine(result):
                result = await result
            if result:
                return True
        except Exception:
            pass

        if asyncio.get_event_loop().time() - start_time > timeout:
            return False

        await asyncio.sleep(interval)


_cache: dict[str, tuple[float, Any]] = {}


async def async_cache_result(
    func: Callable[..., Awaitable[T]],
    cache_key: str,
    cache_duration: float = 300.0,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Cache the result of an async function call.

    Args:
        func: The async function to cache
        cache_key: Unique key for caching
        cache_duration: Cache duration in seconds
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The cached or computed result
    """
    import time

    current_time = time.time()

    # Check if we have a valid cached result
    if cache_key in _cache:
        cached_time, cached_result = _cache[cache_key]
        if current_time - cached_time < cache_duration:
            return cached_result  # type: ignore[no-any-return]

    # Compute and cache the result
    result = await func(*args, **kwargs)
    _cache[cache_key] = (current_time, result)

    return result


def parse_version(version_string: str) -> tuple[int, int, int, str]:
    """
    Parse version string into components.

    Handles various formats:
    - 1.2.3
    - 1.2.3-dev
    - 1.2.3.dev59+g1624e1e.d19800101
    - 0.1.dev59+g1624e1e.d19800101
    """
    # Clean up setuptools-scm dev versions
    clean_version = re.sub(r"\.dev\d+\+.*", "", version_string)

    # Handle dev versions without patch number
    if ".dev" in version_string:
        base_version = version_string.split(".dev")[0]
        parts = base_version.split(".")
        if len(parts) == 2:
            # 0.1.dev59 -> 0.1.0-dev
            major, minor = int(parts[0]), int(parts[1])
            patch = 0
            suffix = "dev"
        else:
            # 1.2.3.dev59 -> 1.2.3-dev
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            suffix = "dev"
    else:
        # Regular version
        parts = clean_version.split(".")
        if len(parts) < 3:
            parts.extend(["0"] * (3 - len(parts)))

        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        suffix = ""

    return major, minor, patch, suffix


def format_version(version: str, level: str) -> str:
    major, minor, patch, suffix = parse_version(version)

    """Format version according to specified level."""
    base_version = f"{major}.{minor}.{patch}"

    if level == "major":
        return str(major)
    elif level == "minor":
        return f"{major}.{minor}"
    elif level == "patch" or level == "full":
        if suffix:
            return f"{base_version}-{suffix}"
        return base_version
    elif level == "docker":
        # Docker-compatible version (no + characters)
        if suffix:
            return f"{base_version}-{suffix}"
        return base_version
    elif level == "npm":
        # NPM-compatible version
        if suffix:
            return f"{base_version}-{suffix}.0"
        return base_version
    elif level == "python":
        # Python-compatible version
        if suffix:
            return f"{base_version}.{suffix}0"
        return base_version
    else:
        raise ValueError(f"Unknown version level: {level}")


def get_claude_docker_home_dir() -> str:
    """Get the Claude Docker home directory path.

    Returns:
        Path to Claude Docker home directory
    """
    import os
    from pathlib import Path

    # Use XDG_DATA_HOME if available, otherwise default to ~/.local/share
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        base_dir = Path(xdg_data_home)
    else:
        base_dir = Path.home() / ".local" / "share"

    claude_dir = base_dir / "claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    return str(claude_dir)


def generate_schema_files(output_dir: Path | None = None) -> list[Path]:
    """Generate JSON Schema files for TOML configuration validation.

    Args:
        output_dir: Directory to write schema files to. If None, uses current directory.

    Returns:
        List of generated schema file paths

    Raises:
        ImportError: If required dependencies are not available
        OSError: If unable to write files
    """
    if output_dir is None:
        output_dir = Path.cwd()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []

    # Generate schema for main Settings model
    schema = generate_json_schema()
    main_schema_path = output_dir / "ccproxy-schema.json"
    save_schema_file(schema, main_schema_path)
    generated_files.append(main_schema_path)

    # Generate a combined schema file that can be used for complete config validation
    combined_schema_path = output_dir / ".ccproxy-schema.json"
    save_schema_file(schema, combined_schema_path)
    generated_files.append(combined_schema_path)

    return generated_files


def generate_taplo_config(output_dir: Path | None = None) -> Path:
    """Generate taplo configuration for TOML editor support.

    Args:
        output_dir: Directory to write taplo config to. If None, uses current directory.

    Returns:
        Path to generated .taplo.toml file

    Raises:
        OSError: If unable to write file
    """
    if output_dir is None:
        output_dir = Path.cwd()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    taplo_config_path = output_dir / ".taplo.toml"

    # Generate taplo configuration that references our schema files
    taplo_config = """# Taplo configuration for Claude Code Proxy TOML files
# This configuration enables schema validation and autocomplete in editors

[[rule]]
name = "ccproxy-config"
include = [
    ".ccproxy.toml",
    "ccproxy.toml",
    "config.toml",
    "**/ccproxy*.toml",
    "**/config*.toml"
]
schema = "ccproxy-schema.json"

[formatting]
# Standard TOML formatting options
indent_string = "  "
trailing_newline = true
crlf = false

[schema]
# Enable schema validation
enabled = true
# Show completions from schema
completion = true
"""

    taplo_config_path.write_text(taplo_config, encoding="utf-8")

    return taplo_config_path


def validate_config_with_schema(
    config_path: Path, schema_path: Path | None = None
) -> bool:
    """Validate a config file against the schema.

    Args:
        config_path: Path to configuration file to validate
        schema_path: Optional path to schema file. If None, generates schema from Settings

    Returns:
        True if validation passes, False otherwise

    Raises:
        ImportError: If check-jsonschema is not available
        FileNotFoundError: If config file doesn't exist
        tomllib.TOMLDecodeError: If TOML file has invalid syntax
        ValueError: For other validation errors
    """
    import json
    import subprocess
    import tempfile
    from typing import Any

    # Import tomllib for Python 3.11+ or fallback to tomli
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]

    from ccproxy.config.settings import Settings

    config_path = Path()

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Determine the file type
    suffix = config_path.suffix.lower()

    if suffix == ".toml":
        # Read and parse TOML - let TOML parse errors bubble up
        with config_path.open("rb") as f:
            toml_data = tomllib.load(f)

        # Get or generate schema
        if schema_path:
            with schema_path.open("r", encoding="utf-8") as f:
                schema = json.load(f)
        else:
            schema = generate_json_schema()

        # Create temporary files for validation
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as schema_file:
            json.dump(schema, schema_file, indent=2)
            temp_schema_path = schema_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as json_file:
            json.dump(toml_data, json_file, indent=2)
            temp_json_path = json_file.name

        try:
            # Use check-jsonschema to validate
            result = subprocess.run(
                ["check-jsonschema", "--schemafile", temp_schema_path, temp_json_path],
                capture_output=True,
                text=True,
                check=False,
            )

            # Clean up temporary files
            Path(temp_schema_path).unlink(missing_ok=True)
            Path(temp_json_path).unlink(missing_ok=True)

            return result.returncode == 0

        except FileNotFoundError as e:
            # Clean up temporary files
            Path(temp_schema_path).unlink(missing_ok=True)
            Path(temp_json_path).unlink(missing_ok=True)
            raise ImportError(
                "check-jsonschema command not found. "
                "Install with: pip install check-jsonschema"
            ) from e
        except Exception as e:
            # Clean up temporary files in case of error
            Path(temp_schema_path).unlink(missing_ok=True)
            Path(temp_json_path).unlink(missing_ok=True)
            raise ValueError(f"Validation error: {e}") from e

    elif suffix == ".json":
        # Parse JSON to validate it's well-formed - let JSON parse errors bubble up
        with config_path.open("r", encoding="utf-8") as f:
            json.load(f)

        # Get or generate schema
        if schema_path:
            temp_schema_path = str(schema_path)
            cleanup_schema = False
        else:
            schema = generate_json_schema()
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as schema_file:
                json.dump(schema, schema_file, indent=2)
                temp_schema_path = schema_file.name
                cleanup_schema = True

        try:
            result = subprocess.run(
                [
                    "check-jsonschema",
                    "--schemafile",
                    temp_schema_path,
                    str(config_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if cleanup_schema:
                Path(temp_schema_path).unlink(missing_ok=True)

            return result.returncode == 0

        except FileNotFoundError as e:
            if cleanup_schema:
                Path(temp_schema_path).unlink(missing_ok=True)
            raise ImportError(
                "check-jsonschema command not found. "
                "Install with: pip install check-jsonschema"
            ) from e
        except Exception as e:
            if cleanup_schema:
                Path(temp_schema_path).unlink(missing_ok=True)
            raise ValueError(f"Validation error: {e}") from e

    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. Only TOML (.toml) files are supported."
        )


# TODO: Remove this function or update this function
def generate_json_schema() -> dict[str, Any]:
    """Generate JSON Schema from Settings model.

    Returns:
        JSON Schema dictionary

    Raises:
        ImportError: If required dependencies are not available
    """
    try:
        from ccproxy.config.settings import Settings
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}") from e

    schema = Settings.model_json_schema()

    # Add schema metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Claude Code Proxy API Configuration"

    # Add examples for common properties
    properties = schema.get("properties", {})
    if "host" in properties:
        properties["host"]["examples"] = ["127.0.0.1", "0.0.0.0", "localhost"]
    if "port" in properties:
        properties["port"]["examples"] = [8000, 8080, 3000]
    if "log_level" in properties:
        properties["log_level"]["examples"] = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if "cors_origins" in properties:
        properties["cors_origins"]["examples"] = [
            ["*"],
            ["https://example.com", "https://app.example.com"],
            ["http://localhost:3000"],
        ]

    return schema


def save_schema_file(schema: dict[str, Any], output_path: Path) -> None:
    """Save JSON Schema to a file.

    Args:
        schema: JSON Schema dictionary to save
        output_path: Path to write schema file to

    Raises:
        OSError: If unable to write file
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def validate_toml_with_schema(
    config_path: Path, schema_path: Path | None = None
) -> bool:
    """Validate a TOML config file against JSON Schema.

    Args:
        config_path: Path to TOML configuration file
        schema_path: Optional path to schema file. If None, generates schema from Settings

    Returns:
        True if validation passes, False otherwise

    Raises:
        ImportError: If check-jsonschema is not available
        FileNotFoundError: If config file doesn't exist
        ValueError: If unable to parse or validate file
    """
    # This is a thin wrapper around validate_config_with_schema for TOML files
    config_path = Path(config_path)

    if config_path.suffix.lower() != ".toml":
        raise ValueError(f"Expected TOML file, got: {config_path.suffix}")

    return validate_config_with_schema(config_path)
