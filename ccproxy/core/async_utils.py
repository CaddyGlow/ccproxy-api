"""Async utilities for the Claude Code Proxy API."""

import asyncio
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


def format_version(version: str) -> str:
    """Format version string for display.

    Args:
        version: Version string to format

    Returns:
        Formatted version string
    """
    import re

    # Clean up version string
    version = version.strip()

    # Remove 'v' prefix if present
    if version.startswith("v"):
        version = version[1:]

    # Ensure it follows semantic versioning pattern
    if not re.match(r"^\d+\.\d+\.\d+", version):
        return f"1.0.0-{version}"

    return version


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


def generate_schema_files(output_dir: Path) -> list[Path]:
    """Generate JSON Schema files for TOML configuration validation.

    Args:
        output_dir: Directory to write schema files to

    Returns:
        List of generated schema file paths

    Raises:
        ImportError: If required dependencies are not available
        OSError: If unable to write files
    """
    try:
        import json
        from typing import Any

        from ccproxy.config.docker_settings import DockerSettings

        # Import the settings and docker settings models
        from ccproxy.config.settings import Settings
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}") from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []

    # Generate schema for main Settings model
    settings_schema = Settings.model_json_schema()
    settings_schema_path = output_dir / "ccproxy-schema.json"

    with settings_schema_path.open("w", encoding="utf-8") as f:
        json.dump(settings_schema, f, indent=2, ensure_ascii=False)
    generated_files.append(settings_schema_path)

    # Generate schema for DockerSettings model
    docker_schema = DockerSettings.model_json_schema()
    docker_schema_path = output_dir / "docker-settings-schema.json"

    with docker_schema_path.open("w", encoding="utf-8") as f:
        json.dump(docker_schema, f, indent=2, ensure_ascii=False)
    generated_files.append(docker_schema_path)

    # Generate a combined schema file that can be used for complete config validation
    combined_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "ccproxy-config-schema",
        "title": "Claude Code Proxy Configuration Schema",
        "description": "JSON Schema for validating Claude Code Proxy configuration files",
        "type": "object",
        "properties": settings_schema.get("properties", {}),
        "additionalProperties": settings_schema.get("additionalProperties", False),
        "definitions": settings_schema.get("$defs", {}),
    }

    combined_schema_path = output_dir / ".ccproxy-schema.json"
    with combined_schema_path.open("w", encoding="utf-8") as f:
        json.dump(combined_schema, f, indent=2, ensure_ascii=False)
    generated_files.append(combined_schema_path)

    return generated_files


def generate_taplo_config(output_dir: Path) -> Path:
    """Generate taplo configuration for TOML editor support.

    Args:
        output_dir: Directory to write taplo config to

    Returns:
        Path to generated taplo.toml file

    Raises:
        OSError: If unable to write file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    taplo_config_path = output_dir / "taplo.toml"

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
schema = ".ccproxy-schema.json"

[[rule]]
name = "ccproxy-docker"
include = [
    "**/docker-settings.toml",
    "**/docker.toml"
]
schema = "docker-settings-schema.json"

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

    with taplo_config_path.open("w", encoding="utf-8") as f:
        f.write(taplo_config)

    return taplo_config_path


def validate_config_with_schema(config_path: Path) -> bool:
    """Validate a TOML config file against the schema.

    Args:
        config_path: Path to configuration file to validate

    Returns:
        True if validation passes, False otherwise

    Raises:
        ImportError: If check-jsonschema is not available
        FileNotFoundError: If config file doesn't exist
        ValueError: If unable to parse or validate file
    """
    try:
        import json
        import subprocess
        import tempfile
        from typing import Any

        # Import tomllib for Python 3.11+ or fallback to tomli
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef,import-not-found]

        from ccproxy.config.settings import Settings
    except ImportError as e:
        raise ImportError(
            f"Required dependencies not available: {e}. "
            "Install with: pip install check-jsonschema"
        ) from e

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # For TOML files, we need to convert to JSON first since check-jsonschema
    # doesn't directly support TOML validation
    if config_path.suffix.lower() == ".toml":
        try:
            # Read and parse TOML
            with config_path.open("rb") as f:
                toml_data = tomllib.load(f)
        except Exception as e:
            raise ValueError(f"Unable to parse TOML file {config_path}: {e}") from e

        # Generate schema in a temporary file
        schema = Settings.model_json_schema()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as schema_file:
            json.dump(schema, schema_file, indent=2)
            schema_path = schema_file.name

        # Convert TOML data to JSON for validation
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as json_file:
            json.dump(toml_data, json_file, indent=2)
            json_path = json_file.name

        try:
            # Use check-jsonschema to validate
            result = subprocess.run(
                ["check-jsonschema", "--schemafile", schema_path, json_path],
                capture_output=True,
                text=True,
                check=False,
            )

            # Clean up temporary files
            Path(schema_path).unlink(missing_ok=True)
            Path(json_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return True
            else:
                # Log validation errors
                if result.stderr:
                    raise ValueError(f"Schema validation failed: {result.stderr}")
                if result.stdout:
                    raise ValueError(f"Schema validation failed: {result.stdout}")
                return False

        except FileNotFoundError as e:
            raise ImportError(
                "check-jsonschema command not found. "
                "Install with: pip install check-jsonschema"
            ) from e
        except Exception as e:
            # Clean up temporary files in case of error
            Path(schema_path).unlink(missing_ok=True)
            Path(json_path).unlink(missing_ok=True)
            raise ValueError(f"Validation error: {e}") from e

    else:
        # For JSON/YAML files, use check-jsonschema directly
        schema = Settings.model_json_schema()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as schema_file:
            json.dump(schema, schema_file, indent=2)
            schema_path = schema_file.name

        try:
            result = subprocess.run(
                ["check-jsonschema", "--schemafile", schema_path, str(config_path)],
                capture_output=True,
                text=True,
                check=False,
            )

            Path(schema_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return True
            else:
                if result.stderr:
                    raise ValueError(f"Schema validation failed: {result.stderr}")
                if result.stdout:
                    raise ValueError(f"Schema validation failed: {result.stdout}")
                return False

        except FileNotFoundError as e:
            raise ImportError(
                "check-jsonschema command not found. "
                "Install with: pip install check-jsonschema"
            ) from e
        except Exception as e:
            Path(schema_path).unlink(missing_ok=True)
            raise ValueError(f"Validation error: {e}") from e
