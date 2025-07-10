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
