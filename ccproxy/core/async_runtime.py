"""Abstraction layer for async primitives used across the project.

The helpers in this module now lean on ``anyio`` for generic functionality while
preserving compatibility with existing ``asyncio`` constructs that remain in the
codebase (for example ``Task`` and the subprocess helpers).  The intent is that
callers continue importing utilities from this module only, allowing us to
evolve the underlying runtime without broad churn.
"""

from __future__ import annotations

import asyncio
import functools
import time
from asyncio import subprocess as asyncio_subprocess
from builtins import TimeoutError as BuiltinTimeoutError
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

import anyio
import sniffio
from anyio import abc as anyio_abc


_T = TypeVar("_T")


def _get_cancelled_error_type() -> type[BaseException]:
    """Best-effort detection of the backend cancellation exception."""

    try:
        return anyio.get_cancelled_exc_class()
    except (sniffio.AsyncLibraryNotFoundError, RuntimeError):
        # No async context active yet – fall back to asyncio's cancelled error
        # since we currently run atop the asyncio backend.
        return asyncio.CancelledError


class AsyncRuntime:
    """Facade around the active async backend."""

    CancelledError = _get_cancelled_error_type()
    TimeoutError = BuiltinTimeoutError

    def create_task(
        self,
        coro: Coroutine[Any, Any, _T],
        *,
        name: str | None = None,
    ) -> asyncio.Task[_T]:
        """Create a background task using the active runtime."""
        return asyncio.create_task(coro, name=name)

    async def gather(
        self,
        *aws: Awaitable[Any],
        return_exceptions: bool = False,
    ) -> list[Any]:
        """Gather awaitables using the active runtime."""
        results = await asyncio.gather(*aws, return_exceptions=return_exceptions)
        return list(results)

    async def wait_for(
        self,
        awaitable: Awaitable[_T],
        timeout: float | None,
    ) -> _T:
        """Wait for an awaitable with an optional timeout."""
        if timeout is None:
            return await awaitable

        try:
            with anyio.fail_after(timeout):
                return await awaitable
        except BuiltinTimeoutError as exc:  # pragma: no cover - defensive
            raise self.TimeoutError(str(exc)) from exc

    async def wait(
        self,
        aws: Iterable[asyncio.Future[Any]],
        *,
        timeout: float | None = None,
        return_when: str = asyncio.ALL_COMPLETED,
    ) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
        """Wait for awaitables using the active runtime."""
        done, pending = await asyncio.wait(
            aws, timeout=timeout, return_when=return_when
        )
        return done, pending

    def create_lock(self) -> anyio.Lock:
        """Return a new lock instance."""
        return anyio.Lock()

    def create_event(self) -> asyncio.Event:
        """Return a new event instance."""
        return asyncio.Event()

    def create_semaphore(self, value: int) -> anyio.Semaphore:
        """Return a new semaphore instance."""
        return anyio.Semaphore(value)

    def create_queue(self, maxsize: int = 0) -> asyncio.Queue[Any]:
        """Return a new queue instance."""
        return asyncio.Queue(maxsize=maxsize)

    @asynccontextmanager
    async def task_group(self) -> AsyncIterator[anyio_abc.TaskGroup]:
        """Yield an anyio task group tied to the active runtime."""

        async with anyio.create_task_group() as group:
            yield group

    def cancel_scope(self) -> anyio.CancelScope:
        """Return a new anyio cancel scope."""

        return anyio.CancelScope()

    def memory_object_stream(
        self, max_buffer_size: int = 0
    ) -> tuple[anyio_abc.ObjectSendStream[Any], anyio_abc.ObjectReceiveStream[Any]]:
        """Return connected send/receive streams for in-memory messaging."""

        return anyio.create_memory_object_stream(max_buffer_size)

    def create_future(self) -> asyncio.Future[Any]:
        """Create a future bound to the active event loop."""
        loop = asyncio.get_running_loop()
        return loop.create_future()

    async def sleep(self, delay: float) -> None:
        """Sleep for the requested delay."""
        await anyio.sleep(delay)

    def run(self, awaitable: Awaitable[_T]) -> _T:
        """Run an awaitable to completion using the active runtime."""
        if not asyncio.iscoroutine(awaitable):
            raise TypeError("runtime.run() expects a coroutine object")

        async def _runner() -> _T:
            return cast(_T, await awaitable)

        return anyio.run(_runner)

    async def run_in_executor(
        self, func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T:
        """Execute ``func`` in the default executor."""
        if kwargs:
            func = functools.partial(func, **kwargs)

        return await anyio.to_thread.run_sync(func, *args)

    async def to_thread(
        self, func: Callable[..., _T], /, *args: Any, **kwargs: Any
    ) -> _T:
        """Execute ``func`` in a worker thread via the runtime."""
        if kwargs:
            func = functools.partial(func, **kwargs)

        return await anyio.to_thread.run_sync(func, *args)

    async def create_subprocess_exec(
        self, *cmd: Any, **kwargs: Any
    ) -> asyncio_subprocess.Process:
        """Spawn a subprocess using the active runtime."""
        return await asyncio.create_subprocess_exec(*cmd, **kwargs)

    def get_loop_time(self) -> float:
        """Return the loop's time helper."""
        try:
            return anyio.current_time()
        except sniffio.AsyncLibraryNotFoundError:
            return time.perf_counter()

    def current_task(self) -> asyncio.Task[Any] | None:
        """Return the current task, if any."""
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None


runtime = AsyncRuntime()


def create_task(
    coro: Coroutine[Any, Any, _T],
    *,
    name: str | None = None,
) -> asyncio.Task[_T]:
    """Proxy helper for ``AsyncRuntime.create_task``."""
    return runtime.create_task(coro, name=name)


async def gather(
    *aws: Awaitable[Any],
    return_exceptions: bool = False,
) -> list[Any]:
    """Gather awaitables using the active runtime."""
    return await runtime.gather(*aws, return_exceptions=return_exceptions)


async def wait_for(awaitable: Awaitable[_T], timeout: float | None) -> _T:
    """Wait for an awaitable with a timeout using the active runtime."""
    return await runtime.wait_for(awaitable, timeout)


async def wait(
    aws: Iterable[asyncio.Future[Any]],
    *,
    timeout: float | None = None,
    return_when: str = asyncio.ALL_COMPLETED,
) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
    """Wait for awaitables using the active runtime."""
    return await runtime.wait(aws, timeout=timeout, return_when=return_when)


async def sleep(delay: float) -> None:
    """Sleep for the requested delay using the runtime."""
    await runtime.sleep(delay)


def run(awaitable: Awaitable[_T]) -> _T:
    """Run an awaitable to completion."""
    return runtime.run(awaitable)


async def run_in_executor(func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Execute ``func`` in the default executor via the runtime."""
    return await runtime.run_in_executor(func, *args, **kwargs)


async def create_subprocess_exec(
    *cmd: Any, **kwargs: Any
) -> asyncio_subprocess.Process:
    """Spawn a subprocess via the runtime abstraction."""
    return await runtime.create_subprocess_exec(*cmd, **kwargs)


async def to_thread(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Execute ``func`` using the runtime's thread helper."""
    return await runtime.to_thread(func, *args, **kwargs)


def create_lock() -> anyio.Lock:
    """Return a runtime-managed lock."""
    return runtime.create_lock()


def create_event() -> asyncio.Event:
    """Return a runtime-managed event."""
    return runtime.create_event()


def create_semaphore(value: int) -> anyio.Semaphore:
    """Return a runtime-managed semaphore."""
    return runtime.create_semaphore(value)


def create_queue(maxsize: int = 0) -> asyncio.Queue[Any]:
    """Return a runtime-managed queue."""
    return runtime.create_queue(maxsize=maxsize)


@asynccontextmanager
async def task_group() -> AsyncIterator[anyio_abc.TaskGroup]:
    """Yield an anyio task group tied to the runtime."""

    async with runtime.task_group() as group:
        yield group


def cancel_scope() -> anyio.CancelScope:
    """Return a runtime-managed cancel scope."""

    return runtime.cancel_scope()


def memory_object_stream(
    max_buffer_size: int = 0,
) -> tuple[anyio_abc.ObjectSendStream[Any], anyio_abc.ObjectReceiveStream[Any]]:
    """Return runtime-managed memory object streams."""

    return runtime.memory_object_stream(max_buffer_size)


def create_future() -> asyncio.Future[Any]:
    """Return a runtime-managed future."""
    return runtime.create_future()


def loop_time() -> float:
    """Return the runtime loop time helper."""
    return runtime.get_loop_time()


def current_task() -> asyncio.Task[Any] | None:
    """Return the current task from the runtime."""
    return runtime.current_task()


CancelledError = AsyncRuntime.CancelledError
TimeoutError = AsyncRuntime.TimeoutError
ALL_COMPLETED = asyncio.ALL_COMPLETED
FIRST_COMPLETED = asyncio.FIRST_COMPLETED
QueueEmpty = asyncio.QueueEmpty
QueueFull = asyncio.QueueFull
Task = asyncio.Task
Event = asyncio.Event
Lock = anyio.Lock
Queue = asyncio.Queue
Future = asyncio.Future
InvalidStateError = asyncio.InvalidStateError
Semaphore = anyio.Semaphore
PIPE = asyncio_subprocess.PIPE
STDOUT = asyncio_subprocess.STDOUT
