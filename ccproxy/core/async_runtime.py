"""Abstraction layer for async primitives used across the project.

This module centralizes creation of async primitives and task helpers.  It
currently delegates to ``asyncio`` but provides a single choke point for
upcoming ``anyio`` migration work.  Modules should import helpers from here
instead of directly touching ``asyncio`` so that the underlying runtime can be
swapped without widespread churn.
"""

from __future__ import annotations

import asyncio
from asyncio import subprocess as asyncio_subprocess
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar


_T = TypeVar("_T")


class AsyncRuntime:
    """Facade around the active async backend."""

    CancelledError = asyncio.CancelledError
    TimeoutError = asyncio.TimeoutError

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
        return await asyncio.wait_for(awaitable, timeout=timeout)

    def create_lock(self) -> asyncio.Lock:
        """Return a new lock instance."""
        return asyncio.Lock()

    def create_event(self) -> asyncio.Event:
        """Return a new event instance."""
        return asyncio.Event()

    def create_semaphore(self, value: int) -> asyncio.Semaphore:
        """Return a new semaphore instance."""
        return asyncio.Semaphore(value)

    def create_queue(self, maxsize: int = 0) -> asyncio.Queue[Any]:
        """Return a new queue instance."""
        return asyncio.Queue(maxsize=maxsize)

    def create_future(self) -> asyncio.Future[Any]:
        """Create a future bound to the active event loop."""
        loop = asyncio.get_running_loop()
        return loop.create_future()

    async def sleep(self, delay: float) -> None:
        """Sleep for the requested delay."""
        await asyncio.sleep(delay)

    def run(self, awaitable: Coroutine[Any, Any, _T]) -> _T:
        """Run an awaitable to completion using the active runtime."""
        return asyncio.run(awaitable)

    async def run_in_executor(
        self, func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T:
        """Execute ``func`` in the default executor."""
        loop = asyncio.get_running_loop()

        if kwargs:
            from functools import partial

            func = partial(func, **kwargs)

        return await loop.run_in_executor(None, func, *args)

    async def to_thread(
        self, func: Callable[..., _T], /, *args: Any, **kwargs: Any
    ) -> _T:
        """Execute ``func`` in a worker thread via the runtime."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def create_subprocess_exec(
        self, *cmd: Any, **kwargs: Any
    ) -> asyncio_subprocess.Process:
        """Spawn a subprocess using the active runtime."""
        return await asyncio.create_subprocess_exec(*cmd, **kwargs)

    def get_loop_time(self) -> float:
        """Return the loop's time helper."""
        return asyncio.get_event_loop().time()

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


async def sleep(delay: float) -> None:
    """Sleep for the requested delay using the runtime."""
    await runtime.sleep(delay)


def run(awaitable: Coroutine[Any, Any, _T]) -> _T:
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


def create_lock() -> asyncio.Lock:
    """Return a runtime-managed lock."""
    return runtime.create_lock()


def create_event() -> asyncio.Event:
    """Return a runtime-managed event."""
    return runtime.create_event()


def create_semaphore(value: int) -> asyncio.Semaphore:
    """Return a runtime-managed semaphore."""
    return runtime.create_semaphore(value)


def create_queue(maxsize: int = 0) -> asyncio.Queue[Any]:
    """Return a runtime-managed queue."""
    return runtime.create_queue(maxsize=maxsize)


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
