"""Coverage for asynchronous utility helpers."""

import asyncio
from types import SimpleNamespace

import pytest

from ccproxy.core import async_utils


@pytest.mark.asyncio
async def test_run_in_executor_supports_kwargs() -> None:
    def combine(a: int, b: int = 0) -> int:
        return a + b

    result = await async_utils.run_in_executor(combine, 2, b=3)

    assert result == 5


@pytest.mark.asyncio
async def test_safe_await_handles_timeout() -> None:
    # safe_await returns a coroutine that needs to be awaited
    # For asyncio.sleep(), it returns None
    result1: None = await async_utils.safe_await(asyncio.sleep(0), timeout=0.1)  # type: ignore[func-returns-value]
    assert result1 is None  # plain sleep returns None

    result2: None = await async_utils.safe_await(asyncio.sleep(0.05), timeout=0.001)  # type: ignore[func-returns-value]
    assert result2 is None


@pytest.mark.asyncio
async def test_safe_await_logs_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        async_utils,
        "get_logger",
        lambda name: SimpleNamespace(
            debug=lambda *args, **kwargs: calls.append(kwargs["error"])
        ),
    )

    async def bad() -> str:
        raise RuntimeError("boom")

    result = await async_utils.safe_await(bad())

    assert result is None
    assert calls == ["boom"]


@pytest.mark.asyncio
async def test_gather_with_concurrency_limits_parallelism() -> None:
    active = 0
    peak = 0

    async def task(val: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)  # yield control
        active -= 1
        return val

    results = await async_utils.gather_with_concurrency(1, *(task(i) for i in range(3)))

    assert results == [0, 1, 2]
    assert peak == 1


@pytest.mark.asyncio
async def test_async_timer_reports_elapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([100.0, 100.5, 100.5])

    def fake_perf_counter() -> float:
        try:
            return next(times)
        except StopIteration:
            return 100.5

    monkeypatch.setattr("time.perf_counter", fake_perf_counter)

    async with async_utils.async_timer() as elapsed:
        pass

    assert pytest.approx(elapsed(), rel=1e-3) == 0.5


@pytest.mark.asyncio
async def test_retry_async_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    original_sleep = asyncio.sleep

    async def immediate_sleep(_: float) -> None:
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("retry")
        return "ok"

    result = await async_utils.retry_async(flaky, max_retries=3, delay=0)

    assert result == "ok"
    assert attempts == 3


@pytest.mark.asyncio
async def test_wait_for_condition_handles_async_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = 0

    async def condition() -> bool:
        nonlocal counter
        counter += 1
        await asyncio.sleep(0)
        return counter >= 2

    success = await async_utils.wait_for_condition(condition, timeout=0.1, interval=0)

    assert success is True


@pytest.mark.asyncio
async def test_async_cache_result_reuses_cached_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_utils._cache.clear()
    calls = 0

    async def compute(x: int) -> int:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return x * 2

    first = await async_utils.async_cache_result(compute, "key", 10, 2)
    second = await async_utils.async_cache_result(compute, "key", 10, 4)

    assert first == 4
    assert second == 4
    assert calls == 1


def test_parse_and_format_version_variants() -> None:
    assert async_utils.parse_version("1.2.3") == (1, 2, 3, "")
    assert async_utils.parse_version("0.1.dev5+g123") == (0, 1, 0, "dev")

    assert async_utils.format_version("1.2.3", "major") == "1"
    assert async_utils.format_version("1.2.3", "minor") == "1.2"
    assert async_utils.format_version("1.2.3", "patch") == "1.2.3"
    assert async_utils.format_version("1.2.3", "docker") == "1.2"
    assert async_utils.format_version("1.2.3", "npm") == "1.2.3"
    assert async_utils.format_version("1.2.3", "python") == "1.2.3"
