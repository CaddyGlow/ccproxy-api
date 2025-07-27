"""Simple request logging utility for content logging across all service layers."""

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog


logger = structlog.get_logger(__name__)

# Global batching settings for streaming logs
_STREAMING_BATCH_SIZE = 8192  # Batch chunks until we have 8KB
_STREAMING_BATCH_TIMEOUT = 0.1  # Or flush after 100ms
_streaming_batches: dict[str, dict[str, Any]] = {}  # request_id -> batch info


def should_log_requests() -> bool:
    """Check if request logging is enabled via environment variable.

    Returns:
        True if CCPROXY_LOG_REQUESTS is set to 'true' (case-insensitive)
    """
    return os.environ.get("CCPROXY_LOG_REQUESTS", "false").lower() == "true"


def get_request_log_dir() -> Path | None:
    """Get the request log directory from environment variable.

    Returns:
        Path object if CCPROXY_REQUEST_LOG_DIR is set and valid, None otherwise
    """
    log_dir = os.environ.get("CCPROXY_REQUEST_LOG_DIR")
    if not log_dir:
        return None

    path = Path(log_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger.error(
            "failed_to_create_request_log_dir",
            log_dir=log_dir,
            error=str(e),
        )
        return None


def get_timestamp_prefix() -> str:
    """Generate timestamp prefix in YYYYMMDDhhmmss format.

    Returns:
        Timestamp string in YYYYMMDDhhmmss format (UTC)
    """
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


async def write_request_log(
    request_id: str,
    log_type: str,
    data: dict[str, Any],
    timestamp: str | None = None,
) -> None:
    """Write request/response data to JSON file.

    Args:
        request_id: Unique request identifier
        log_type: Type of log (e.g., 'middleware_request', 'upstream_response')
        data: Data to log as JSON
        timestamp: Optional timestamp prefix (defaults to current time)
    """
    if not should_log_requests():
        return

    log_dir = get_request_log_dir()
    if not log_dir:
        return

    timestamp = timestamp or get_timestamp_prefix()
    filename = f"{timestamp}_{request_id}_{log_type}.json"
    file_path = log_dir / filename

    try:
        # Write JSON data to file asynchronously
        def write_file() -> None:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)

        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, write_file)

        logger.debug(
            "request_log_written",
            request_id=request_id,
            log_type=log_type,
            file_path=str(file_path),
        )

    except Exception as e:
        logger.error(
            "failed_to_write_request_log",
            request_id=request_id,
            log_type=log_type,
            file_path=str(file_path),
            error=str(e),
        )


async def write_streaming_log(
    request_id: str,
    log_type: str,
    data: bytes,
    timestamp: str | None = None,
) -> None:
    """Write streaming data to raw file.

    Args:
        request_id: Unique request identifier
        log_type: Type of log (e.g., 'middleware_streaming', 'upstream_streaming')
        data: Raw bytes to log
        timestamp: Optional timestamp prefix (defaults to current time)
    """
    if not should_log_requests():
        return

    log_dir = get_request_log_dir()
    if not log_dir:
        return

    timestamp = timestamp or get_timestamp_prefix()
    filename = f"{timestamp}_{request_id}_{log_type}.raw"
    file_path = log_dir / filename

    try:
        # Write raw data to file asynchronously
        def write_file() -> None:
            with file_path.open("wb") as f:
                f.write(data)

        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, write_file)

        logger.debug(
            "streaming_log_written",
            request_id=request_id,
            log_type=log_type,
            file_path=str(file_path),
            data_size=len(data),
        )

    except Exception as e:
        logger.error(
            "failed_to_write_streaming_log",
            request_id=request_id,
            log_type=log_type,
            file_path=str(file_path),
            error=str(e),
        )


async def append_streaming_log(
    request_id: str,
    log_type: str,
    data: bytes,
    timestamp: str | None = None,
) -> None:
    """Append streaming data using batching for performance.

    Args:
        request_id: Unique request identifier
        log_type: Type of log (e.g., 'middleware_streaming', 'upstream_streaming')
        data: Raw bytes to append
        timestamp: Optional timestamp prefix (defaults to current time)
    """
    if not should_log_requests():
        return

    log_dir = get_request_log_dir()
    if not log_dir:
        return

    timestamp = timestamp or get_timestamp_prefix()
    batch_key = f"{request_id}_{log_type}"

    # Get or create batch for this request/log_type combination
    if batch_key not in _streaming_batches:
        _streaming_batches[batch_key] = {
            "request_id": request_id,
            "log_type": log_type,
            "timestamp": timestamp,
            "data": bytearray(),
            "chunk_count": 0,
            "first_chunk_time": asyncio.get_event_loop().time(),
            "last_flush_task": None,
        }

    batch = _streaming_batches[batch_key]
    batch["data"].extend(data)
    batch["chunk_count"] += 1

    # Cancel previous flush task if it exists
    if batch["last_flush_task"] and not batch["last_flush_task"].done():
        batch["last_flush_task"].cancel()

    # Check if we should flush now
    should_flush = (
        len(batch["data"]) >= _STREAMING_BATCH_SIZE
        or batch["chunk_count"] >= 50  # Max 50 chunks per batch
    )

    if should_flush:
        await _flush_streaming_batch(batch_key)
    else:
        # Schedule a delayed flush
        batch["last_flush_task"] = asyncio.create_task(
            _delayed_flush_streaming_batch(batch_key, _STREAMING_BATCH_TIMEOUT)
        )


async def _delayed_flush_streaming_batch(batch_key: str, delay: float) -> None:
    """Flush a streaming batch after a delay."""
    try:
        await asyncio.sleep(delay)
        if batch_key in _streaming_batches:
            await _flush_streaming_batch(batch_key)
    except asyncio.CancelledError:
        # Task was cancelled, don't flush
        pass


async def _flush_streaming_batch(batch_key: str) -> None:
    """Flush a streaming batch to disk."""
    if batch_key not in _streaming_batches:
        return

    batch = _streaming_batches.pop(batch_key)

    if not batch["data"]:
        return  # Nothing to flush

    log_dir = get_request_log_dir()
    if not log_dir:
        return

    filename = f"{batch['timestamp']}_{batch['request_id']}_{batch['log_type']}.raw"
    file_path = log_dir / filename

    try:
        # Append batched data to file asynchronously
        def append_file() -> None:
            with file_path.open("ab") as f:
                f.write(batch["data"])

        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, append_file)

        logger.debug(
            "streaming_batch_flushed",
            request_id=batch["request_id"],
            log_type=batch["log_type"],
            file_path=str(file_path),
            batch_size=len(batch["data"]),
            chunk_count=batch["chunk_count"],
        )

    except Exception as e:
        logger.error(
            "failed_to_flush_streaming_batch",
            request_id=batch["request_id"],
            log_type=batch["log_type"],
            file_path=str(file_path),
            error=str(e),
        )


async def flush_all_streaming_batches() -> None:
    """Flush all pending streaming batches. Call this on shutdown."""
    batch_keys = list(_streaming_batches.keys())
    for batch_key in batch_keys:
        await _flush_streaming_batch(batch_key)
