import asyncio
import time
from pathlib import Path

import aiofiles
import structlog


logger = structlog.get_logger(__name__)


class AccessLogWriter:
    """Simple async file writer for access logs.

    Features:
    - Async file I/O for performance
    - Optional buffering to reduce I/O operations
    - Thread-safe with asyncio.Lock
    - Auto-creates parent directories
    """

    def __init__(
        self,
        log_file: str,
        buffer_size: int = 100,
        flush_interval: float = 1.0,
    ):
        """Initialize the writer.

        Args:
            log_file: Path to the log file
            buffer_size: Number of entries to buffer before writing
            flush_interval: Time in seconds between automatic flushes
        """
        self.log_file = Path(log_file)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self._buffer: list[str] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._last_flush = time.time()

        # Ensure parent directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, line: str) -> None:
        """Write a line to the log file.

        Lines are buffered and written in batches for performance.

        Args:
            line: The formatted log line to write
        """
        async with self._lock:
            self._buffer.append(line)

            # Flush if buffer is full
            if len(self._buffer) >= self.buffer_size:
                await self._flush()
            else:
                # Schedule a flush if not already scheduled
                self._schedule_flush()

    async def _flush(self) -> None:
        """Flush the buffer to disk.

        This method assumes the lock is already held.
        """
        if not self._buffer:
            return

        try:
            # Write all buffered lines at once
            async with aiofiles.open(self.log_file, "a") as f:
                await f.write("\n".join(self._buffer) + "\n")

            self._buffer.clear()
            self._last_flush = time.time()

        except Exception as e:
            logger.error(
                "access_log_write_error",
                error=str(e),
                log_file=str(self.log_file),
                buffer_size=len(self._buffer),
            )

    def _schedule_flush(self) -> None:
        """Schedule an automatic flush after the flush interval."""
        if self._flush_task and not self._flush_task.done():
            return  # Already scheduled

        self._flush_task = asyncio.create_task(self._auto_flush())

    async def _auto_flush(self) -> None:
        """Automatically flush the buffer after the flush interval."""
        await asyncio.sleep(self.flush_interval)
        async with self._lock:
            await self._flush()

    async def close(self) -> None:
        """Close the writer and flush any remaining data."""
        async with self._lock:
            await self._flush()

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
