"""Response caching for API requests."""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """A cached response entry."""

    key: str
    data: Any
    timestamp: float
    ttl: float

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl


class ResponseCache:
    """In-memory response cache with TTL support."""

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000) -> None:
        """Initialize the response cache.

        Args:
            default_ttl: Default time-to-live in seconds (5 minutes)
            max_size: Maximum number of cached entries
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self.logger = logger

    def _generate_key(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Generate a cache key for the request.

        Args:
            method: HTTP method
            url: Request URL
            body: Request body
            headers: Request headers

        Returns:
            Cache key string
        """
        # Include important headers in cache key
        cache_headers = {}
        if headers:
            for header in ["authorization", "x-api-key", "content-type"]:
                if header in headers:
                    cache_headers[header] = headers[header]

        key_parts = [
            method,
            url,
            body.decode("utf-8") if body else "",
            json.dumps(cache_headers, sort_keys=True),
        ]

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any | None:
        """Get a cached response if available and not expired.

        Args:
            method: HTTP method
            url: Request URL
            body: Request body
            headers: Request headers

        Returns:
            Cached response data or None
        """
        key = self._generate_key(method, url, body, headers)

        if key in self._cache:
            entry = self._cache[key]

            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self.logger.debug("cache_entry_expired", key=key[:8])
                return None

            # Update access order (LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self.logger.debug("cache_hit", key=key[:8])
            return entry.data

        self.logger.debug("cache_miss", key=key[:8])
        return None

    def set(
        self,
        method: str,
        url: str,
        data: Any,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Cache a response.

        Args:
            method: HTTP method
            url: Request URL
            data: Response data to cache
            body: Request body
            headers: Request headers
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Don't cache streaming responses
        if hasattr(data, "__aiter__"):
            return

        key = self._generate_key(method, url, body, headers)
        ttl = ttl or self.default_ttl

        # Enforce max size with LRU eviction
        if (
            len(self._cache) >= self.max_size
            and key not in self._cache
            and self._access_order
        ):
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            self.logger.debug("cache_evicted", key=oldest_key[:8])

        # Store the entry
        self._cache[key] = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            ttl=ttl,
        )

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self.logger.debug("cache_set", key=key[:8], ttl=ttl)

    def invalidate(
        self,
        method: str | None = None,
        url: str | None = None,
        pattern: str | None = None,
    ) -> int:
        """Invalidate cached entries.

        Args:
            method: HTTP method to match (None for any)
            url: URL to match (None for any)
            pattern: URL pattern to match (None for any)

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []

        for key, entry in self._cache.items():
            should_remove = False

            # Check if entry matches invalidation criteria
            if pattern and pattern in str(entry.data.get("url", "")):
                should_remove = True
            elif method and url:
                test_key = self._generate_key(method, url)
                if key == test_key:
                    should_remove = True

            if should_remove:
                keys_to_remove.append(key)

        # Remove matched entries
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

        if keys_to_remove:
            self.logger.info(
                "cache_invalidated",
                count=len(keys_to_remove),
                method=method,
                url=url,
                pattern=pattern,
            )

        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached entries."""
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        self.logger.info("cache_cleared", count=count)

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

        if expired_keys:
            self.logger.debug("cache_cleanup", removed=len(expired_keys))

        return len(expired_keys)

    @property
    def size(self) -> int:
        """Get the current cache size."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "oldest_entry": self._access_order[0][:8] if self._access_order else None,
            "newest_entry": self._access_order[-1][:8] if self._access_order else None,
        }
