"""Caching utilities for CCProxy.

This module provides caching decorators and utilities to improve performance
by caching frequently accessed data like detection results and auth status.
"""

import functools
import threading
import time
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

from ccproxy.core.logging import TraceBoundLogger, get_logger


logger: TraceBoundLogger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class TTLCache:
    """Thread-safe TTL (Time To Live) cache with LRU eviction."""

    def __init__(self, maxsize: int = 128, ttl: float = 300.0):
        """Initialize TTL cache.

        Args:
            maxsize: Maximum number of entries to cache
            ttl: Time to live for entries in seconds
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: dict[Hashable, tuple[Any, float]] = {}
        self._access_order: dict[Hashable, int] = {}
        self._access_counter = 0
        self._lock = threading.RLock()

    def get(self, key: Hashable) -> Any | None:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None

            value, expiry_time = self._cache[key]

            # Check if expired
            if time.time() > expiry_time:
                self._cache.pop(key, None)
                self._access_order.pop(key, None)
                return None

            # Update access order
            self._access_counter += 1
            self._access_order[key] = self._access_counter

            return value

    def set(self, key: Hashable, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            now = time.time()
            expiry_time = now + self.ttl

            # Add/update entry
            self._cache[key] = (value, expiry_time)
            self._access_counter += 1
            self._access_order[key] = self._access_counter

            # Evict expired entries first
            self._evict_expired()

            # Evict oldest entries if over maxsize
            while len(self._cache) > self.maxsize:
                self._evict_oldest()

    def delete(self, key: Hashable) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.pop(key, None)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_counter = 0

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key for key, (_, expiry_time) in self._cache.items() if now > expiry_time
        ]

        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_order.pop(key, None)

    def _evict_oldest(self) -> None:
        """Remove oldest accessed entry."""
        if not self._access_order:
            return

        oldest_key = min(self._access_order, key=lambda k: self._access_order[k])
        self._cache.pop(oldest_key, None)
        self._access_order.pop(oldest_key, None)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "ttl": self.ttl,
            }


def ttl_cache(maxsize: int = 128, ttl: float = 300.0) -> Callable[[F], F]:
    """TTL cache decorator for functions.

    Args:
        maxsize: Maximum number of entries to cache
        ttl: Time to live for cached results in seconds
    """

    def decorator(func: F) -> F:
        cache = TTLCache(maxsize=maxsize, ttl=ttl)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key from function args/kwargs
            key = _make_cache_key(func.__name__, args, kwargs)

            # Try to get from cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.trace(
                    "cache_hit",
                    function=func.__name__,
                    key_hash=hash(key) if isinstance(key, tuple) else key,
                )
                return cached_result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)

            logger.trace(
                "cache_miss_and_set",
                function=func.__name__,
                key_hash=hash(key) if isinstance(key, tuple) else key,
                cache_size=len(cache._cache),
            )

            return result

        # Add cache management methods
        wrapper.cache_info = cache.stats  # type: ignore
        wrapper.cache_clear = cache.clear  # type: ignore

        return wrapper  # type: ignore

    return decorator


def async_ttl_cache(
    maxsize: int = 128, ttl: float = 300.0
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """TTL cache decorator for async functions.

    Args:
        maxsize: Maximum number of entries to cache
        ttl: Time to live for cached results in seconds
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache = TTLCache(maxsize=maxsize, ttl=ttl)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key from function args/kwargs
            key = _make_cache_key(func.__name__, args, kwargs)

            # Try to get from cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.trace(
                    "async_cache_hit",
                    function=func.__name__,
                    key_hash=hash(key) if isinstance(key, tuple) else key,
                )
                return cached_result

            # Call async function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result)

            logger.trace(
                "async_cache_miss_and_set",
                function=func.__name__,
                key_hash=hash(key) if isinstance(key, tuple) else key,
                cache_size=len(cache._cache),
            )

            return result

        # Add cache management methods
        wrapper.cache_info = cache.stats  # type: ignore
        wrapper.cache_clear = cache.clear  # type: ignore

        return wrapper

    return decorator


def _make_cache_key(
    func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Hashable:
    """Create a hashable cache key from function arguments."""
    try:
        # Try to create a simple key for basic types
        key_parts = [func_name]

        # Add positional args
        for arg in args:
            if hasattr(arg, "__dict__"):
                # For objects, use class name and id (weak ref to avoid memory leaks)
                key_parts.append(f"{type(arg).__name__}:{id(arg)}")
            else:
                key_parts.append(arg)

        # Add keyword args (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            if hasattr(v, "__dict__"):
                key_parts.append(f"{k}={type(v).__name__}:{id(v)}")
            else:
                key_parts.append(f"{k}={v}")

        return tuple(key_parts)

    except (TypeError, ValueError):
        # Fallback to string representation
        return f"{func_name}:{hash(str(args))}:{hash(str(sorted(kwargs.items())))}"


class AuthStatusCache:
    """Specialized cache for auth status checks with shorter TTL."""

    def __init__(self, ttl: float = 60.0):  # 1 minute TTL for auth status
        """Initialize auth status cache.

        Args:
            ttl: Time to live for auth status in seconds
        """
        self._cache = TTLCache(maxsize=32, ttl=ttl)

    def get_auth_status(self, provider: str) -> bool | None:
        """Get cached auth status for provider."""
        return self._cache.get(f"auth_status:{provider}")

    def set_auth_status(self, provider: str, is_authenticated: bool) -> None:
        """Cache auth status for provider."""
        self._cache.set(f"auth_status:{provider}", is_authenticated)

    def invalidate_auth_status(self, provider: str) -> None:
        """Invalidate auth status for provider."""
        self._cache.delete(f"auth_status:{provider}")

    def clear(self) -> None:
        """Clear all auth status cache."""
        self._cache.clear()


# Global instances for common use cases
_detection_cache = TTLCache(maxsize=64, ttl=600.0)  # 10 minute TTL for detection
_auth_cache = AuthStatusCache(ttl=60.0)  # 1 minute TTL for auth status
_config_cache = TTLCache(maxsize=32, ttl=300.0)  # 5 minute TTL for plugin configs


def cache_detection_result(key: str, result: Any) -> None:
    """Cache a detection result."""
    _detection_cache.set(f"detection:{key}", result)


def get_cached_detection_result(key: str) -> Any | None:
    """Get cached detection result."""
    return _detection_cache.get(f"detection:{key}")


def cache_plugin_config(plugin_name: str, config: Any) -> None:
    """Cache plugin configuration."""
    _config_cache.set(f"plugin_config:{plugin_name}", config)


def get_cached_plugin_config(plugin_name: str) -> Any | None:
    """Get cached plugin configuration."""
    return _config_cache.get(f"plugin_config:{plugin_name}")


def clear_all_caches() -> None:
    """Clear all global caches."""
    _detection_cache.clear()
    _auth_cache.clear()
    _config_cache.clear()
    logger.info("all_caches_cleared", category="cache")


def get_cache_stats() -> dict[str, Any]:
    """Get statistics for all caches."""
    return {
        "detection_cache": _detection_cache.stats(),
        "auth_cache": _auth_cache._cache.stats(),
        "config_cache": _config_cache.stats(),
    }
