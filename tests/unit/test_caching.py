"""Unit tests for caching utilities."""

import asyncio
import time

import pytest

from ccproxy.utils.caching import (
    AuthStatusCache,
    TTLCache,
    async_ttl_cache,
    ttl_cache,
)


class TestTTLCache:
    """Test TTL cache implementation."""

    def test_basic_operations(self):
        """Test basic cache get/set/delete operations."""
        cache = TTLCache(maxsize=2, ttl=1.0)

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test non-existent key
        assert cache.get("nonexistent") is None

        # Test delete
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = TTLCache(maxsize=10, ttl=0.1)  # 100ms TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_maxsize_eviction(self):
        """Test LRU eviction when maxsize is exceeded."""
        cache = TTLCache(maxsize=2, ttl=10.0)  # Long TTL

        # Fill cache to max
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access key1 to make it more recent
        cache.get("key1")

        # Add third key, should evict key2 (oldest)
        cache.set("key3", "value3")

        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"  # Should exist

    def test_clear(self):
        """Test cache clear operation."""
        cache = TTLCache(maxsize=10, ttl=10.0)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats(self):
        """Test cache statistics."""
        cache = TTLCache(maxsize=5, ttl=60.0)

        cache.set("key1", "value1")
        stats = cache.stats()

        assert stats["maxsize"] == 5
        assert stats["ttl"] == 60.0
        assert stats["size"] == 1


class TestTTLCacheDecorator:
    """Test TTL cache decorator."""

    def test_function_caching(self):
        """Test that function results are cached."""
        call_count = 0

        @ttl_cache(maxsize=10, ttl=10.0)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same argument - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

        # Call with different argument
        result3 = expensive_function(3)
        assert result3 == 6
        assert call_count == 2

    def test_cache_clear(self):
        """Test cache clear functionality."""
        call_count = 0

        @ttl_cache(maxsize=10, ttl=10.0)
        def test_function(x):
            nonlocal call_count
            call_count += 1
            return x

        # First call
        test_function(1)
        assert call_count == 1

        # Second call - cached
        test_function(1)
        assert call_count == 1

        # Clear cache
        test_function.cache_clear()  # type: ignore[attr-defined]

        # Third call - should call function again
        test_function(1)
        assert call_count == 2


class TestAsyncTTLCacheDecorator:
    """Test async TTL cache decorator."""

    @pytest.mark.asyncio
    async def test_async_function_caching(self):
        """Test that async function results are cached."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl=10.0)
        async def expensive_async_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            return x * 2

        # First call
        result1 = await expensive_async_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same argument - should use cache
        result2 = await expensive_async_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment

        # Call with different argument
        result3 = await expensive_async_function(3)
        assert result3 == 6
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_cache_expiration(self):
        """Test that async cache entries expire."""
        call_count = 0

        @async_ttl_cache(maxsize=10, ttl=0.1)  # 100ms TTL
        async def test_function(x):
            nonlocal call_count
            call_count += 1
            return x

        # First call
        await test_function(1)
        assert call_count == 1

        # Second call - should be cached
        await test_function(1)
        assert call_count == 1

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Third call - cache expired
        await test_function(1)
        assert call_count == 2


class TestAuthStatusCache:
    """Test auth status cache."""

    def test_auth_status_operations(self):
        """Test auth status cache operations."""
        cache = AuthStatusCache(ttl=1.0)

        # Test set and get
        cache.set_auth_status("provider1", True)
        assert cache.get_auth_status("provider1") is True

        # Test non-existent provider
        assert cache.get_auth_status("nonexistent") is None

        # Test invalidation
        cache.invalidate_auth_status("provider1")
        assert cache.get_auth_status("provider1") is None

    def test_auth_status_expiration(self):
        """Test that auth status expires."""
        cache = AuthStatusCache(ttl=0.1)  # 100ms TTL

        cache.set_auth_status("provider1", True)
        assert cache.get_auth_status("provider1") is True

        # Wait for expiration
        time.sleep(0.15)
        assert cache.get_auth_status("provider1") is None

    def test_auth_cache_clear(self):
        """Test clearing all auth cache."""
        cache = AuthStatusCache(ttl=10.0)

        cache.set_auth_status("provider1", True)
        cache.set_auth_status("provider2", False)

        cache.clear()

        assert cache.get_auth_status("provider1") is None
        assert cache.get_auth_status("provider2") is None


class TestCacheIntegration:
    """Integration tests for caching functionality."""

    @pytest.mark.asyncio
    async def test_mock_detection_service_caching(self):
        """Test that detection service methods can be cached."""
        call_count = 0

        class MockDetectionService:
            @async_ttl_cache(maxsize=8, ttl=10.0)
            async def initialize_detection(self):
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.01)  # Simulate work
                return {"version": "1.0.0", "available": True}

        service = MockDetectionService()

        # First call
        result1 = await service.initialize_detection()
        assert call_count == 1
        assert result1["version"] == "1.0.0"

        # Second call - should be cached
        result2 = await service.initialize_detection()
        assert call_count == 1  # No additional calls
        assert result2 == result1

    def test_mock_auth_manager_caching(self):
        """Test that auth manager methods can be cached."""
        call_count = 0

        class MockAuthManager:
            def __init__(self):
                self._auth_cache = AuthStatusCache(ttl=60.0)

            async def is_authenticated(self):
                # Check cache first
                cached_result = self._auth_cache.get_auth_status("test-provider")
                if cached_result is not None:
                    return cached_result

                # Simulate expensive auth check
                nonlocal call_count
                call_count += 1
                result = True  # Mock always authenticated

                # Cache result
                self._auth_cache.set_auth_status("test-provider", result)
                return result

        auth_manager = MockAuthManager()

        # First call
        result1 = asyncio.run(auth_manager.is_authenticated())
        assert result1 is True
        assert call_count == 1

        # Second call - should be cached
        result2 = asyncio.run(auth_manager.is_authenticated())
        assert result2 is True
        assert call_count == 1  # No additional calls
