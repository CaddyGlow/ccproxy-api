"""Pricing cache management for dynamic model pricing."""

import json
import time
from typing import Any

import httpx

from ccproxy.core.logging import get_plugin_logger

from .config import PricingConfig


logger = get_plugin_logger(__name__)


class PricingCache:
    """Manages caching of model pricing data from external sources."""

    def __init__(self, settings: PricingConfig) -> None:
        """Initialize pricing cache.

        Args:
            settings: Pricing configuration settings
        """
        self.settings = settings
        self.cache_dir = settings.cache_dir
        self.cache_file = self.cache_dir / "model_pricing.json"

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_cache_valid(self) -> bool:
        """Check if cached pricing data is still valid.

        Returns:
            True if cache exists and is not expired
        """
        if not self.cache_file.exists():
            return False

        try:
            stat = self.cache_file.stat()
            age_seconds = time.time() - stat.st_mtime
            age_hours = age_seconds / 3600

            is_valid = age_hours < self.settings.cache_ttl_hours
            return is_valid

        except OSError as e:
            logger.warning("cache_stats_check_failed", error=str(e))
            return False

    def load_cached_data(self) -> dict[str, Any] | None:
        """Load pricing data from cache.

        Returns:
            Cached pricing data or None if cache is invalid/corrupted
        """
        if not self.is_cache_valid():
            return None

        try:
            with self.cache_file.open(encoding="utf-8") as f:
                data = json.load(f)

            return data  # type: ignore[no-any-return]

        except (OSError, json.JSONDecodeError) as e:
            logger.warning("cache_load_failed", error=str(e))
            return None

    async def download_pricing_data(
        self, timeout: int | None = None
    ) -> dict[str, Any] | None:
        """Download fresh pricing data from source URL.

        Args:
            timeout: Request timeout in seconds (uses settings default if None)

        Returns:
            Downloaded pricing data or None if download failed
        """
        if timeout is None:
            timeout = self.settings.download_timeout

        try:
            logger.debug("pricing_download_start", url=self.settings.source_url)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(self.settings.source_url)
                response.raise_for_status()

                data = response.json()
                logger.debug("pricing_download_completed", model_count=len(data))
                return data  # type: ignore[no-any-return]

        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.error("pricing_download_failed", error=str(e))
            return None

    def save_to_cache(self, data: dict[str, Any]) -> bool:
        """Save pricing data to cache.

        Args:
            data: Pricing data to cache

        Returns:
            True if successfully saved, False otherwise
        """
        try:
            # Write to temporary file first, then atomic rename
            temp_file = self.cache_file.with_suffix(".tmp")

            with temp_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_file.replace(self.cache_file)

            return True

        except OSError as e:
            logger.error("cache_save_failed", error=str(e))
            return False

    async def get_pricing_data(
        self, force_refresh: bool = False
    ) -> dict[str, Any] | None:
        """Get pricing data, from cache if valid or by downloading fresh data.

        Args:
            force_refresh: Force download even if cache is valid

        Returns:
            Pricing data or None if both cache and download fail
        """
        # Try cache first unless forced refresh
        if not force_refresh:
            cached_data = self.load_cached_data()
            if cached_data is not None:
                return cached_data

        # Download fresh data
        fresh_data = await self.download_pricing_data()
        if fresh_data is not None:
            # Save to cache for next time
            self.save_to_cache(fresh_data)
            return fresh_data

        # If download failed, try to use stale cache as fallback
        if not force_refresh:
            logger.warning("pricing_download_failed_using_stale_cache")
            try:
                with self.cache_file.open(encoding="utf-8") as f:
                    stale_data = json.load(f)
                logger.warning("stale_cache_used")
                return stale_data  # type: ignore[no-any-return]
            except (OSError, json.JSONDecodeError):
                pass

        logger.error("pricing_data_unavailable")
        return None

    def clear_cache(self) -> bool:
        """Clear cached pricing data.

        Returns:
            True if cache was cleared successfully
        """
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
            return True
        except OSError as e:
            logger.error("cache_clear_failed", error=str(e))
            return False

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cache status.

        Returns:
            Dictionary with cache information
        """
        info = {
            "cache_file": str(self.cache_file),
            "cache_dir": str(self.cache_dir),
            "source_url": self.settings.source_url,
            "ttl_hours": self.settings.cache_ttl_hours,
            "exists": self.cache_file.exists(),
            "valid": False,
            "age_hours": None,
            "size_bytes": None,
        }

        if self.cache_file.exists():
            try:
                stat = self.cache_file.stat()
                age_seconds = time.time() - stat.st_mtime
                age_hours = age_seconds / 3600

                info.update(
                    {
                        "valid": age_hours < self.settings.cache_ttl_hours,
                        "age_hours": age_hours,
                        "size_bytes": stat.st_size,
                    }
                )
            except OSError:
                pass

        return info
