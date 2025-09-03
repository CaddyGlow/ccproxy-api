"""Proxy and SSL configuration management service."""

import os
from pathlib import Path
from typing import Any

import httpx
import structlog


logger = structlog.get_logger(__name__)


class ProxyConfiguration:
    """Manages proxy and SSL configuration from environment."""

    def __init__(self) -> None:
        """Initialize by reading environment variables.

        - Calls _init_proxy_url()
        - Calls _init_ssl_context()
        - Caches configuration
        """
        self._proxy_url = self._init_proxy_url()
        self._ssl_verify = self._init_ssl_context()

        if self._proxy_url:
            logger.info("proxy_configuration_detected", proxy_url=self._proxy_url)
        if isinstance(self._ssl_verify, str):
            logger.info("custom_ca_bundle_configured", ca_bundle=self._ssl_verify)
        elif not self._ssl_verify:
            logger.warning("ssl_verification_disabled_not_recommended_for_production")

    def _init_proxy_url(self) -> str | None:
        """Extract proxy URL from environment.

        - Checks HTTPS_PROXY (highest priority)
        - Falls back to ALL_PROXY
        - Falls back to HTTP_PROXY
        - Handles case variations
        """
        # Check in order of priority
        proxy_vars = [
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "HTTP_PROXY",
            "http_proxy",
        ]

        for var in proxy_vars:
            proxy_url = os.getenv(var)
            if proxy_url:
                return proxy_url

        return None

    def _init_ssl_context(self) -> str | bool:
        """Configure SSL verification and CA bundle.

        - Checks REQUESTS_CA_BUNDLE for custom CA
        - Checks SSL_CERT_FILE as fallback
        - Checks SSL_VERIFY for disabling (not recommended)
        - Returns: path | True | False
        """
        # Check for custom CA bundle
        ca_bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
        if ca_bundle:
            ca_path = Path(ca_bundle)
            if ca_path.exists() and ca_path.is_file():
                return str(ca_path)
            else:
                logger.warning("ca_bundle_file_not_found", ca_bundle=ca_bundle)

        # Check if SSL verification should be disabled
        ssl_verify = os.getenv("SSL_VERIFY", "true").lower()
        return ssl_verify not in ("false", "0", "no", "off")

    @property
    def proxy_url(self) -> str | None:
        """Get configured proxy URL if any."""
        return self._proxy_url

    @property
    def ssl_verify(self) -> str | bool:
        """Get SSL verification setting."""
        return self._ssl_verify

    def get_httpx_client_config(self) -> dict[str, Any]:
        """Build configuration dict for httpx.AsyncClient.

        - Includes 'proxy' if proxy configured
        - Includes 'verify' for SSL settings
        - Ready to pass to client constructor
        """
        config = {
            "verify": self._ssl_verify,
            "timeout": 120.0,  # Default timeout
            "follow_redirects": False,
            "limits": httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000,
                keepalive_expiry=30.0,
            ),
        }

        if self._proxy_url:
            config["proxy"] = self._proxy_url

        return config
