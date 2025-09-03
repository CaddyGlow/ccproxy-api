from typing import Literal

from pydantic import BaseModel, ConfigDict


class AccessLogConfig(BaseModel):
    """Configuration for access logging.

    Supports logging at both client and provider levels with
    different formats for each.
    """

    # Global enable/disable
    enabled: bool = True

    # Client-level access logging
    client_enabled: bool = True
    client_format: Literal["combined", "common", "structured"] = "structured"
    client_log_file: str = "/tmp/ccproxy/access.log"

    # Provider-level access logging (optional)
    provider_enabled: bool = False
    provider_format: Literal["structured"] = "structured"
    provider_log_file: str = "/tmp/ccproxy/provider_access.log"

    # Path filters (only for client level)
    exclude_paths: list[str] = ["/health", "/metrics", "/readyz", "/livez"]

    # Performance options
    buffer_size: int = 100  # Buffer this many log entries before writing
    flush_interval: float = 1.0  # Flush buffer every N seconds

    # BaseModel's ConfigDict does not support case_sensitive; remove for mypy compatibility
    model_config = ConfigDict()
