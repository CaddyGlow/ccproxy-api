from dataclasses import dataclass


@dataclass(frozen=True)
class FormatContext:
    """Format conversion context for handler configuration."""

    source_format: str | None = None
    target_format: str | None = None
    conversion_needed: bool = False
    streaming_mode: str | None = None  # "auto", "force", "never"
