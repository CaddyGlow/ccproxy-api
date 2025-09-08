"""Runtime configuration settings - binary resolution configuration."""

from pydantic import BaseModel, Field, field_validator


# === Binary Resolution Configuration ===


class BinarySettings(BaseModel):
    """Binary resolution and package manager fallback settings."""

    fallback_enabled: bool = Field(
        default=True,
        description="Enable package manager fallback when binaries are not found",
    )

    package_manager_only: bool = Field(
        default=True,
        description="Skip direct binary lookup and use package managers exclusively",
    )

    preferred_package_manager: str | None = Field(
        default=None,
        description="Preferred package manager (bunx, pnpm, npx). If not set, auto-detects based on availability",
    )

    package_manager_priority: list[str] = Field(
        default_factory=lambda: ["bunx", "pnpm", "npx"],
        description="Priority order for trying package managers when preferred is not set",
    )

    cache_results: bool = Field(
        default=True,
        description="Cache binary resolution results to avoid repeated lookups",
    )

    @field_validator("preferred_package_manager")
    @classmethod
    def validate_preferred_package_manager(cls, v: str | None) -> str | None:
        """Validate preferred package manager."""
        if v is not None:
            valid_managers = ["bunx", "pnpm", "npx"]
            if v not in valid_managers:
                raise ValueError(
                    f"Invalid package manager: {v}. Must be one of {valid_managers}"
                )
        return v

    @field_validator("package_manager_priority")
    @classmethod
    def validate_package_manager_priority(cls, v: list[str]) -> list[str]:
        """Validate package manager priority list."""
        valid_managers = {"bunx", "pnpm", "npx"}
        for manager in v:
            if manager not in valid_managers:
                raise ValueError(
                    f"Invalid package manager in priority list: {manager}. "
                    f"Must be one of {valid_managers}"
                )
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for manager in v:
            if manager not in seen:
                seen.add(manager)
                result.append(manager)
        return result
