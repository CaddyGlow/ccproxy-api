from pydantic import BaseModel, Field


class DuckDBStorageConfig(BaseModel):
    """Config for the DuckDB storage plugin.

    Notes:
    - By default this plugin mirrors core Observability settings and path.
    - You can override the database path if needed via plugin config.
    """

    enabled: bool = Field(
        default=True,
        description="Enable DuckDB storage plugin",
    )
    database_path: str | None = Field(
        default=None, description="Optional override for DuckDB database path"
    )
    optimize_on_shutdown: bool = Field(
        default=False,
        description="Run PRAGMA optimize on shutdown (file-backed DB only)",
    )
