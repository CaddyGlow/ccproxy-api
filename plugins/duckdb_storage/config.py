from pydantic import BaseModel, Field


class DuckDBStorageConfig(BaseModel):
    """Config for the DuckDB storage plugin.

    Notes:
    - By default this plugin mirrors core Observability settings and path.
    - You can override the database path if needed via plugin config.
    """

    enabled: bool = Field(
        default=True,
        description="Enable DuckDB storage plugin (mirrors observability needs)",
    )
    use_observability_settings: bool = Field(
        default=True,
        description="Use Settings.observability.* to control enablement and path",
    )
    database_path: str | None = Field(
        default=None, description="Optional override for DuckDB database path"
    )
    register_app_state_alias: bool = Field(
        default=True,
        description="Also set app.state.duckdb_storage for backward compatibility",
    )
