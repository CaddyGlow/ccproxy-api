"""Coverage for helper utilities in the config CLI command."""

from pydantic import BaseModel, Field

from ccproxy.cli.commands.config import commands as config_cmd


class NestedConfig(BaseModel):
    value: int = Field(1, description="Nested value")


class SampleConfig(BaseModel):
    server_host: str = "127.0.0.1"
    security_token: str = "api_token_value"
    cors_origins: list[str] = []
    nested: NestedConfig = NestedConfig()


def test_format_value_handles_various_types() -> None:
    assert config_cmd._format_value(None) == "[dim]Auto-detect[/dim]"
    assert config_cmd._format_value(True) == "True"
    assert config_cmd._format_value("api_token_value") == "[green]Set[/green]"
    assert config_cmd._format_value([]) == "[dim]None[/dim]"


def test_get_field_description_falls_back() -> None:
    field_info = SampleConfig.model_fields["server_host"]

    assert config_cmd._get_field_description(field_info) == "Configuration setting"


def test_generate_rows_and_grouping() -> None:
    model = SampleConfig(cors_origins=["https://example.com"])

    rows = config_cmd._generate_config_rows_from_model(model)
    grouped = config_cmd._group_config_rows(rows)

    assert grouped["Server Configuration"][0][0] == "host"
    # secret string should report as set and be grouped under security
    security_entries = grouped["Security Configuration"]
    assert any(
        name == "token" and value == "[green]Set[/green]"
        for name, value, _ in security_entries
    )

    general_entries = grouped["General Configuration"]
    assert any(name == "nested" for name, _, _ in general_entries)
    assert any(name == "value" for name, _, _ in general_entries)
