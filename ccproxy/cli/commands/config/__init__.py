"""Config command module for Claude Code Proxy API."""

from ccproxy.cli.commands.config.commands import app, config_list
from ccproxy.cli.commands.config.schema_commands import (
    config_schema,
    config_validate,
)


# Register schema commands with the app
app.command(name="schema")(config_schema)
app.command(name="validate")(config_validate)

__all__ = ["app", "config_list"]
