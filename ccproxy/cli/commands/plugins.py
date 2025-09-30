"""CLI commands for interacting with plugins."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table

from ccproxy.config.settings import Settings
from ccproxy.core.plugins.discovery import (
    PluginDiscovery,
    PluginFilter,
    build_combined_plugin_denylist,
)
from ccproxy.core.plugins.interfaces import PluginFactory
from ccproxy.templates import PluginTemplateType, build_plugin_scaffold


app = typer.Typer(
    name="plugins", help="Manage and inspect plugins.", no_args_is_help=True
)


PLUGIN_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class PluginConfigField:
    """Renderable representation of a plugin configuration field."""

    name: str
    type_label: str
    default_label: str
    value_label: str
    description: str
    required: bool


@dataclass(frozen=True)
class PluginMetadata:
    """Aggregated metadata and configuration for a plugin."""

    name: str
    version: str | None
    description: str | None
    enabled: bool
    status_reason: str | None
    config_fields: tuple[PluginConfigField, ...]


def _format_annotation(annotation: Any) -> str:
    """Return a human-readable label for a type annotation."""

    if annotation is None:
        return "Any"
    module = getattr(annotation, "__module__", "")
    if module == "typing":
        return str(annotation).removeprefix("typing.")
    if hasattr(annotation, "__name__"):
        return str(annotation.__name__)
    return str(annotation)


def _format_default(field: Any) -> str:
    """Render default value or factory for display."""

    default_factory = getattr(field, "default_factory", None)
    if default_factory is not None:
        factory_name = getattr(default_factory, "__name__", repr(default_factory))
        return f"<factory:{factory_name}>"

    if field.is_required():
        return "required"

    default_value = getattr(field, "default", None)
    if isinstance(default_value, str):
        return f'"{default_value}"'
    return repr(default_value)


def _format_value(value: Any) -> str:
    """Render an actual configuration value for display."""

    if value is None:
        return "—"
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def describe_config_model(
    config_class: type[BaseModel] | None,
    config_instance: BaseModel | None = None,
) -> tuple[PluginConfigField, ...]:
    """Convert a plugin config model into display-ready field metadata."""

    if config_class is None:
        return ()

    fields_info: list[PluginConfigField] = []
    for field_name, field in config_class.model_fields.items():
        type_label = _format_annotation(field.annotation)
        default_label = _format_default(field)
        description = field.description or ""
        required = field.is_required()
        value_label = "—"

        if config_instance is not None:
            value = getattr(config_instance, field_name, None)
            value_label = _format_value(value)

        fields_info.append(
            PluginConfigField(
                name=field_name,
                type_label=type_label,
                default_label=default_label,
                value_label=value_label,
                description=description,
                required=required,
            )
        )

    return tuple(fields_info)


def _load_all_plugin_factories(
    settings: Settings,
) -> tuple[dict[str, PluginFactory], PluginFilter, set[str]]:
    """Load plugin factories without applying filters for inspection."""

    plugin_dirs = [Path(path) for path in settings.plugin_discovery.directories]
    discovery = PluginDiscovery(plugin_dirs)

    combined_denylist = build_combined_plugin_denylist(
        getattr(settings, "disabled_plugins", None),
        getattr(settings, "plugins", None),
    )
    filter_config = PluginFilter(
        enabled_plugins=getattr(settings, "enabled_plugins", None),
        disabled_plugins=combined_denylist,
    )

    factories = discovery.load_entry_point_factories(plugin_filter=None)

    if not settings.plugins_disable_local_discovery:
        discovery.discover_plugins()
        filesystem_factories = discovery.load_all_factories(plugin_filter=None)
        for name, factory in filesystem_factories.items():
            factories[name] = factory

    return factories, filter_config, combined_denylist


def _build_config_instance(
    manifest: Any,
    settings: Settings,
) -> BaseModel | None:
    """Instantiate the plugin config using current settings."""

    config_class = getattr(manifest, "config_class", None)
    if config_class is None:
        return None

    config_data = settings.plugins.get(manifest.name)
    try:
        if config_data is None:
            return config_class()  # type: ignore[no-any-return]
        return config_class.model_validate(config_data)  # type: ignore[no-any-return]
    except ValidationError:
        # Fall back to defaults to avoid breaking the CLI view
        try:
            return cast(BaseModel, config_class())
        except ValidationError:
            return None


def _derive_status_reason(
    name: str,
    settings: Settings,
    combined_denylist: set[str],
) -> str | None:
    """Determine why a plugin is disabled, if applicable."""

    if name in combined_denylist:
        return "disabled via config"
    if settings.enabled_plugins is not None and name not in set(
        settings.enabled_plugins
    ):
        return "disabled via not allow-listed"
    if not settings.enable_plugins:
        return "disabled via plugin system disabled"
    return None


def _select_scaffold_root(settings: Settings) -> Path:
    """Choose a sensible default root for new plugin scaffolds."""

    directories = settings.plugin_discovery.directories
    if not directories:
        return Path.cwd()
    for candidate in reversed(directories):
        candidate_path = Path(candidate)
        parts = candidate_path.parts
        if len(parts) >= 2 and parts[-2:] == ("ccproxy", "plugins"):
            continue
        return candidate_path

    return Path(directories[-1])


def gather_plugin_metadata(settings: Settings) -> tuple[PluginMetadata, ...]:
    """Collect plugin metadata and configuration for CLI display."""

    factories, filter_config, combined_denylist = _load_all_plugin_factories(settings)

    metadata_list: list[PluginMetadata] = []
    for name in sorted(factories):
        factory = factories[name]
        manifest = factory.get_manifest()
        config_instance = _build_config_instance(manifest, settings)
        config_fields = describe_config_model(manifest.config_class, config_instance)
        enabled = settings.enable_plugins and filter_config.is_enabled(name)
        status_reason = (
            None
            if enabled
            else _derive_status_reason(name, settings, combined_denylist)
        )

        metadata_list.append(
            PluginMetadata(
                name=name,
                version=getattr(manifest, "version", None),
                description=getattr(manifest, "description", None),
                enabled=enabled,
                status_reason=status_reason,
                config_fields=config_fields,
            )
        )

    return tuple(metadata_list)


@app.command(name="list")
def list_plugins() -> None:
    """List all discovered plugins and high-level details."""

    console = Console()
    settings_obj = Settings.from_config()

    plugins = gather_plugin_metadata(settings_obj)
    if not plugins:
        console.print("No plugins found.")
        return

    table = Table(
        title="Discovered Plugins",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Plugin", style="bold")
    table.add_column("Version", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Config Fields", style="yellow")
    table.add_column("Description", style="dim")

    for plugin in plugins:
        status = "Enabled" if plugin.enabled else "Disabled"
        if plugin.status_reason:
            status = f"{status} ({plugin.status_reason})"
        config_count = str(len(plugin.config_fields)) if plugin.config_fields else "0"
        table.add_row(
            plugin.name,
            plugin.version or "unknown",
            status,
            config_count,
            plugin.description or "",
        )

    console.print(table)


@app.command()
def settings(
    plugin: str | None = typer.Argument(None, help="Plugin to inspect"),
) -> None:
    """Show configuration fields for plugins."""

    console = Console()
    settings_obj = Settings.from_config()

    plugins = gather_plugin_metadata(settings_obj)
    if not plugins:
        console.print("No plugins found.")
        return

    if plugin is not None:
        plugins = tuple(p for p in plugins if p.name == plugin)
        if not plugins:
            console.print(f"Plugin '{plugin}' not found.")
            return

    for plugin_meta in plugins:
        header = f"[bold]{plugin_meta.name}[/bold]"
        version = plugin_meta.version or "unknown"
        status = "enabled" if plugin_meta.enabled else "disabled"
        if plugin_meta.status_reason:
            status = f"{plugin_meta.status_reason}"
        console.print(f"\n{header} (v{version}, {status})")

        if not plugin_meta.config_fields:
            console.print("  No configuration fields declared.")
            continue

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Field", style="bold")
        table.add_column("Type", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Default", style="yellow")
        table.add_column("Description", style="dim")

        for field in plugin_meta.config_fields:
            field_label = f"{field.name}*" if field.required else field.name
            table.add_row(
                field_label,
                field.type_label,
                field.value_label,
                field.default_label,
                field.description,
            )

        console.print(table)


@app.command()
def dependencies() -> None:
    """Display how plugin dependencies are managed."""

    console = Console()
    console.print(
        "Plugin dependencies are managed at the package level (pyproject.toml/extras)."
    )


@app.command()
def scaffold(
    plugin_name: Annotated[
        str,
        typer.Argument(
            help="New plugin package name (snake_case).",
        ),
    ],
    plugin_type: Annotated[
        PluginTemplateType,
        typer.Option(
            "--type",
            "-t",
            help="Scaffold type to generate (system, provider, auth).",
            case_sensitive=False,
        ),
    ] = PluginTemplateType.SYSTEM,
    description: Annotated[
        str,
        typer.Option(
            "--description",
            "-d",
            help="Plugin description stored in the manifest.",
        ),
    ] = "Custom CCProxy plugin.",
    version: Annotated[
        str,
        typer.Option(
            "--version",
            "-v",
            help="Semver version recorded in the manifest.",
        ),
    ] = "0.1.0",
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            "-p",
            help="Directory to create the plugin in (defaults to user plugin dir).",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = None,
    include_tests: Annotated[
        bool,
        typer.Option(
            "--with-tests/--no-tests",
            help="Include placeholder pytest files in the scaffold.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Overwrite existing files when the directory already exists.",
        ),
    ] = False,
) -> None:
    """Generate a plugin scaffold to jump-start development."""

    console = Console()
    settings_obj = Settings.from_config()
    raw_name = plugin_name.strip()
    normalised = raw_name.lower()

    if not PLUGIN_NAME_PATTERN.match(normalised):
        raise typer.BadParameter(
            "Plugin name must start with a letter and use lowercase, digits, or underscores.",
            param_hint="plugin_name",
        )

    plugin_name = normalised

    if output_path is None:
        target_root = _select_scaffold_root(settings_obj)
    else:
        target_root = output_path

    target_root = target_root.expanduser()
    target_root.mkdir(parents=True, exist_ok=True)

    target_dir = target_root / plugin_name
    if target_dir.exists():
        has_content = any(target_dir.iterdir())
        if has_content and not force:
            console.print(
                f"[red]Directory {target_dir} already exists. Use --force to overwrite.[/red]"
            )
            raise typer.Exit(code=1)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = build_plugin_scaffold(
            plugin_name=plugin_name,
            description=description,
            version=version,
            template_type=plugin_type,
            include_tests=include_tests,
        )
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[red]Failed to build scaffold: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    created: list[tuple[str, str]] = []
    for relative_path, content in files.items():
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        action = "overwrote" if destination.exists() else "created"
        destination.write_text(content, encoding="utf-8")
        created.append((action, relative_path))

    console.print(
        f"[bold green]Plugin scaffold ready[/bold green] in [cyan]{target_dir}[/cyan]"
    )
    if raw_name != plugin_name:
        console.print(
            f"  • Normalised plugin name to [bold]{plugin_name}[/bold] from '{raw_name}'."
        )
    for action, relative_path in created:
        console.print(f"  • {action}: {relative_path}")
    console.print(
        "  • Update config and runtime files before enabling the plugin.",
        style="dim",
    )
    if settings_obj.plugins_disable_local_discovery:
        console.print(
            "  • Local plugin discovery is disabled. Set `plugins_disable_local_discovery = false`"
            " in your config or export `PLUGINS_DISABLE_LOCAL_DISCOVERY=false` to load filesystem"
            " plugins.",
            style="yellow",
        )
    if not settings_obj.enable_plugins:
        console.print(
            "  • Plugin system is disabled (`enable_plugins = false`). Update configuration to"
            " load plugins.",
            style="yellow",
        )
