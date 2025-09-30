"""Generic Pydantic model introspection and display utility for settings help."""

from __future__ import annotations

import dataclasses
import enum
import inspect
from typing import Any, get_args, get_origin

from pydantic import BaseModel, SecretStr
from pydantic.fields import FieldInfo
from rich.box import HEAVY_HEAD
from rich.console import Console
from rich.table import Table


console = Console()

ELLIPSIS = "…"


def _is_model(obj: Any) -> bool:
    """Check if an object is a Pydantic BaseModel subclass."""
    return inspect.isclass(obj) and issubclass(obj, BaseModel)


def _typename(tp: Any) -> str:
    """Return human-friendly type name for Unions, Literals, containers, etc."""
    origin = get_origin(tp)
    if origin is None:
        if isinstance(tp, type):
            try:
                if issubclass(tp, enum.Enum):
                    return f"Enum[{tp.__name__}]"
            except TypeError:
                pass
            name_str: str = tp.__name__
            return name_str
        return str(tp)

    args_str = ", ".join(_typename(a) for a in get_args(tp))
    name = getattr(origin, "__name__", str(origin))

    # Handle Union types (including | syntax)
    if name in {"Union", "types.UnionType", "UnionType"}:
        return " | ".join(_typename(a) for a in get_args(tp))

    # Handle common generic types
    if name in {"list", "List"}:
        return f"list[{args_str}]"
    if name in {"dict", "Dict"}:
        return f"dict[{args_str}]"
    if name in {"Annotated"}:
        # Show inner type for Annotated
        inner_args = get_args(tp)
        return _typename(inner_args[0]) if inner_args else "Any"

    result: str = f"{name}[{args_str}]" if args_str else name
    return result


def _is_secret_field(name: str) -> bool:
    """Check if a field name suggests it contains secret data."""
    secret_patterns = ["token", "key", "secret", "password", "credential", "auth"]
    return any(pattern in name.lower() for pattern in secret_patterns)


def _render_value(val: Any, width: int = 22, field_name: str = "") -> str:
    """Render a value for display with truncation and secret masking."""
    # Handle None
    if val is None:
        return "—"

    # Mask secrets
    if isinstance(val, SecretStr):
        return "***"
    if field_name and _is_secret_field(field_name):
        return "***" if val else "—"

    # Handle enums
    if isinstance(val, enum.Enum):
        return str(val.value)

    # Handle Pydantic models - show class name only
    if isinstance(val, BaseModel):
        return val.__class__.__name__

    # Handle dataclasses
    if dataclasses.is_dataclass(val):
        return val.__class__.__name__

    # Convert to string and truncate if needed
    s = repr(val)
    if len(s) > width:
        return s[: width - 1] + ELLIPSIS
    return s


def _default_for_field(field: FieldInfo) -> Any:
    """Extract default value or factory from a Pydantic field."""
    from pydantic_core import PydanticUndefined

    # Check if field has a default value
    if hasattr(field, "default") and field.default is not PydanticUndefined:
        return field.default

    # Check if field has a default_factory
    if hasattr(field, "default_factory") and field.default_factory is not None:
        factory = field.default_factory
        if callable(factory):
            try:
                return factory()  # type: ignore[call-arg]
            except Exception:
                return "<factory>"
        return "<factory>"

    # Field is required
    return "required"


def _choices_from_type(tp: Any) -> list[str]:
    """Extract enum/literal choices from a type annotation."""
    origin = get_origin(tp)

    # Handle direct enum types
    if origin is None:
        try:
            if inspect.isclass(tp) and issubclass(tp, enum.Enum):
                return [str(m.value) for m in tp]
        except TypeError:
            pass
        return []

    # Handle Literal types
    if hasattr(origin, "__name__") and origin.__name__ in {"Literal"}:
        return [repr(x) for x in get_args(tp)]

    return []


def _is_field_hidden(field: FieldInfo) -> bool:
    """Check if a field should be hidden from display.

    Fields can be hidden by setting:
    - field.exclude = True
    - field.json_schema_extra = {"config_example_hidden": True}
    """
    # Check if field is explicitly excluded
    if field.exclude:
        return True

    # Check json_schema_extra for config_example_hidden flag
    json_schema_extra = field.json_schema_extra
    if json_schema_extra:
        # Handle both dict and callable forms
        if callable(json_schema_extra):
            try:
                # Pydantic v2 callable signature: (schema_dict, handler)
                extra_dict = json_schema_extra({})
                if isinstance(extra_dict, dict):
                    return bool(extra_dict.get("config_example_hidden", False))
            except Exception:
                pass
        elif isinstance(json_schema_extra, dict):
            return bool(json_schema_extra.get("config_example_hidden", False))

    return False


def build_table_for_model(
    model_cls: type[BaseModel],
    instance: BaseModel | None = None,
    *,
    title: str | None = None,
    show_value: bool = True,
) -> Table:
    """Build a Rich table for a Pydantic model with field information.

    Args:
        model_cls: The Pydantic model class to display
        instance: Optional instance to show actual values
        title: Optional table title (defaults to model class name)
        show_value: Whether to include the Value column (useful for schema-only display)

    Returns:
        A Rich Table ready for printing
    """
    table = Table(
        title=title or model_cls.__name__,
        box=HEAVY_HEAD,
        show_lines=False,
        header_style="bold",
    )

    table.add_column("Field", style="bold")
    table.add_column("Type", style="cyan")
    if show_value:
        table.add_column("Value", style="green")
    table.add_column("Default", style="yellow")
    table.add_column("Description", style="dim")

    schema = model_cls.model_json_schema()
    required_fields = set(schema.get("required", []))

    for field_name, field in model_cls.model_fields.items():
        # Skip hidden fields
        if _is_field_hidden(field):
            continue
        prop = schema.get("properties", {}).get(field_name, {})

        # Get type string
        tp_str = (
            _typename(field.annotation)
            if field.annotation is not None
            else prop.get("type", "object")
        )

        # Get default value
        default = _default_for_field(field)

        # Get actual value if instance provided
        if show_value and instance is not None:
            val = getattr(
                instance, field_name, default if default != "required" else None
            )
            value_str = _render_value(val, width=22, field_name=field_name)
        else:
            value_str = None

        # Get description
        desc = prop.get("description", "") or field.description or ""

        # Add choices to description if available
        choices = _choices_from_type(field.annotation)
        if choices and len(choices) <= 5:  # Only show if reasonable number
            choices_str = ", ".join(choices[:5])
            if len(choices) > 5:
                choices_str += "..."
            desc = f"{desc} Choices: {choices_str}".strip()

        # Mark required fields with *
        display_name = f"{field_name}*" if field_name in required_fields else field_name

        # Build row
        if show_value and value_str is not None:
            table.add_row(
                display_name,
                tp_str,
                value_str,
                _render_value(default, width=22),
                desc,
            )
        else:
            table.add_row(
                display_name,
                tp_str,
                _render_value(default, width=22),
                desc,
            )

    return table


def collect_nested_models(model_cls: type[BaseModel]) -> list[type[BaseModel]]:
    """Recursively find all nested BaseModel types in a model's fields.

    Args:
        model_cls: The Pydantic model class to scan

    Returns:
        List of unique BaseModel subclasses found, sorted by name
    """
    nested_models: set[type[BaseModel]] = set()

    def walk(tp: Any) -> None:
        """Recursively walk type annotations to find BaseModel subclasses."""
        origin = get_origin(tp)

        if origin is None:
            # Direct type - check if it's a BaseModel
            try:
                if inspect.isclass(tp) and issubclass(tp, BaseModel):
                    nested_models.add(tp)
                    # Recursively scan this model's fields
                    for field in tp.model_fields.values():
                        if field.annotation is not None:
                            walk(field.annotation)
            except TypeError:
                pass
            return

        # Generic type - walk the type arguments
        for arg in get_args(tp):
            walk(arg)

    # Scan all fields in the model
    for field in model_cls.model_fields.values():
        if field.annotation is not None:
            walk(field.annotation)

    # Exclude the model itself
    nested_models.discard(model_cls)

    # Return sorted by name for stable output
    return sorted(nested_models, key=lambda c: c.__name__)


def print_settings_help(
    model_cls: type[BaseModel],
    instance: BaseModel | None = None,
    *,
    title_left: str = "",
    version: str | None = None,
    enabled: bool | None = None,
) -> None:
    """Print comprehensive settings help for a Pydantic model.

    Displays:
    1. Main table with all fields (with values if instance provided)
    2. Nested Configuration Types section with schema tables for each nested model

    Args:
        model_cls: The Pydantic model class to display
        instance: Optional instance to show actual values
        title_left: Optional prefix for the title
        version: Optional version string to display
        enabled: Optional enabled status to display
    """
    # Build header
    header = title_left + model_cls.__name__ if title_left else model_cls.__name__
    suffix = []
    if version:
        suffix.append(f"v{version}")
    if enabled is not None:
        suffix.append("enabled" if enabled else "disabled")
    if suffix:
        header += " (" + ", ".join(suffix) + ")"

    # Print main table
    console.print(f"\n{header}", style="bold")
    console.print(
        build_table_for_model(model_cls, instance, show_value=instance is not None)
    )

    # Print nested types
    nested = collect_nested_models(model_cls)
    if nested:
        console.print("\n[bold cyan]Nested Configuration Types:[/bold cyan]\n")
        for nested_cls in nested:
            console.print(build_table_for_model(nested_cls, show_value=False))
            console.print()
