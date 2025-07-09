from collections.abc import Callable, Sequence
from typing import Any

import click
import typer
from typer.core import DEFAULT_MARKUP_MODE, MarkupMode, TyperCommand, TyperGroup
from typer.models import CommandFunctionType, Default


class TyperDefaultCommand(TyperCommand):
    """Type that indicates if a command is the default command."""


class TyperGroupWithDefault(TyperGroup):
    """Use a default command if specified."""

    def __init__(
        self,
        *,
        name: str | None = None,
        commands: dict[str, click.Command] | Sequence[click.Command] | None = None,
        rich_markup_mode: MarkupMode = DEFAULT_MARKUP_MODE,
        rich_help_panel: str | None = None,
        **attrs: Any,
    ) -> None:
        super().__init__(
            name=name,
            commands=commands,
            rich_markup_mode=rich_markup_mode,
            rich_help_panel=rich_help_panel,
            **attrs,
        )
        # find the default command if any
        self.default_command = None
        if len(self.commands) > 1:
            for name, command in reversed(self.commands.items()):  # type: ignore[call-overload]
                if isinstance(command, TyperDefaultCommand):
                    self.default_command = name
                    break

    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        # if --help is specified, show the group help
        # else if default command was specified in the group and no args or no subcommand is specified, use the default command
        if (
            self.default_command
            and (not args or args[0] not in self.commands)
            and "--help" not in args
            and "--show-completion" not in args
            and "--install-completion" not in args
        ):
            args = [self.default_command] + args
        return super().make_context(info_name, args, parent, **extra)


class Typer(typer.Typer):
    """Typer with default command support."""

    def __init__(
        self,
        *,
        name: str | None = Default(None),
        invoke_without_command: bool = Default(False),
        no_args_is_help: bool = Default(False),
        subcommand_metavar: str | None = Default(None),
        chain: bool = Default(False),
        result_callback: Callable[..., Any] | None = Default(None),
        context_settings: dict[Any, Any] | None = Default(None),
        callback: Callable[..., Any] | None = Default(None),
        help: str | None = Default(None),
        epilog: str | None = Default(None),
        short_help: str | None = Default(None),
        options_metavar: str = Default("[OPTIONS]"),
        add_help_option: bool = Default(True),
        hidden: bool = Default(False),
        deprecated: bool = Default(False),
        add_completion: bool = True,
        rich_markup_mode: MarkupMode = Default(DEFAULT_MARKUP_MODE),
        rich_help_panel: str | None = Default(None),
        pretty_exceptions_enable: bool = True,
        pretty_exceptions_show_locals: bool = True,
        pretty_exceptions_short: bool = True,
    ):
        super().__init__(
            name=name,
            cls=TyperGroupWithDefault,
            invoke_without_command=invoke_without_command,
            no_args_is_help=no_args_is_help,
            subcommand_metavar=subcommand_metavar,
            chain=chain,
            result_callback=result_callback,
            context_settings=context_settings,
            callback=callback,
            help=help,
            epilog=epilog,
            short_help=short_help,
            options_metavar=options_metavar,
            add_help_option=add_help_option,
            hidden=hidden,
            deprecated=deprecated,
            add_completion=add_completion,
            rich_markup_mode=rich_markup_mode,
            rich_help_panel=rich_help_panel,
            pretty_exceptions_enable=pretty_exceptions_enable,
            pretty_exceptions_show_locals=pretty_exceptions_show_locals,
            pretty_exceptions_short=pretty_exceptions_short,
        )

    def command(
        self,
        name: str | None = None,
        *,
        cls: type[TyperCommand] | None = None,
        context_settings: dict[Any, Any] | None = None,
        help: str | None = None,
        epilog: str | None = None,
        short_help: str | None = None,
        options_metavar: str = "[OPTIONS]",
        add_help_option: bool = True,
        no_args_is_help: bool = False,
        hidden: bool = False,
        deprecated: bool = False,
        rich_help_panel: str | None = Default(None),
        default: bool = False,
    ) -> Callable[[CommandFunctionType], CommandFunctionType]:
        return super().command(
            name,
            cls=TyperDefaultCommand if default else cls,
            context_settings=context_settings,
            help=help,
            epilog=epilog,
            short_help=short_help,
            options_metavar=options_metavar,
            add_help_option=add_help_option,
            no_args_is_help=no_args_is_help,
            hidden=hidden,
            deprecated=deprecated,
            rich_help_panel=rich_help_panel,
        )


# use it instantiating Typer and some commands
# app = Typer()
#
#
# @app.command()
# def some_command():
#     click.echo("This command is not the default command")
#
#
# @app.command(default=True)
# def default_command():
#     click.echo("This command is the default command")
