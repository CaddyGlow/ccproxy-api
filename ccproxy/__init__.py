import sys
import typing


# Apply TypedDict patch for Python < 3.12 at the root of the package
if sys.version_info < (3, 12):
    import typing_extensions

    typing.TypedDict = typing_extensions.TypedDict

from ._version import __version__


__all__ = ["__version__"]
