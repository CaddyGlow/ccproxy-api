from contextlib import contextmanager
from pathlib import Path


# Fix for typing.TypedDict not supported
# pydantic.errors.PydanticUserError:
# Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.
# For further information visit https://errors.pydantic.dev/2.11/u/typed-dict-version
@contextmanager
def patched_typing():
    import typing

    import typing_extensions

    original = typing.TypedDict
    typing.TypedDict = typing_extensions.TypedDict
    try:
        yield
    finally:
        typing.TypedDict = original


def get_package_dir() -> Path:
    try:
        import importlib.util

        # Get the path to the claude_code_proxy package and resolve it
        spec = importlib.util.find_spec("claude_code_proxy")
        if spec and spec.origin:
            package_dir = Path(spec.origin).parent.parent.resolve()
        else:
            package_dir = Path(__file__).parent.parent.parent.resolve()
    except Exception:
        package_dir = Path(__file__).parent.parent.parent.resolve()

    return package_dir
