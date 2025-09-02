"""ccproxy.plugins dynamic namespace bridge.

This package enables dynamic discovery of built-in plugins that live in the
top-level `plugins/` namespace by mapping imports like
`ccproxy.plugins.foo` -> `plugins.foo` at import time.

It allows PEP 420-style namespace behavior without requiring each plugin to be
physically located under `ccproxy/plugins/`.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType


class _CcproxyPluginsRedirectFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    prefix = "ccproxy.plugins."

    # MetaPathFinder API
    def find_spec(
        self, fullname: str, path: object | None, target: object | None = None
    ):  # type: ignore[override]
        if not fullname.startswith(self.prefix) or fullname == self.prefix[:-1]:
            return None

        plugin_name = fullname[len(self.prefix) :]
        target_name = f"plugins.{plugin_name}"

        # If target already imported, just alias
        if target_name in sys.modules:
            return importlib.util.spec_from_loader(fullname, self)

        # Try to locate the target module in top-level `plugins` package
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None

        return importlib.util.spec_from_loader(fullname, self)

    # Loader API
    def create_module(self, spec):  # type: ignore[override]
        # Defer to exec_module
        return None

    def exec_module(self, module: ModuleType) -> None:  # type: ignore[override]
        plugin_name = module.__name__[len(self.prefix) :]
        target_name = f"plugins.{plugin_name}"

        # Import the target module
        target_mod = importlib.import_module(target_name)

        # Alias target module under ccproxy.plugins.<name>
        sys.modules[module.__name__] = target_mod

        # Expose submodule on parent package for attribute access
        parent_name = module.__name__.rsplit(".", 1)[0]
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, plugin_name.split(".")[0], target_mod)


# Install the finder once
_installed = any(isinstance(f, _CcproxyPluginsRedirectFinder) for f in sys.meta_path)
if not _installed:
    sys.meta_path.insert(0, _CcproxyPluginsRedirectFinder())
