"""Base utilities for plugin authors (PEP 420 migration).

This package consolidates base helpers previously under `plugins/base`.
Most core adapter and middleware primitives now live in:

- `ccproxy.services.adapters.base` (BaseAdapter)
- `ccproxy.core.plugins.middleware` (middleware specs/manager)

This module intentionally exports no additional symbols.
"""

__all__: list[str] = []
