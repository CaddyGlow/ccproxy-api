"""Namespace package for in-repo plugins used in tests.

This file ensures mypy maps modules under this directory as
`plugins.<name>` rather than treating subfolders as top-level packages
like `metrics.*`, avoiding duplicate module mapping errors.
"""

