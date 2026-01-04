"""Metrics plugin for CCProxy.

This plugin provides Prometheus metrics collection and export functionality
using the hook system for event-driven metric updates.
"""

from .plugin import factory


__all__ = ["factory"]
