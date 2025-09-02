from __future__ import annotations

from ccproxy.plugins.request_tracer.config import RequestTracerConfig


def test_request_tracer_dirs_defaults_and_overrides() -> None:
    cfg = RequestTracerConfig()
    assert cfg.get_json_log_dir() == cfg.log_dir
    assert cfg.get_raw_log_dir() == cfg.log_dir

    cfg2 = RequestTracerConfig(request_log_dir="/tmp/json", raw_log_dir="/tmp/raw")
    assert cfg2.get_json_log_dir() == "/tmp/json"
    assert cfg2.get_raw_log_dir() == "/tmp/raw"


def test_request_tracer_path_filters() -> None:
    cfg = RequestTracerConfig(exclude_paths=["/health", "/metrics"])  # default-like
    assert not cfg.should_trace_path("/health")
    assert not cfg.should_trace_path("/metrics")
    assert cfg.should_trace_path("/api/v1/messages")

    cfg_only = RequestTracerConfig(include_paths=["/api"])  # include restricts
    assert cfg_only.should_trace_path("/api/v1/messages")
    assert not cfg_only.should_trace_path("/other")
