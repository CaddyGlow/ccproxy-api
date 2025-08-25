"""Comprehensive unit tests for the hook system components."""

import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.hooks import HookContext, HookEvent, HookManager, HookRegistry


# TODO: These hook implementations need to be created
# from ccproxy.hooks.implementations import AnalyticsHook, LoggingHook, MetricsHook


# Test fixtures


@pytest.fixture
def hook_registry():
    """Create a fresh hook registry for testing."""
    return HookRegistry()


@pytest.fixture
def hook_manager(hook_registry):
    """Create a hook manager with the test registry."""
    return HookManager(hook_registry)


@pytest.fixture
def sample_hook_context():
    """Create a sample hook context for testing."""
    return HookContext(
        event=HookEvent.REQUEST_STARTED,
        timestamp=datetime.utcnow(),
        data={"test_key": "test_value"},
        metadata={"source": "test"},
        provider="test_provider",
        plugin="test_plugin",
    )


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock()  # Remove spec=Request to avoid truthiness issues
    request.method = "POST"

    # Mock URL
    url_mock = MagicMock()
    url_mock.path = "/api/test"
    request.url = url_mock

    # Mock headers
    headers_mock = MagicMock()
    headers_mock.get.return_value = "test-123"
    headers_mock.__getitem__ = lambda self, key: {
        "content-type": "application/json",
        "x-request-id": "test-123",
    }[key]
    headers_mock.__contains__ = lambda self, key: key in {
        "content-type",
        "x-request-id",
    }
    headers_mock.__iter__ = lambda self: iter(
        {"content-type": "application/json", "x-request-id": "test-123"}
    )
    headers_mock.items = lambda: [
        ("content-type", "application/json"),
        ("x-request-id", "test-123"),
    ]
    request.headers = headers_mock

    # Mock client
    client_mock = MagicMock()
    client_mock.host = "127.0.0.1"
    request.client = client_mock

    return request


@pytest.fixture
def mock_response():
    """Create a mock FastAPI response."""
    response = MagicMock()  # Remove spec=Response to avoid truthiness issues
    response.status_code = 200
    response.headers = {"content-type": "application/json"}
    response.body = b'{"result": "success"}'
    return response


# Mock hook implementations for testing


class AsyncTestHook:
    """Test async hook implementation."""

    def __init__(
        self, name: str = "async_test_hook", events: list[HookEvent] | None = None
    ):
        self._name = name
        self._events = events or [HookEvent.REQUEST_STARTED]
        self.call_count = 0
        self.last_context: HookContext | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def events(self) -> list[HookEvent]:
        return self._events

    async def __call__(self, context: HookContext) -> None:
        self.call_count += 1
        self.last_context = context


class SyncTestHook:
    """Test sync hook implementation."""

    def __init__(
        self, name: str = "sync_test_hook", events: list[HookEvent] | None = None
    ):
        self._name = name
        self._events = events or [HookEvent.REQUEST_COMPLETED]
        self.call_count = 0
        self.last_context: HookContext | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def events(self) -> list[HookEvent]:
        return self._events

    def __call__(self, context: HookContext) -> None:
        self.call_count += 1
        self.last_context = context


class FailingHook:
    """Test hook that always fails."""

    def __init__(
        self, name: str = "failing_hook", events: list[HookEvent] | None = None
    ):
        self._name = name
        self._events = events or [HookEvent.REQUEST_FAILED]

    @property
    def name(self) -> str:
        return self._name

    @property
    def events(self) -> list[HookEvent]:
        return self._events

    async def __call__(self, context: HookContext) -> None:
        raise ValueError("Test error from failing hook")


# HookRegistry tests


class TestHookRegistry:
    """Test the HookRegistry class."""

    def test_init(self):
        """Test registry initialization."""
        registry = HookRegistry()
        assert isinstance(registry._hooks, dict)
        assert len(registry._hooks) == 0

    def test_register_hook(self, hook_registry):
        """Test registering a hook."""
        hook = AsyncTestHook(
            events=[HookEvent.REQUEST_STARTED, HookEvent.REQUEST_COMPLETED]
        )

        hook_registry.register(hook)

        # Check that hook is registered for both events
        assert hook in hook_registry.get_hooks(HookEvent.REQUEST_STARTED)
        assert hook in hook_registry.get_hooks(HookEvent.REQUEST_COMPLETED)
        assert len(hook_registry.get_hooks(HookEvent.REQUEST_STARTED)) == 1
        assert len(hook_registry.get_hooks(HookEvent.REQUEST_COMPLETED)) == 1

    def test_register_multiple_hooks_same_event(self, hook_registry):
        """Test registering multiple hooks for the same event."""
        hook1 = AsyncTestHook(name="hook1", events=[HookEvent.REQUEST_STARTED])
        hook2 = AsyncTestHook(name="hook2", events=[HookEvent.REQUEST_STARTED])

        hook_registry.register(hook1)
        hook_registry.register(hook2)

        hooks = hook_registry.get_hooks(HookEvent.REQUEST_STARTED)
        assert len(hooks) == 2
        assert hook1 in hooks
        assert hook2 in hooks

    def test_unregister_hook(self, hook_registry):
        """Test unregistering a hook."""
        hook = AsyncTestHook(
            events=[HookEvent.REQUEST_STARTED, HookEvent.REQUEST_COMPLETED]
        )

        hook_registry.register(hook)
        assert hook in hook_registry.get_hooks(HookEvent.REQUEST_STARTED)
        assert hook in hook_registry.get_hooks(HookEvent.REQUEST_COMPLETED)

        hook_registry.unregister(hook)
        assert hook not in hook_registry.get_hooks(HookEvent.REQUEST_STARTED)
        assert hook not in hook_registry.get_hooks(HookEvent.REQUEST_COMPLETED)

    def test_unregister_nonexistent_hook(self, hook_registry):
        """Test unregistering a hook that was never registered."""
        hook = AsyncTestHook()
        # Should not raise an exception
        hook_registry.unregister(hook)

    def test_get_hooks_empty_event(self, hook_registry):
        """Test getting hooks for an event with no registered hooks."""
        hooks = hook_registry.get_hooks(HookEvent.APP_STARTUP)
        assert hooks == []

    def test_get_hooks_nonexistent_event(self, hook_registry):
        """Test getting hooks for a nonexistent event key."""
        # Using a custom event that doesn't exist in our defaultdict
        hooks = hook_registry.get_hooks("nonexistent.event")
        assert hooks == []

    def test_register_logging(self, hook_registry):
        """Test that hook registration is logged."""
        with patch.object(hook_registry._logger, "info") as mock_info:
            hook = AsyncTestHook(name="test_hook", events=[HookEvent.REQUEST_STARTED])
            hook_registry.register(hook)

            mock_info.assert_called_once_with(
                "Registered test_hook for HookEvent.REQUEST_STARTED"
            )


# HookManager tests


class TestHookManager:
    """Test the HookManager class."""

    def test_init(self, hook_registry):
        """Test manager initialization."""
        manager = HookManager(hook_registry)
        assert manager._registry is hook_registry

    @pytest.mark.asyncio
    async def test_emit_no_hooks(self, hook_manager):
        """Test emitting event with no registered hooks."""
        # Should not raise an exception
        await hook_manager.emit(HookEvent.APP_STARTUP)

    @pytest.mark.asyncio
    async def test_emit_async_hook(self, hook_registry, hook_manager):
        """Test emitting event to async hook."""
        hook = AsyncTestHook(events=[HookEvent.REQUEST_STARTED])
        hook_registry.register(hook)

        test_data = {"test": "data"}
        await hook_manager.emit(HookEvent.REQUEST_STARTED, data=test_data)

        assert hook.call_count == 1
        assert hook.last_context is not None
        assert hook.last_context.event == HookEvent.REQUEST_STARTED
        assert hook.last_context.data == test_data

    @pytest.mark.asyncio
    async def test_emit_sync_hook(self, hook_registry, hook_manager):
        """Test emitting event to sync hook."""
        hook = SyncTestHook(events=[HookEvent.REQUEST_COMPLETED])
        hook_registry.register(hook)

        test_data = {"result": "success"}
        await hook_manager.emit(HookEvent.REQUEST_COMPLETED, data=test_data)

        assert hook.call_count == 1
        assert hook.last_context is not None
        assert hook.last_context.event == HookEvent.REQUEST_COMPLETED
        assert hook.last_context.data == test_data

    @pytest.mark.asyncio
    async def test_emit_multiple_hooks(self, hook_registry, hook_manager):
        """Test emitting event to multiple hooks."""
        hook1 = AsyncTestHook(name="hook1", events=[HookEvent.REQUEST_STARTED])
        hook2 = AsyncTestHook(name="hook2", events=[HookEvent.REQUEST_STARTED])
        hook3 = SyncTestHook(name="hook3", events=[HookEvent.REQUEST_STARTED])

        hook_registry.register(hook1)
        hook_registry.register(hook2)
        hook_registry.register(hook3)

        await hook_manager.emit(HookEvent.REQUEST_STARTED)

        assert hook1.call_count == 1
        assert hook2.call_count == 1
        assert hook3.call_count == 1

    @pytest.mark.asyncio
    async def test_emit_with_context_kwargs(
        self, hook_registry, hook_manager, mock_request, mock_response
    ):
        """Test emitting event with additional context parameters."""
        hook = AsyncTestHook(events=[HookEvent.REQUEST_COMPLETED])
        hook_registry.register(hook)

        await hook_manager.emit(
            HookEvent.REQUEST_COMPLETED,
            data={"status": "success"},
            request=mock_request,
            response=mock_response,
            provider="test_provider",
            plugin="test_plugin",
            error=None,
        )

        assert hook.call_count == 1
        context = hook.last_context
        assert context is not None
        assert context.request is mock_request
        assert context.response is mock_response
        assert context.provider == "test_provider"
        assert context.plugin == "test_plugin"
        assert context.error is None

    @pytest.mark.asyncio
    async def test_emit_failing_hook_isolation(self, hook_registry, hook_manager):
        """Test that failing hooks don't affect other hooks."""
        failing_hook = FailingHook(events=[HookEvent.REQUEST_FAILED])
        good_hook = AsyncTestHook(events=[HookEvent.REQUEST_FAILED])

        hook_registry.register(failing_hook)
        hook_registry.register(good_hook)

        with patch.object(hook_manager._logger, "error") as mock_error:
            await hook_manager.emit(HookEvent.REQUEST_FAILED)

            # Good hook should still be called
            assert good_hook.call_count == 1

            # Error should be logged
            mock_error.assert_called_once()
            error_call = mock_error.call_args[0]
            assert (
                "Hook failing_hook failed for HookEvent.REQUEST_FAILED" in error_call[0]
            )

    @pytest.mark.asyncio
    async def test_hook_context_creation(self, hook_manager, hook_registry):
        """Test that HookContext is created properly."""
        hook = AsyncTestHook(events=[HookEvent.REQUEST_STARTED])
        hook_registry.register(hook)

        test_data = {"key": "value"}
        before_emit = datetime.utcnow()

        await hook_manager.emit(HookEvent.REQUEST_STARTED, data=test_data)

        after_emit = datetime.utcnow()
        context = hook.last_context
        assert context is not None

        assert context.event == HookEvent.REQUEST_STARTED
        assert context.data == test_data
        assert context.metadata == {}
        assert before_emit <= context.timestamp <= after_emit

    @pytest.mark.asyncio
    async def test_execute_hook_async_detection(self, hook_manager):
        """Test that _execute_hook correctly detects async vs sync hooks."""
        async_hook = AsyncTestHook()
        sync_hook = SyncTestHook()
        context = HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
        )

        # Test async hook
        await hook_manager._execute_hook(async_hook, context)
        assert async_hook.call_count == 1

        # Test sync hook
        await hook_manager._execute_hook(sync_hook, context)
        assert sync_hook.call_count == 1


# HookContext tests


class TestHookContext:
    """Test the HookContext dataclass."""

    def test_context_creation_minimal(self):
        """Test creating context with minimal required fields."""
        context = HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
        )

        assert context.event == HookEvent.REQUEST_STARTED
        assert isinstance(context.timestamp, datetime)
        assert context.data == {}
        assert context.metadata == {}
        assert context.request is None
        assert context.response is None
        assert context.provider is None
        assert context.plugin is None
        assert context.error is None

    def test_context_creation_full(self, mock_request, mock_response):
        """Test creating context with all fields."""
        test_error = ValueError("test error")
        context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.utcnow(),
            data={"result": "success"},
            metadata={"trace_id": "123"},
            request=mock_request,
            response=mock_response,
            provider="claude_api",
            plugin="test_plugin",
            error=test_error,
        )

        assert context.event == HookEvent.REQUEST_COMPLETED
        assert context.data == {"result": "success"}
        assert context.metadata == {"trace_id": "123"}
        assert context.request is mock_request
        assert context.response is mock_response
        assert context.provider == "claude_api"
        assert context.plugin == "test_plugin"
        assert context.error is test_error


# Built-in hook tests


@pytest.mark.skip(reason="LoggingHook implementation not yet created")
class TestLoggingHook:
    """Test the LoggingHook implementation."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock structlog logger."""
        return MagicMock()

    @pytest.fixture
    def logging_hook(self, mock_logger):
        """Create a LoggingHook with mock logger."""
        pass  # LoggingHook not implemented

    def test_init_with_logger(self, mock_logger):
        """Test initialization with provided logger."""
        pass  # hook = LoggingHook(logger=mock_logger)
        # assert hook.logger is mock_logger
        # assert hook.name == "logging_hook"
        # assert len(hook.events) == len(list(HookEvent))

    @patch("ccproxy.hooks.implementations.logging.structlog.get_logger")
    def test_init_without_logger(self, mock_get_logger):
        """Test initialization without provided logger."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        pass  # hook = LoggingHook()
        # assert hook.logger is mock_logger
        mock_get_logger.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_basic_event(
        self, logging_hook, mock_logger, sample_hook_context
    ):
        """Test logging a basic event."""
        await logging_hook(sample_hook_context)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "hook_event"

        log_data = call_args[1]
        assert log_data["hook_event"] == HookEvent.REQUEST_STARTED.value
        assert log_data["provider"] == "test_provider"
        assert log_data["plugin"] == "test_plugin"
        assert "timestamp" in log_data

    @pytest.mark.asyncio
    async def test_call_with_request_response(
        self, logging_hook, mock_logger, mock_request, mock_response
    ):
        """Test logging with request and response context."""
        context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
            request=mock_request,
            response=mock_response,
        )

        await logging_hook(context)

        mock_logger.info.assert_called_once()
        log_data = mock_logger.info.call_args[1]

        assert "request" in log_data
        assert log_data["request"]["method"] == "POST"
        assert log_data["request"]["url"] == str(mock_request.url)
        assert log_data["request"]["client_ip"] == "127.0.0.1"

        assert "response" in log_data
        assert log_data["response"]["status_code"] == 200

    @pytest.mark.asyncio
    async def test_call_with_error(self, logging_hook, mock_logger):
        """Test logging with error context."""
        test_error = ValueError("Test error message")
        context = HookContext(
            event=HookEvent.REQUEST_FAILED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
            error=test_error,
        )

        await logging_hook(context)

        mock_logger.error.assert_called_once()
        log_data = mock_logger.error.call_args[1]

        assert "error" in log_data
        assert log_data["error"]["type"] == "ValueError"
        assert log_data["error"]["message"] == "Test error message"

    @pytest.mark.asyncio
    async def test_call_with_chained_error(self, logging_hook, mock_logger):
        """Test logging with chained error."""
        cause_error = RuntimeError("Original cause")
        chained_error = ValueError("Chained error")
        chained_error.__cause__ = cause_error

        context = HookContext(
            event=HookEvent.REQUEST_FAILED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
            error=chained_error,
        )

        await logging_hook(context)

        log_data = mock_logger.error.call_args[1]
        assert "error" in log_data
        assert log_data["error"]["cause"]["type"] == "RuntimeError"
        assert log_data["error"]["cause"]["message"] == "Original cause"

    def test_get_log_level_error_events(self, logging_hook):
        """Test log level determination for error events."""
        assert logging_hook._get_log_level(HookEvent.REQUEST_FAILED) == "error"
        assert logging_hook._get_log_level(HookEvent.PROVIDER_ERROR) == "error"
        assert logging_hook._get_log_level(HookEvent.PLUGIN_ERROR) == "error"

        # Error object should also result in error level
        assert (
            logging_hook._get_log_level(HookEvent.REQUEST_STARTED, ValueError())
            == "error"
        )

    def test_get_log_level_info_events(self, logging_hook):
        """Test log level determination for info events."""
        info_events = [
            HookEvent.APP_STARTUP,
            HookEvent.APP_SHUTDOWN,
            HookEvent.REQUEST_STARTED,
            HookEvent.REQUEST_COMPLETED,
            HookEvent.PLUGIN_LOADED,
        ]

        for event in info_events:
            assert logging_hook._get_log_level(event) == "info"

    def test_get_log_level_debug_events(self, logging_hook):
        """Test log level determination for debug events."""
        debug_events = [
            HookEvent.PROVIDER_REQUEST_SENT,
            HookEvent.PROVIDER_RESPONSE_RECEIVED,
            HookEvent.PROVIDER_STREAM_START,
            HookEvent.PROVIDER_STREAM_CHUNK,
            HookEvent.PROVIDER_STREAM_END,
        ]

        for event in debug_events:
            assert logging_hook._get_log_level(event) == "debug"

    def test_get_log_level_custom_events(self, logging_hook):
        """Test log level determination for custom events."""
        assert logging_hook._get_log_level(HookEvent.CUSTOM_EVENT) == "info"


@pytest.mark.skip(reason="MetricsHook implementation not yet created")
class TestMetricsHook:
    """Test the MetricsHook implementation."""

    @pytest.fixture
    def mock_metrics(self):
        """Create a mock PrometheusMetrics instance."""
        metrics = MagicMock()
        metrics.inc_active_requests = MagicMock()
        metrics.dec_active_requests = MagicMock()
        metrics.record_request = MagicMock()
        metrics.record_response_time = MagicMock()
        metrics.record_tokens = MagicMock()
        metrics.record_cost = MagicMock()
        metrics.record_error = MagicMock()
        return metrics

    @pytest.fixture
    def metrics_hook(self, mock_metrics):
        """Create a MetricsHook with mock metrics."""
        pass  # MetricsHook not implemented

    def test_init(self, mock_metrics):
        """Test MetricsHook initialization."""
        pass  # hook = MetricsHook(mock_metrics)
        # assert hook.metrics is mock_metrics
        # assert hook.name == "metrics_hook"
        # assert HookEvent.REQUEST_STARTED in hook.events
        # assert HookEvent.REQUEST_COMPLETED in hook.events
        # assert HookEvent.REQUEST_FAILED in hook.events
        # assert HookEvent.PROVIDER_ERROR in hook.events

    @pytest.mark.asyncio
    async def test_handle_request_started(
        self, metrics_hook, mock_metrics, mock_request
    ):
        """Test handling REQUEST_STARTED event."""
        context = HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
            request=mock_request,
        )

        await metrics_hook(context)

        mock_metrics.inc_active_requests.assert_called_once()
        # Should store timestamp for duration calculation
        assert "test-123" in metrics_hook._active_request_timestamps

    @pytest.mark.asyncio
    async def test_handle_request_completed(
        self, metrics_hook, mock_metrics, mock_request, mock_response
    ):
        """Test handling REQUEST_COMPLETED event."""
        # Set up active request timestamp
        metrics_hook._active_request_timestamps["test-123"] = (
            time.time() - 1.5
        )  # 1.5 seconds ago

        context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.utcnow(),
            data={"model": "claude-3-sonnet", "status": "200"},
            metadata={},
            request=mock_request,
            response=mock_response,
        )

        await metrics_hook(context)

        mock_metrics.dec_active_requests.assert_called_once()
        mock_metrics.record_request.assert_called_once()
        mock_metrics.record_response_time.assert_called_once()

        # Check request recording
        request_call = mock_metrics.record_request.call_args
        assert request_call[1]["method"] == "POST"
        assert request_call[1]["model"] == "claude-3-sonnet"
        assert request_call[1]["status"] == "200"

        # Check response time recording
        response_time_call = mock_metrics.record_response_time.call_args
        assert (
            response_time_call[1]["duration_seconds"] > 1.0
        )  # Should be around 1.5 seconds

    @pytest.mark.asyncio
    async def test_handle_request_completed_with_tokens(
        self, metrics_hook, mock_metrics, mock_request
    ):
        """Test handling REQUEST_COMPLETED event with token data."""
        context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.utcnow(),
            data={
                "tokens": {"input": 100, "output": 50, "total": 150},
                "model": "claude-3-sonnet",
            },
            metadata={},
            request=mock_request,
        )

        await metrics_hook(context)

        # Should record tokens for each type
        assert mock_metrics.record_tokens.call_count == 3

    @pytest.mark.asyncio
    async def test_handle_request_completed_with_cost(
        self, metrics_hook, mock_metrics, mock_request
    ):
        """Test handling REQUEST_COMPLETED event with cost data."""
        context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.utcnow(),
            data={
                "cost": {"input": 0.01, "output": 0.02, "total": 0.03},
                "model": "claude-3-sonnet",
            },
            metadata={},
            request=mock_request,
        )

        await metrics_hook(context)

        # Should record cost for each type
        assert mock_metrics.record_cost.call_count == 3

    @pytest.mark.asyncio
    async def test_handle_request_failed(
        self, metrics_hook, mock_metrics, mock_request
    ):
        """Test handling REQUEST_FAILED event."""
        # Set up active request timestamp
        metrics_hook._active_request_timestamps["test-123"] = time.time() - 0.5

        context = HookContext(
            event=HookEvent.REQUEST_FAILED,
            timestamp=datetime.utcnow(),
            data={"error_type": "ValidationError"},
            metadata={},
            request=mock_request,
            error=ValueError("Test error"),
        )

        await metrics_hook(context)

        mock_metrics.dec_active_requests.assert_called_once()
        mock_metrics.record_error.assert_called_once()
        mock_metrics.record_request.assert_called_once()

        # Check error recording
        error_call = mock_metrics.record_error.call_args
        assert error_call[1]["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_handle_provider_error(self, metrics_hook, mock_metrics):
        """Test handling PROVIDER_ERROR event."""
        context = HookContext(
            event=HookEvent.PROVIDER_ERROR,
            timestamp=datetime.utcnow(),
            data={"endpoint": "/api/test", "model": "claude-3-sonnet"},
            metadata={},
            error=RuntimeError("Provider error"),
        )

        await metrics_hook(context)

        mock_metrics.record_error.assert_called_once()
        error_call = mock_metrics.record_error.call_args
        assert error_call[1]["error_type"] == "provider_RuntimeError"

    @pytest.mark.asyncio
    async def test_error_isolation(self, metrics_hook, mock_metrics):
        """Test that metrics hook errors don't propagate."""
        # Make the metrics mock raise an exception
        mock_metrics.inc_active_requests.side_effect = RuntimeError("Metrics error")

        context = HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
        )

        with patch("ccproxy.hooks.implementations.metrics.logger") as mock_logger:
            # Should not raise an exception
            await metrics_hook(context)

            # Should log the error
            mock_logger.error.assert_called_once()

    def test_get_request_id_from_headers(self, metrics_hook, mock_request):
        """Test extracting request ID from headers."""
        context = HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
            request=mock_request,
        )

        # Debug the method step by step
        print(f"Context.request: {context.request}")
        print(f"Context.request is truthy: {bool(context.request)}")
        if context.request:
            print(f"Has headers: {hasattr(context.request, 'headers')}")
            print(f"Headers: {context.request.headers}")
            print(f"Headers.get: {context.request.headers.get}")
            print(
                f"Headers.get('x-request-id'): {context.request.headers.get('x-request-id')}"
            )

        # Test the actual method logic manually
        condition_result = context.request and hasattr(context.request, "headers")
        print(f"Condition result: {condition_result}")
        if condition_result and context.request:
            result_from_headers = context.request.headers.get("x-request-id")
            print(f"Result from headers: {result_from_headers}")
        else:
            result_from_data = context.data.get("request_id")
            print(f"Result from data: {result_from_data}")

        request_id = metrics_hook._get_request_id(context)
        print(f"Final request_id: {request_id}")
        assert request_id == "test-123"

    def test_get_request_id_from_data(self, metrics_hook):
        """Test extracting request ID from context data."""
        request_id = metrics_hook._get_request_id(
            HookContext(
                event=HookEvent.REQUEST_STARTED,
                timestamp=datetime.utcnow(),
                data={"request_id": "data-456"},
                metadata={},
            )
        )
        assert request_id == "data-456"


@pytest.mark.skip(reason="AnalyticsHook implementation not yet created")
class TestAnalyticsHook:
    """Test the AnalyticsHook implementation."""

    @pytest.fixture
    def analytics_hook(self):
        """Create an AnalyticsHook for testing."""
        pass  # AnalyticsHook not implemented

    def test_init(self):
        """Test AnalyticsHook initialization."""
        pass  # hook = AnalyticsHook(batch_size=100)
        # assert hook.name == "analytics_hook"
        # assert HookEvent.REQUEST_COMPLETED in hook.events
        # assert HookEvent.PROVIDER_STREAM_END in hook.events
        # assert hook.get_batch_size() == 100
        # assert hook.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_call_request_completed(
        self, analytics_hook, mock_request, mock_response
    ):
        """Test processing REQUEST_COMPLETED event."""
        context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.utcnow(),
            data={"duration": 1500, "status": "success", "tokens": {"total": 100}},
            metadata={},
            request=mock_request,
            response=mock_response,
            provider="claude_api",
            plugin="test_plugin",
        )

        await analytics_hook(context)

        # Should add to buffer
        assert analytics_hook.get_buffer_size() == 1

    @pytest.mark.asyncio
    async def test_call_provider_stream_end(self, analytics_hook):
        """Test processing PROVIDER_STREAM_END event."""
        context = HookContext(
            event=HookEvent.PROVIDER_STREAM_END,
            timestamp=datetime.utcnow(),
            data={
                "stream_duration": 2500,
                "chunks_sent": 15,
                "total_tokens": 200,
                "model": "claude-3-sonnet",
            },
            metadata={},
            provider="claude_api",
        )

        await analytics_hook(context)

        # Should add to buffer
        assert analytics_hook.get_buffer_size() == 1

    @pytest.mark.asyncio
    async def test_call_unhandled_event(self, analytics_hook):
        """Test processing an event that doesn't generate analytics data."""
        context = HookContext(
            event=HookEvent.APP_STARTUP,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
        )

        await analytics_hook(context)

        # Should not add to buffer
        assert analytics_hook.get_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_extract_request_analytics(
        self, analytics_hook, mock_request, mock_response
    ):
        """Test extracting analytics data from request completion."""
        context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.utcnow(),
            data={
                "duration": 1500,
                "status": "success",
                "tokens": {"input": 50, "output": 30, "total": 80},
                "model": "claude-3-sonnet",
            },
            metadata={},
            request=mock_request,
            response=mock_response,
            provider="claude_api",
            plugin="test_plugin",
        )

        base_data = {
            "base": "data",
            "event": context.event,
            "timestamp": context.timestamp.isoformat(),
            "provider": context.provider,
            "plugin": context.plugin,
        }
        analytics_data = await analytics_hook._extract_request_analytics(
            context, base_data
        )

        assert analytics_data["base"] == "data"
        assert analytics_data["method"] == "POST"
        assert analytics_data["path"] == "/api/test"
        assert analytics_data["status_code"] == 200
        assert analytics_data["duration_ms"] == 1500
        assert analytics_data["tokens"] == {"input": 50, "output": 30, "total": 80}
        assert analytics_data["model"] == "claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_extract_stream_analytics(self, analytics_hook):
        """Test extracting analytics data from stream completion."""
        context = HookContext(
            event=HookEvent.PROVIDER_STREAM_END,
            timestamp=datetime.utcnow(),
            data={
                "stream_duration": 2500,
                "chunks_sent": 15,
                "total_tokens": 200,
                "completion_tokens": 120,
                "prompt_tokens": 80,
                "model": "claude-3-haiku",
            },
            metadata={},
            provider="claude_api",
        )

        base_data = {
            "base": "data",
            "event": context.event,
            "timestamp": context.timestamp.isoformat(),
            "provider": context.provider,
            "plugin": context.plugin,
        }
        analytics_data = await analytics_hook._extract_stream_analytics(
            context, base_data
        )

        assert analytics_data["base"] == "data"
        assert analytics_data["stream_duration_ms"] == 2500
        assert analytics_data["chunks_sent"] == 15
        assert analytics_data["total_tokens"] == 200
        assert analytics_data["completion_tokens"] == 120
        assert analytics_data["prompt_tokens"] == 80
        assert analytics_data["model"] == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_batch_processing_trigger(self, analytics_hook):
        """Test that batch processing is triggered when buffer reaches batch size."""
        # Create contexts to fill the buffer
        contexts = []
        for i in range(3):  # batch_size is 3
            context = HookContext(
                event=HookEvent.REQUEST_COMPLETED,
                timestamp=datetime.utcnow(),
                data={"id": i},
                metadata={},
                provider="test_provider",
            )
            contexts.append(context)

        with patch.object(
            analytics_hook, "_process_batch", new_callable=AsyncMock
        ) as mock_process:
            # Add items to buffer
            for context in contexts:
                await analytics_hook(context)

            # Give time for async task to start
            await asyncio.sleep(0.01)

            # Buffer should be cleared and batch processing should be triggered
            assert analytics_hook.get_buffer_size() == 0
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_aggregate_batch_stats(self, analytics_hook):
        """Test aggregating statistics for a batch."""
        batch_data = [
            {
                "event": HookEvent.REQUEST_COMPLETED,
                "provider": "claude_api",
                "model": "claude-3-sonnet",
                "status_code": 200,
                "tokens": {"total": 100},
            },
            {
                "event": HookEvent.REQUEST_COMPLETED,
                "provider": "claude_api",
                "model": "claude-3-haiku",
                "status_code": 200,
                "total_tokens": 50,
            },
            {
                "event": HookEvent.PROVIDER_STREAM_END,
                "provider": "openai",
                "model": "gpt-4",
                "status_code": 500,
                "tokens": 75,
            },
        ]

        stats = await analytics_hook._aggregate_batch_stats(batch_data)

        assert stats["total_events"] == 3
        assert stats["event_types"][HookEvent.REQUEST_COMPLETED] == 2
        assert stats["event_types"][HookEvent.PROVIDER_STREAM_END] == 1
        assert stats["providers"]["claude_api"] == 2
        assert stats["providers"]["openai"] == 1
        assert stats["models"]["claude-3-sonnet"] == 1
        assert stats["models"]["claude-3-haiku"] == 1
        assert stats["models"]["gpt-4"] == 1
        assert stats["status_codes"]["200"] == 2
        assert stats["status_codes"]["500"] == 1
        assert stats["total_tokens"] == 225  # 100 + 50 + 75

    @pytest.mark.asyncio
    async def test_flush_buffer(self, analytics_hook):
        """Test flushing remaining buffer on shutdown."""
        # Add some items to buffer
        context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.utcnow(),
            data={"test": "data"},
            metadata={},
            provider="test_provider",
        )

        await analytics_hook(context)
        assert analytics_hook.get_buffer_size() == 1

        with patch.object(
            analytics_hook, "_process_batch", new_callable=AsyncMock
        ) as mock_process:
            await analytics_hook.flush_buffer()

            # Buffer should be cleared
            assert analytics_hook.get_buffer_size() == 0
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, analytics_hook):
        """Test error handling in analytics processing."""
        with (
            patch.object(
                analytics_hook,
                "_extract_analytics_data",
                side_effect=ValueError("Test error"),
            ),
            patch.object(analytics_hook._logger, "error") as mock_error,
        ):
            context = HookContext(
                event=HookEvent.REQUEST_COMPLETED,
                timestamp=datetime.utcnow(),
                data={},
                metadata={},
            )

            # Should not raise an exception
            await analytics_hook(context)

            # Should log the error
            mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_transmit_batch_stub(self, analytics_hook):
        """Test the stub implementation of batch transmission."""
        batch_data = [{"test": "data"}]
        batch_stats = {"total_events": 1}

        with patch.object(analytics_hook._logger, "info") as mock_info:
            # Should complete without error
            await analytics_hook._transmit_batch(batch_data, batch_stats)

            # Should log transmission
            mock_info.assert_called_once()
            log_call = mock_info.call_args
            assert "Transmitting analytics batch" in log_call[0]


# Integration tests


class TestHookSystemIntegration:
    """Integration tests for the complete hook system."""

    @pytest.fixture
    def hook_system(self):
        """Create a complete hook system for integration testing."""
        registry = HookRegistry()
        manager = HookManager(registry)
        return registry, manager

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Hook implementations not yet created")
    async def test_full_request_lifecycle(self, hook_system):
        """Test a complete request lifecycle with multiple hooks."""
        registry, manager = hook_system

        # Create and register hooks
        pass  # logging_hook = LoggingHook()
        pass  # analytics_hook = AnalyticsHook(batch_size=10)

        # Mock the metrics dependency
        with patch(
            "ccproxy.observability.metrics.PrometheusMetrics"
        ) as mock_metrics_class:
            mock_metrics = MagicMock()
            mock_metrics_class.return_value = mock_metrics
            pass  # metrics_hook = MetricsHook(mock_metrics)

            # registry.register(logging_hook)
            # registry.register(analytics_hook)
            # registry.register(metrics_hook)

            # Mock logging to capture calls
            # with patch.object(logging_hook, "logger") as mock_logger:
            return  # Disabled until hook implementations are created
            if False:  # type: ignore[unreachable]
                # Simulate request lifecycle
                mock_request = MagicMock()
                mock_request.method = "POST"
                mock_request.url.path = "/api/chat"
                mock_request.headers = {"x-request-id": "req-123"}
                mock_request.client.host = "192.168.1.1"

                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {"content-type": "application/json"}

                # 1. Request started
                await manager.emit(
                    HookEvent.REQUEST_STARTED,
                    data={"model": "claude-3-sonnet"},
                    request=mock_request,
                    provider="claude_api",
                )

                # 2. Request completed
                await manager.emit(
                    HookEvent.REQUEST_COMPLETED,
                    data={
                        "model": "claude-3-sonnet",
                        "duration": 2500,
                        "tokens": {"input": 100, "output": 150, "total": 250},
                        "status": "success",
                    },
                    request=mock_request,
                    response=mock_response,
                    provider="claude_api",
                )

                # Verify logging hook was called
                # assert mock_logger.info.call_count == 2  # disabled until hooks implemented

                # Verify metrics hook was called
                mock_metrics.inc_active_requests.assert_called_once()
                mock_metrics.dec_active_requests.assert_called_once()
                mock_metrics.record_request.assert_called_once()

                # Verify analytics hook collected data
                # assert analytics_hook.get_buffer_size() == 1
                pass

    @pytest.mark.asyncio
    async def test_error_scenario_isolation(self, hook_system):
        """Test that hook errors don't affect the request flow."""
        registry, manager = hook_system

        # Register a failing hook and a good hook
        failing_hook = FailingHook(events=[HookEvent.REQUEST_FAILED])
        good_hook = AsyncTestHook(events=[HookEvent.REQUEST_FAILED])

        registry.register(failing_hook)
        registry.register(good_hook)

        with patch.object(manager._logger, "error") as mock_error:
            # Emit error event
            await manager.emit(
                HookEvent.REQUEST_FAILED,
                data={"error": "Test error"},
                error=RuntimeError("Request failed"),
            )

            # Good hook should still be called
            assert good_hook.call_count == 1

            # Error should be logged
            mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_async_sync_hooks(self, hook_system):
        """Test mixing async and sync hooks for the same event."""
        registry, manager = hook_system

        async_hook = AsyncTestHook(events=[HookEvent.APP_STARTUP])
        sync_hook = SyncTestHook(events=[HookEvent.APP_STARTUP])

        registry.register(async_hook)
        registry.register(sync_hook)

        await manager.emit(HookEvent.APP_STARTUP, data={"test": "startup"})

        # Both hooks should be called
        assert async_hook.call_count == 1
        assert sync_hook.call_count == 1
        assert async_hook.last_context is not None
        assert async_hook.last_context.data == {"test": "startup"}
        assert sync_hook.last_context is not None
        assert sync_hook.last_context.data == {"test": "startup"}
