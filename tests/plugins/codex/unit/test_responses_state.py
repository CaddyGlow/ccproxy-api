import pytest

from ccproxy.plugins.codex.responses_state import (
    CodexResponsesStateStore,
    ResponsesStateNotFoundError,
)


def test_responses_state_expands_previous_response_id_tool_loop() -> None:
    store = CodexResponsesStateStore(max_entries=8, ttl_seconds=60)
    headers = {"authorization": "Bearer client-a"}
    scope = store.scope_for_headers(headers)

    stored = store.store_response(
        scope=scope,
        request_payload={
            "input": "What is the weather in Paris?",
            "model": "gpt-5",
        },
        response_payload={
            "id": "resp_tool_1",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "call_weather",
                    "call_id": "call_weather",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                    "status": "completed",
                }
            ],
        },
    )
    assert stored is True

    prepared, prepared_scope, previous_response_id = store.prepare_payload(
        {
            "model": "gpt-5",
            "previous_response_id": "resp_tool_1",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": '{"temperature":"25C"}',
                }
            ],
        },
        headers=headers,
    )

    assert prepared_scope == scope
    assert previous_response_id == "resp_tool_1"
    assert "previous_response_id" not in prepared
    assert [item["type"] for item in prepared["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert prepared["input"][1]["id"] == "fc_weather"
    assert prepared["input"][1]["call_id"] == "call_weather"
    assert prepared["input"][2]["call_id"] == "call_weather"


def test_responses_state_is_scoped_by_client_auth() -> None:
    store = CodexResponsesStateStore(max_entries=8, ttl_seconds=60)
    scope = store.scope_for_headers({"authorization": "Bearer client-a"})
    assert store.store_response(
        scope=scope,
        request_payload={"input": "hello"},
        response_payload={
            "id": "resp_a",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
        },
    )

    with pytest.raises(ResponsesStateNotFoundError):
        store.prepare_payload(
            {"previous_response_id": "resp_a", "input": "continue"},
            headers={"authorization": "Bearer client-b"},
        )


def test_responses_state_does_not_store_failed_response() -> None:
    store = CodexResponsesStateStore(max_entries=8, ttl_seconds=60)
    scope = store.scope_for_headers({"authorization": "Bearer client-a"})
    assert (
        store.store_response(
            scope=scope,
            request_payload={"input": "hello"},
            response_payload={
                "id": "resp_failed",
                "status": "failed",
                "output": [],
            },
        )
        is False
    )
