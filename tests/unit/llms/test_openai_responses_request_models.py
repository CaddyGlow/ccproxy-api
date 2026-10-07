from ccproxy.llms.models import openai as openai_models


def test_response_request_accepts_previous_response_id_and_function_output() -> None:
    request = openai_models.ResponseRequest.model_validate(
        {
            "model": "gpt-5.5",
            "previous_response_id": "resp_123",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": '{"temperature":"25C"}',
                }
            ],
            "reasoning": {"effort": "xhigh"},
            "stream": True,
            "stream_options": {"include_usage": True},
            "prompt_cache_retention": "24h",
            "safety_identifier": "user-123",
        }
    )

    assert request.previous_response_id == "resp_123"
    assert isinstance(request.input, list)
    assert request.input[0]["type"] == "function_call_output"
    assert request.reasoning == {"effort": "xhigh"}
    assert request.stream is True
    assert request.stream_options is not None
    assert request.stream_options.include_usage is True
    assert request.prompt_cache_retention == "24h"
    assert request.safety_identifier == "user-123"
