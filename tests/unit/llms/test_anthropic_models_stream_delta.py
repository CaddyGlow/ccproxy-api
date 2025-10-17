import pytest

from ccproxy.llms.models import anthropic as anthropic_models


@pytest.mark.unit
def test_content_block_delta_accepts_text_delta() -> None:
    evt = anthropic_models.ContentBlockDeltaEvent.model_validate(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        }
    )
    assert evt.delta.type == "text_delta"
    assert getattr(evt.delta, "text", "") == "hi"


@pytest.mark.unit
def test_content_block_delta_accepts_input_json_delta() -> None:
    evt = anthropic_models.ContentBlockDeltaEvent.model_validate(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"foo":'},
        }
    )
    assert evt.delta.type == "input_json_delta"
    assert getattr(evt.delta, "partial_json", "") == '{"foo":'


@pytest.mark.unit
def test_content_block_delta_accepts_thinking_delta() -> None:
    evt = anthropic_models.ContentBlockDeltaEvent.model_validate(
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "thinking_delta", "thinking": "pondering"},
        }
    )
    assert evt.delta.type == "thinking_delta"
    assert getattr(evt.delta, "thinking", "") == "pondering"


@pytest.mark.unit
def test_content_block_delta_accepts_signature_delta() -> None:
    evt = anthropic_models.ContentBlockDeltaEvent.model_validate(
        {
            "type": "content_block_delta",
            "index": 3,
            "delta": {"type": "signature_delta", "signature": "sig-123"},
        }
    )
    assert evt.delta.type == "signature_delta"
    assert getattr(evt.delta, "signature", "") == "sig-123"
