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
