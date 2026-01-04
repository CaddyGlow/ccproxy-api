from ccproxy.llms.formatters.common import (
    THINKING_PATTERN,
    ThinkingSegment,
    merge_thinking_segments,
)


def test_thinking_segment_to_xml_round_trip() -> None:
    segment = ThinkingSegment(thinking="Thoughts", signature="sig")
    xml = segment.to_xml()
    match = THINKING_PATTERN.search(xml)
    assert match is not None
    captured_signature, captured_text = match.groups()
    assert captured_signature == "sig"
    assert captured_text == "Thoughts"

    rebuilt = ThinkingSegment.from_xml(captured_signature, captured_text)
    assert rebuilt == segment


def test_merge_thinking_segments_collapses_adjacent() -> None:
    segments = [
        ThinkingSegment(thinking="A", signature="sig"),
        ThinkingSegment(thinking="B", signature="sig"),
        ThinkingSegment(thinking="C", signature=None),
    ]

    merged = merge_thinking_segments(segments)
    assert len(merged) == 2
    assert merged[0].thinking == "AB"
    assert merged[0].signature == "sig"
    assert merged[1].thinking == "C"
    assert merged[1].signature is None


def test_thinking_segment_to_block() -> None:
    segment = ThinkingSegment(thinking="Ok", signature=None)
    block = segment.to_block()
    assert block.type == "thinking"
    assert block.thinking == "Ok"
    assert block.signature == ""
