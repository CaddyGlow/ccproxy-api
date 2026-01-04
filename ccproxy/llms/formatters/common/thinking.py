"""Shared helpers for reasoning/thinking segment handling."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ccproxy.llms.models import anthropic as anthropic_models


THINKING_PATTERN = re.compile(
    r"<thinking(?:\s+signature=\"([^\"]*)\")?>(.*?)</thinking>",
    re.DOTALL,
)
THINKING_OPEN_PATTERN = re.compile(
    r"<thinking(?:\s+signature=\"([^\"]*)\")?\s*>",
    re.IGNORECASE,
)
THINKING_CLOSE_PATTERN = re.compile(r"</thinking>", re.IGNORECASE)


@dataclass(slots=True)
class ThinkingSegment:
    """Lightweight reasoning segment mirroring Anthropic's ThinkingBlock."""

    thinking: str
    signature: str | None = None

    def to_block(self) -> anthropic_models.ThinkingBlock:
        return anthropic_models.ThinkingBlock(
            type="thinking",
            thinking=self.thinking,
            signature=self.signature or "",
        )

    def to_xml(self) -> str:
        signature = (self.signature or "").strip()
        signature_attr = f' signature="{signature}"' if signature else ""
        return f"<thinking{signature_attr}>{self.thinking}</thinking>"

    @classmethod
    def from_xml(cls, signature: str | None, text: str) -> ThinkingSegment:
        return cls(thinking=text, signature=signature or None)


def merge_thinking_segments(
    segments: Iterable[ThinkingSegment],
) -> list[ThinkingSegment]:
    """Collapse adjacent segments that share the same signature."""

    merged: list[ThinkingSegment] = []
    for segment in segments:
        text = segment.thinking if isinstance(segment.thinking, str) else None
        if not text:
            continue
        signature = segment.signature or None
        if merged and merged[-1].signature == signature:
            merged[-1] = ThinkingSegment(
                thinking=f"{merged[-1].thinking}{text}",
                signature=signature,
            )
        else:
            merged.append(ThinkingSegment(thinking=text, signature=signature))
    return merged


__all__ = [
    "THINKING_PATTERN",
    "THINKING_OPEN_PATTERN",
    "THINKING_CLOSE_PATTERN",
    "ThinkingSegment",
    "merge_thinking_segments",
]
