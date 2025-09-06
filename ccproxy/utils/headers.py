"""Header handling utilities with an ergonomic class API.

HeaderBag provides ordered, canonicalized header storage with convenient
case-insensitive access while preserving original intent:
- Preserve incoming header order (ASGI raw headers)
- Canonicalize names (e.g., "Accept", "Content-Type", custom headers title-cased)
- Support dict views and pair iteration

Thin helper functions are kept for backward compatibility.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping


# Common header casing overrides for non-trivial capitalization
_SPECIAL_CASES: dict[str, str] = {
    "www-authenticate": "WWW-Authenticate",
    "etag": "ETag",
    "dnt": "DNT",
    "te": "TE",
}


def canonicalize_header_name(name: str) -> str:
    """Return a canonical HTTP-style header name.

    Examples:
    - "content-type" -> "Content-Type"
    - "accept" -> "Accept"
    - "x-request-id" -> "X-Request-ID"
    - Applies overrides for known special cases like "ETag".
    """
    key = name.strip().lower()
    if key in _SPECIAL_CASES:
        return _SPECIAL_CASES[key]

    # Title-case hyphen-separated parts, e.g., x-request-id -> X-Request-Id
    # This is a reasonable default for custom headers as well.
    parts = [p for p in key.split("-") if p]
    return "-".join(p.capitalize() for p in parts)


class HeaderBag:
    """Ordered, canonicalized header container with case-insensitive access.

    Internally stores a list of (name, value) pairs with canonicalized names
    while preserving insertion order. Accessors perform case-insensitive name
    matching where appropriate.
    """

    def __init__(
        self,
        pairs: Iterable[tuple[str, str]] | None = None,
        preserve_case: bool = False,
        case_mode: str | None = None,
    ) -> None:
        self._pairs: list[tuple[str, str]] = []
        if case_mode is None:
            self._case_mode = "preserve" if preserve_case else "canonical"
        else:
            cm = str(case_mode).lower()
            self._case_mode = (
                cm if cm in {"preserve", "lower", "canonical"} else "canonical"
            )
        if pairs:
            for name, value in pairs:
                self.add(name, value)

    @classmethod
    def from_request(cls, request: object, case_mode: str | None = None) -> HeaderBag:
        """Build from a FastAPI/Starlette request preserving order and casing.

        Prefers `request.headers.raw` if available; otherwise attempts ASGI scope
        headers; falls back to the mapping interface.
        """
        # Try `request.headers.raw` (list[tuple[bytes, bytes]])
        try:
            raw = request.headers.raw  # type: ignore[attr-defined]
            pairs = ((k.decode("latin-1"), v.decode("latin-1")) for k, v in raw)
            return cls.from_pairs(pairs, case_mode=case_mode)
        except Exception:
            pass

        # Try ASGI scope
        try:
            scope = getattr(request, "scope", {}) or {}
            raw = scope.get("headers")
            if isinstance(raw, list):
                pairs = ((k.decode("latin-1"), v.decode("latin-1")) for k, v in raw)
                return cls.from_pairs(pairs, case_mode=case_mode)
        except Exception:
            pass

        # Fallback: mapping
        try:
            mapping = dict(getattr(request, "headers", {}))
            return cls.from_mapping(mapping, case_mode=case_mode)
        except Exception:
            return cls(case_mode=case_mode)

    @classmethod
    def from_mapping(
        cls,
        mapping: MutableMapping[str, str] | dict[str, str],
        case_mode: str | None = None,
    ) -> HeaderBag:
        return cls.from_pairs(mapping.items(), case_mode=case_mode)

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[str, str]] | list[tuple[str, str]],
        case_mode: str | None = None,
    ) -> HeaderBag:
        bag = cls(case_mode=case_mode)
        for name, value in pairs:
            bag.add(name, value)
        return bag

    @classmethod
    def from_httpx_response(
        cls, response: object, case_mode: str | None = None
    ) -> HeaderBag:
        """Build from an httpx.Response preserving order and duplicates.

        Prefers response.headers.raw when available; falls back to items().
        """
        try:
            raw = response.headers.raw  # type: ignore[attr-defined]
            pairs = ((k.decode("latin-1"), v.decode("latin-1")) for k, v in raw)
            return cls.from_pairs(pairs, case_mode=case_mode)
        except Exception:
            pass
        try:
            items = response.headers.items()  # type: ignore[attr-defined]
            return cls.from_pairs(items, case_mode=case_mode)
        except Exception:
            return cls(case_mode=case_mode)

    @classmethod
    def from_starlette_response(
        cls, response: object, case_mode: str | None = None
    ) -> HeaderBag:
        """Build from a Starlette/FastAPI Response preserving order and duplicates.

        Prefers response.raw_headers; falls back to response.headers.items().
        """
        try:
            raw = response.raw_headers  # list[tuple[bytes, bytes]]
            if isinstance(raw, list):
                pairs = ((k.decode("latin-1"), v.decode("latin-1")) for k, v in raw)
                return cls.from_pairs(pairs, case_mode=case_mode)
        except Exception:
            pass
        try:
            items = response.headers.items()  # type: ignore[attr-defined]
            return cls.from_pairs(items, case_mode=case_mode)
        except Exception:
            return cls(case_mode=case_mode)

    def add(self, name: str, value: str) -> None:
        """Append a header value (does not replace existing entries)."""
        raw = str(name)
        if self._case_mode == "preserve":
            final = raw
        elif self._case_mode == "lower":
            final = raw.lower()
        else:
            final = canonicalize_header_name(raw)
        self._pairs.append((final, str(value)))

    def set(self, name: str, value: str) -> None:
        """Replace any existing entries for name and append the new value."""
        if self._case_mode == "preserve" or self._case_mode == "lower":
            key_l = name.lower()
        else:
            key_l = canonicalize_header_name(name).lower()
        self._pairs = [(n, v) for (n, v) in self._pairs if n.lower() != key_l]
        self.add(name, value)

    def get(self, name: str, default: str | None = None) -> str | None:
        key_l = name.lower()
        for n, v in reversed(self._pairs):
            if n.lower() == key_l:
                return v
        return default

    def exclude(self, names: Iterable[str]) -> HeaderBag:
        """Return a new HeaderBag excluding case-insensitive header names."""
        exclude_set = {n.lower() for n in names}
        return HeaderBag(
            (n, v) for (n, v) in self._pairs if n.lower() not in exclude_set
        )

    def to_dict(self, last_wins: bool = True) -> dict[str, str]:
        """Materialize as an ordered dict; by default, last value wins per name."""
        out: dict[str, str] = {}
        if last_wins:
            for n, v in self._pairs:
                out[n] = v
        else:
            # keep first occurrence
            for n, v in self._pairs:
                if n not in out:
                    out[n] = v
        return out

    def items(self) -> Iterator[tuple[str, str]]:
        return iter(self._pairs)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return self.items()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._pairs)


def canonicalize_headers_preserve_order(
    headers: Iterable[tuple[str, str]] | dict[str, str],
) -> dict[str, str]:
    """Compatibility wrapper using HeaderBag under the hood."""
    if isinstance(headers, dict):
        return HeaderBag.from_mapping(headers).to_dict()
    return HeaderBag.from_pairs(headers).to_dict()


def extract_ordered_request_headers(request: object) -> dict[str, str]:
    """Extract headers from a Starlette/FastAPI request preserving order and casing.

    Uses the ASGI `scope["headers"]` or `request.headers.raw` when available to
    obtain the incoming header order. Falls back to `request.headers.items()` if
    raw access is unavailable.
    """
    return HeaderBag.from_request(request).to_dict()
