import uuid

from pytest import MonkeyPatch

from ccproxy.llms.formatters.common import ensure_identifier, normalize_suffix


def test_normalize_suffix_splits_first_underscore() -> None:
    assert normalize_suffix("resp_1234") == "1234"
    assert normalize_suffix("plain") == "plain"


def test_ensure_identifier_reuses_matching_prefix() -> None:
    identifier, suffix = ensure_identifier("resp", "resp_existing")
    assert identifier == "resp_existing"
    assert suffix == "existing"


def test_ensure_identifier_translates_resp_prefix() -> None:
    identifier, suffix = ensure_identifier("msg", "resp_existing")
    assert identifier == "msg_existing"
    assert suffix == "existing"


def test_ensure_identifier_generates_uuid(monkeypatch: MonkeyPatch) -> None:
    class DummyUUID:
        hex = "cafefeed"

    def fake_uuid4() -> DummyUUID:  # pragma: no cover - simple stub
        return DummyUUID()

    monkeypatch.setattr(uuid, "uuid4", fake_uuid4)

    identifier, suffix = ensure_identifier("resp")
    assert identifier == "resp_cafefeed"
    assert suffix == "cafefeed"
