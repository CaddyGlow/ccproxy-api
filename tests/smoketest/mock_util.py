import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse


MOCKS_DIR = Path(__file__).parent / "mocks"


def _ensure_dir() -> None:
    MOCKS_DIR.mkdir(parents=True, exist_ok=True)


def is_record_mode() -> bool:
    return os.getenv("RECORD_MOCKS", "").lower() in {"1", "true", "yes", "on"}


def normal_paths(name: str) -> Path:
    _ensure_dir()
    return MOCKS_DIR / f"{name}.mock.json"


def stream_paths(name: str) -> tuple[Path, Path]:
    _ensure_dir()
    return (
        MOCKS_DIR / f"{name}.mock.stream.jsonl",
        MOCKS_DIR / f"{name}.mock.stream.headers.json",
    )


def save_normal_mock(
    name: str, status: int, headers: dict[str, str], body: Any
) -> None:
    path = normal_paths(name)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"status": status, "headers": headers, "body": body}, f)


def load_normal_mock(name: str) -> dict[str, Any]:
    path = normal_paths(name)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_stream_headers(name: str, status: int, headers: dict[str, str]) -> None:
    _, headers_path = stream_paths(name)
    with headers_path.open("w", encoding="utf-8") as f:
        json.dump({"status": status, "headers": headers}, f)


def load_stream_headers(name: str) -> dict[str, Any]:
    _, headers_path = stream_paths(name)
    with headers_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_stream_lines(name: str, lines: Iterable[str]) -> None:
    lines_path, _ = stream_paths(name)
    with lines_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def load_stream_lines(name: str) -> Iterable[str]:
    lines_path, _ = stream_paths(name)
    with lines_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def hop_by_hop_filter(headers: dict[str, str]) -> dict[str, str]:
    forbidden = {"connection", "transfer-encoding", "content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in forbidden}


def make_mock_middleware(
    routes: dict[tuple[str, str], str],
) -> Callable[[Request, Callable[[Request], Any]], Any]:
    """Create a middleware that records or replays mocks for specific routes.

    routes: mapping of (method, path) -> mock name
    """

    record = is_record_mode()

    async def middleware(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        path = request.url.path
        method = request.method.upper()
        key = (method, path)
        name = routes.get(key)

        if not name:
            return await call_next(request)

        if not record:
            # Playback mode
            if path.endswith("/chat/completions") or path.endswith("/responses"):
                data = load_normal_mock(name)
                headers = hop_by_hop_filter(dict(data.get("headers", {}).items()))
                status = int(data.get("status", 200))
                body = data.get("body", {})
                return JSONResponse(content=body, status_code=status, headers=headers)

            # Streaming playback
            hdrs = load_stream_headers(name)
            headers = hop_by_hop_filter(dict(hdrs.get("headers", {}).items()))
            status = int(hdrs.get("status", 200))

            async def gen() -> AsyncIterator[bytes]:
                for line in load_stream_lines(name):
                    yield (line + "\n").encode()
                    await asyncio.sleep(0)

            return StreamingResponse(
                gen(),
                status_code=status,
                headers=headers,
                media_type=headers.get("content-type", "text/event-stream"),
            )

        # Record mode
        response = await call_next(request)
        # Clone/capture body. For JSON, read and store; for streams, read full text.
        raw = await response.aread()
        headers = hop_by_hop_filter(dict(dict(response.headers).items()))
        status = int(response.status_code)
        content_type = headers.get("content-type", "")
        if "text/event-stream" in content_type:
            text = raw.decode(errors="ignore")
            lines = list(text.splitlines())
            save_stream_headers(name, status, headers)
            save_stream_lines(name, lines)
        else:
            try:
                body = json.loads(raw.decode() or "{}")
            except Exception:
                body = {}
            save_normal_mock(name, status, headers, body)

        # Return a new response with the same content
        if "text/event-stream" in content_type:

            async def regen() -> AsyncIterator[bytes]:
                for ln in lines:
                    yield (ln + "\n").encode()

            return StreamingResponse(
                regen(), status_code=status, headers=headers, media_type=content_type
            )
        else:
            return Response(
                content=raw,
                status_code=status,
                headers=headers,
                media_type=content_type,
            )

    return middleware
