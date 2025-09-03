from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse


router = APIRouter()


@router.get("/dashboard")
async def get_metrics_dashboard() -> HTMLResponse:
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    dashboard_folder = project_root / "ccproxy" / "static" / "dashboard"
    dashboard_index = dashboard_folder / "index.html"

    if not dashboard_folder.exists():
        raise HTTPException(
            status_code=404,
            detail="Dashboard not found. Build it with 'cd dashboard && bun run build:prod'",
        )
    if not dashboard_index.exists():
        raise HTTPException(
            status_code=404,
            detail="Dashboard index.html not found. Rebuild with 'cd dashboard && bun run build:prod'",
        )

    try:
        html_content = dashboard_index.read_text(encoding="utf-8")
        return HTMLResponse(
            content=html_content,
            status_code=200,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Type": "text/html; charset=utf-8",
            },
        )
    except (OSError, PermissionError) as e:
        raise HTTPException(
            status_code=500, detail=f"Dashboard file access error: {str(e)}"
        ) from e
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=500, detail=f"Dashboard file encoding error: {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to serve dashboard: {str(e)}"
        ) from e


@router.get("/dashboard/favicon.svg")
async def get_dashboard_favicon() -> FileResponse:
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    favicon_path = project_root / "ccproxy" / "static" / "dashboard" / "favicon.svg"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(
        path=str(favicon_path),
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )
