from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.services.exporter import build_docx
from app.storage import get_project

router = APIRouter(tags=["export"])


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "response"


@router.get("/projects/{project_id}/export")
async def export_project(
    project_id: str,
    include_drafts: bool = Query(False),
) -> StreamingResponse:
    project = await get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    buf = build_docx(project, include_drafts=include_drafts)
    filename = f"{_safe_name(project['name'])}_response.docx"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
