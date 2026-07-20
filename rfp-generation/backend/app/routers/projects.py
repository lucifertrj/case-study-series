from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.services.extraction import extract_project_questions
from app.services.generation import generate_answers_for_project
from app.storage import get_project, list_projects, upsert_project

router = APIRouter(prefix="/projects", tags=["projects"])

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md"}


def _counts(project: dict) -> dict:
    qs = project.get("questions", [])
    total = len(qs)
    approved = sum(1 for q in qs if q.get("status") == "approved")
    draft = sum(1 for q in qs if q.get("status") == "draft")
    unanswered = sum(1 for q in qs if q.get("status") == "unanswered")
    return {
        "total_questions": total,
        "approved_count": approved,
        "draft_count": draft,
        "unanswered_count": unanswered,
    }


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_project(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    client: str = Form(...),
    due_date: str | None = Form(None),
    file: UploadFile = File(...),
) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    project_id = str(uuid.uuid4())
    dest = os.path.join(settings.UPLOAD_DIR, f"rfp-{project_id}{ext}")

    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)

    project: dict = {
        "id": project_id,
        "name": name,
        "client": client,
        "due_date": due_date,
        "rfp_filename": file.filename or f"rfp{ext}",
        "status": "extracting",
        "questions": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await upsert_project(project)
    background_tasks.add_task(extract_project_questions, project_id, dest)
    return {**project, **_counts(project)}


@router.get("")
async def get_projects() -> list:
    projects = await list_projects()
    return sorted(
        [{**p, **_counts(p)} for p in projects],
        key=lambda p: p.get("created_at", ""),
        reverse=True,
    )


@router.get("/{project_id}")
async def get_project_detail(project_id: str) -> dict:
    p = await get_project(project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return {**p, **_counts(p)}


@router.post("/{project_id}/generate-all", status_code=status.HTTP_202_ACCEPTED)
async def generate_all(project_id: str, background_tasks: BackgroundTasks) -> dict:
    if not await get_project(project_id):
        raise HTTPException(404, "Project not found")
    background_tasks.add_task(generate_answers_for_project, project_id)
    return {"status": "scheduled", "project_id": project_id}
