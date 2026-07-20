from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.config import settings
from app.storage import get_document, list_documents, remove_document, upsert_document

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXT = {".pdf", ".docx", ".txt", ".md"}


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    doc_id = str(uuid.uuid4())
    dest = os.path.join(settings.UPLOAD_DIR, f"{doc_id}{ext}")

    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)

    doc: dict = {
        "id": doc_id,
        "filename": file.filename or f"{doc_id}{ext}",
        "file_type": ext.lstrip("."),
        "status": "ingesting",
        "chunk_count": 0,
        "cognee_ref": None,
        "error_msg": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await upsert_document(doc)

    from app.services.ingestion import ingest_document

    background_tasks.add_task(ingest_document, doc_id, dest)
    return doc


@router.get("")
async def get_documents() -> list:
    docs = await list_documents()
    return sorted(docs, key=lambda d: d.get("created_at", ""), reverse=True)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str) -> None:
    if not await get_document(doc_id):
        raise HTTPException(404, "Document not found")
    await remove_document(doc_id)
