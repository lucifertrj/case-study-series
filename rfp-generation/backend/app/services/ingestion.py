from __future__ import annotations

import logging
import os
from pathlib import Path

from app.config import settings

from app.storage import get_document, upsert_document

logger = logging.getLogger(__name__)


def configure_cognee_env() -> None:
    """Configure cognee before the first lazy import."""
    os.environ.setdefault("VECTOR_DB_PROVIDER", "qdrant")
    os.environ["VECTOR_DB_URL"] = settings.QDRANT_URL
    os.environ["VECTOR_DB_KEY"] = settings.QDRANT_API_KEY
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ["LLM_API_KEY"] = settings.LLM_API_KEY
    os.environ["OPENAI_API_KEY"] = settings.LLM_API_KEY
    os.environ.setdefault("LLM_MODEL", settings.LLM_MODEL)
    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
    os.environ.setdefault("EMBEDDING_API_KEY", settings.LLM_API_KEY)


def get_cognee():
    configure_cognee_env()
    import cognee

    return cognee


def _estimate_chunks(file_path: str) -> int:
    try:
        return max(1, os.path.getsize(file_path) // 3200)
    except Exception:
        return 1


async def ingest_document(doc_id: str, file_path: str) -> None:
    try:
        logger.info("Ingesting document %s at %s", doc_id, file_path)

        cognee = get_cognee()
        await cognee.add([file_path], dataset_name=settings.COGNEE_DATASET)
        await cognee.cognify([settings.COGNEE_DATASET])

        doc = await get_document(doc_id)
        if doc:
            doc.update({
                "status": "indexed",
                "chunk_count": _estimate_chunks(file_path),
                "cognee_ref": Path(file_path).name,
            })
            await upsert_document(doc)

        logger.info("Ingestion complete for %s", doc_id)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", doc_id)
        doc = await get_document(doc_id)
        if doc:
            doc.update({"status": "failed", "error_msg": str(exc)[:2000]})
            await upsert_document(doc)
