from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.storage import get_project, upsert_project

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.LLM_API_KEY)
    return _client

EXTRACTION_SYSTEM = (
    "You are an RFP analyst. Extract all questions and requirements from this "
    "RFP document. Return ONLY a JSON object with key 'questions' whose value is "
    "an array of {\"number\": <int>, \"text\": \"<self-contained question>\"} objects. "
    "Each item must be distinct and self-contained. No commentary, no markdown."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# ── File parsers ───────────────────────────────────────────────────────────────

def _parse_pdf(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _parse_docx(path: str) -> str:
    from docx import Document as D
    doc = D(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _parse_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)
    if ext in {".txt", ".md"}:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {ext}")


# ── LLM extraction ─────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 40_000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start + max_chars // 2:
                end = nl
        chunks.append(text[start:end])
        start = end
    return chunks


async def _extract_llm(text: str) -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    client = get_openai_client()
    for chunk in _chunk_text(text):
        resp = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": f"RFP document:\n\n{chunk}"},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = _JSON_RE.search(raw)
            obj = json.loads(m.group(0)) if m else {}

        items = obj.get("questions", []) if isinstance(obj, dict) else []
        for item in items:
            text_val = str(item.get("text", "")).strip()
            if text_val:
                all_items.append({"text": text_val})

    # Dedupe and renumber.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in all_items:
        key = item["text"].lower()
        if key not in seen:
            seen.add(key)
            out.append({"number": len(out) + 1, "text": item["text"]})
    return out


# ── Orchestration ──────────────────────────────────────────────────────────────

async def extract_project_questions(project_id: str, file_path: str) -> None:
    try:
        logger.info("Extracting questions for project %s", project_id)

        text = _parse_file(file_path)
        if not text.strip():
            raise ValueError("RFP file appears empty after parsing")

        items = await _extract_llm(text)
        if not items:
            raise ValueError("No questions extracted from RFP")

        questions = [
            {
                "id": str(uuid.uuid4()),
                "number": item["number"],
                "text": item["text"],
                "status": "unanswered",
                "answer": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for item in items
        ]

        project = await get_project(project_id)
        if project:
            project.update({"status": "ready", "questions": questions})
            await upsert_project(project)

        logger.info("Extracted %d questions for project %s", len(questions), project_id)

    except Exception as exc:
        logger.exception("Extraction failed for project %s", project_id)
        project = await get_project(project_id)
        if project:
            project.update({"status": "failed", "error_msg": str(exc)[:2000]})
            await upsert_project(project)
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
