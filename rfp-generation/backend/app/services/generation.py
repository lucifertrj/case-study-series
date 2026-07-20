from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.services.retrieval import retrieve
from app.storage import get_project, update_question, upsert_project

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.LLM_API_KEY)
    return _client

GENERATION_SYSTEM = (
    "You are a company representative responding to an RFP on behalf of your organization. "
    "Answer using ONLY the provided context below. "
    "Write in first-person plural (we, our, us). Be specific, professional, and concise. "
    "If the context does not contain sufficient information, respond with exactly: "
    "KNOWLEDGE_GAP: <brief description of what is missing>. "
    "Do NOT hallucinate or invent facts not present in the context."
)


def _build_messages(question_text: str, chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    if chunks:
        blocks = []
        for i, c in enumerate(chunks, 1):
            doc = c.get("document_name") or "unknown"
            loc = c.get("locator") or ""
            header = f"[Source {i} | doc={doc}" + (f" | loc={loc}" if loc else "") + "]"
            blocks.append(f"{header}\n{c.get('chunk_text', '')}")
        context = "\n\n---\n\n".join(blocks)
    else:
        context = "(no relevant knowledge-base context was retrieved)"

    return [
        {"role": "system", "content": GENERATION_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question_text}\n\nWrite the answer now."},
    ]


async def generate_answer(question_text: str) -> tuple[str, list[dict[str, Any]]]:
    """Return (answer_text, sources_list)."""
    chunks = await retrieve(question_text)

    client = get_openai_client()
    resp = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0.2,
        messages=_build_messages(question_text, chunks),
    )
    content = (resp.choices[0].message.content or "").strip()

    sources = [
        {
            "document_name": c.get("document_name"),
            "chunk_text": c.get("chunk_text", "")[:8000],
            "locator": c.get("locator") or "",
            "score": float(c.get("score") or 0.0),
        }
        for c in chunks
    ]
    return content, sources


async def generate_answers_for_project(project_id: str) -> None:
    """Background task: generate drafts for every unanswered question."""
    project = await get_project(project_id)
    if not project:
        return

    for question in project.get("questions", []):
        if question.get("status") != "unanswered":
            continue
        try:
            answer_text, sources = await generate_answer(question["text"])
            await update_question(project_id, question["id"], {
                "status": "draft",
                "answer": {
                    "content": answer_text,
                    "generated_by": "llm",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "sources": sources,
                },
            })
        except Exception:
            logger.exception("Failed to generate answer for question %s", question["id"])
