from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.generation import generate_answer
from app.storage import find_question, update_question

router = APIRouter(tags=["questions"])


class StatusUpdate(BaseModel):
    status: Literal["unanswered", "draft", "approved"]


class ContentUpdate(BaseModel):
    content: str


@router.post("/questions/{question_id}/generate")
async def generate_one(question_id: str) -> dict:
    proj, question = await find_question(question_id)
    if not proj:
        raise HTTPException(404, "Question not found")

    answer_text, sources = await generate_answer(question["text"])

    await update_question(proj["id"], question_id, {
        "status": "draft",
        "answer": {
            "content": answer_text,
            "generated_by": "llm",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
        },
    })

    _, q = await find_question(question_id)
    return q  # type: ignore[return-value]


@router.patch("/questions/{question_id}")
async def update_question_status(question_id: str, body: StatusUpdate) -> dict:
    proj, question = await find_question(question_id)
    if not proj:
        raise HTTPException(404, "Question not found")

    await update_question(proj["id"], question_id, {"status": body.status})
    _, q = await find_question(question_id)
    return q  # type: ignore[return-value]


@router.patch("/answers/{question_id}")
async def update_answer(question_id: str, body: ContentUpdate) -> dict:
    proj, question = await find_question(question_id)
    if not proj:
        raise HTTPException(404, "Question not found")

    existing = question.get("answer") or {}
    prior_gen = existing.get("generated_by", "human")
    generated_by = "mixed" if prior_gen == "llm" else prior_gen

    await update_question(proj["id"], question_id, {
        "status": "draft" if question.get("status") == "unanswered" else question.get("status"),
        "answer": {
            **existing,
            "content": body.content,
            "generated_by": generated_by,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    })
    _, q = await find_question(question_id)
    return (q or {}).get("answer") or {}  # type: ignore[return-value]


@router.get("/answers/{question_id}")
async def get_answer(question_id: str) -> dict:
    _, question = await find_question(question_id)
    if not question:
        raise HTTPException(404, "Question not found")
    answer = question.get("answer")
    if not answer:
        raise HTTPException(404, "No answer for this question yet")
    return answer
