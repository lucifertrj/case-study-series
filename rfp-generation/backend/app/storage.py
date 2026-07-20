from __future__ import annotations

import asyncio
from typing import Any

_lock = asyncio.Lock()

_documents: list[dict[str, Any]] = []
_projects: list[dict[str, Any]] = []


# ── documents ──────────────────────────────────────────────────────────────────

async def list_documents() -> list[dict[str, Any]]:
    async with _lock:
        return list(_documents)


async def get_document(doc_id: str) -> dict[str, Any] | None:
    async with _lock:
        return next((d for d in _documents if d["id"] == doc_id), None)


async def upsert_document(doc: dict[str, Any]) -> None:
    async with _lock:
        idx = next((i for i, d in enumerate(_documents) if d["id"] == doc["id"]), None)
        if idx is None:
            _documents.append(doc)
        else:
            _documents[idx] = doc


async def remove_document(doc_id: str) -> None:
    async with _lock:
        _documents[:] = [d for d in _documents if d["id"] != doc_id]


# ── projects ───────────────────────────────────────────────────────────────────

async def list_projects() -> list[dict[str, Any]]:
    async with _lock:
        return list(_projects)


async def get_project(project_id: str) -> dict[str, Any] | None:
    async with _lock:
        return next((p for p in _projects if p["id"] == project_id), None)


async def upsert_project(project: dict[str, Any]) -> None:
    async with _lock:
        idx = next((i for i, p in enumerate(_projects) if p["id"] == project["id"]), None)
        if idx is None:
            _projects.append(project)
        else:
            _projects[idx] = project


async def find_question(
    question_id: str,
) -> tuple[dict[str, Any], dict[str, Any]] | tuple[None, None]:
    async with _lock:
        for proj in _projects:
            for q in proj.get("questions", []):
                if q["id"] == question_id:
                    return proj, q
    return None, None


async def update_question(project_id: str, question_id: str, updates: dict[str, Any]) -> None:
    async with _lock:
        for proj in _projects:
            if proj["id"] != project_id:
                continue
            for q in proj.get("questions", []):
                if q["id"] == question_id:
                    q.update(updates)
                    return
