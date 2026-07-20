from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from app.config import settings

logger = logging.getLogger(__name__)

_openai: AsyncOpenAI | None = None
_qdrant: AsyncQdrantClient | None = None


def get_openai_client() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=settings.LLM_API_KEY)
    return _openai


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            check_compatibility=False,
        )
    return _qdrant


def _to_result(chunk_text: str, locator: str | None, score: float, doc: str | None) -> dict[str, Any]:
    return {
        "chunk_text": chunk_text,
        "locator": locator,
        "score": float(score or 0.0),
        "document_name": doc,
    }


async def _cognee_search(query: str) -> list[dict[str, Any]]:
    """Try cognee's high-level search API in a few modes."""
    from app.services.ingestion import get_cognee

    cognee = get_cognee()
    try:
        from cognee.api.v1.search import SearchType  # type: ignore
    except Exception:
        return []

    results: list[dict[str, Any]] = []

    # Try each available search type; skip missing enums gracefully.
    for mode_name in ("CHUNKS", "INSIGHTS", "SUMMARIES"):
        mode = getattr(SearchType, mode_name, None)
        if mode is None:
            continue
        try:
            raw = await cognee.search(query_type=mode, query_text=query)
        except TypeError:
            try:
                raw = await cognee.search(query, search_type=mode)
            except Exception as exc:  # pragma: no cover
                logger.debug("cognee.search(%s) failed: %s", mode_name, exc)
                continue
        except Exception as exc:  # pragma: no cover
            logger.debug("cognee.search(%s) failed: %s", mode_name, exc)
            continue

        for item in raw or []:
            text: str | None = None
            locator: str | None = None
            score: float = 0.0
            doc: str | None = None

            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("chunk")
                    or item.get("payload", {}).get("text")
                    or item.get("content")
                )
                locator = item.get("locator") or item.get("id")
                score = float(item.get("score") or item.get("similarity") or 0.0)
                payload = item.get("payload") or {}
                doc = (
                    item.get("document_name")
                    or item.get("source")
                    or payload.get("source")
                    or payload.get("file_name")
                )
            else:
                text = str(item)

            if text:
                results.append(_to_result(text, locator, score, doc))

        if results:
            break

    return results[: settings.TOP_K]


async def _qdrant_fallback(query: str) -> list[dict[str, Any]]:
    """Fallback: embed query with OpenAI, search every Qdrant collection."""
    openai = get_openai_client()
    qdrant = get_qdrant_client()

    try:
        emb = await openai.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        vector = emb.data[0].embedding
    except Exception as exc:  # pragma: no cover
        logger.exception("OpenAI embedding failed: %s", exc)
        return []

    try:
        cols = await qdrant.get_collections()
        collection_names = [c.name for c in cols.collections]
    except Exception as exc:  # pragma: no cover
        logger.exception("Qdrant collection listing failed: %s", exc)
        return []

    aggregated: list[dict[str, Any]] = []
    for name in collection_names:
        try:
            hits = await qdrant.search(
                collection_name=name,
                query_vector=vector,
                limit=settings.TOP_K,
                with_payload=True,
            )
        except Exception as exc:
            logger.debug("Qdrant search on %s failed: %s", name, exc)
            continue

        for hit in hits:
            payload = hit.payload or {}
            text = (
                payload.get("text")
                or payload.get("chunk")
                or payload.get("content")
                or payload.get("page_content")
            )
            if not text:
                continue
            doc = (
                payload.get("source")
                or payload.get("file_name")
                or payload.get("document_name")
            )
            locator = str(hit.id) if hit.id is not None else None
            aggregated.append(
                _to_result(str(text), locator, float(hit.score or 0.0), doc)
            )

    aggregated.sort(key=lambda r: r["score"], reverse=True)
    return aggregated[: settings.TOP_K]


async def retrieve(query: str) -> list[dict[str, Any]]:
    """Retrieve up to TOP_K relevant chunks for a question."""
    if not query or not query.strip():
        return []

    try:
        results = await _cognee_search(query)
    except Exception as exc:  # pragma: no cover
        logger.exception("cognee search errored, falling back: %s", exc)
        results = []

    if results:
        return results

    return await _qdrant_fallback(query)
