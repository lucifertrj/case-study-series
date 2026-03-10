from __future__ import annotations

import os

from openai import OpenAI
from qdrant_client import models
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
)

from config import (
    INTENT_MODEL,
    PROPERTY_QA_COLLECTION_NAME,
    PROPERTY_QA_DENSE_VECTOR_NAME,
    PROPERTY_QA_SPARSE_MODEL,
    PROPERTY_QA_SPARSE_VECTOR_NAME,
)
from embeddings import embed
from qdrant_client_singleton import get_qdrant

_DEFAULT_TOP_K = 5
_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client



def retrieve_property_context(question: str, top_k: int = _DEFAULT_TOP_K) -> list[dict]:
    client = get_qdrant()
    dense_query = embed(question)
    availability_filter = Filter(
        must=[FieldCondition(key="available", match=MatchValue(value=True))]
    )

    response = client.query_points(
        collection_name=PROPERTY_QA_COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_query,
                using=PROPERTY_QA_DENSE_VECTOR_NAME,
                filter=availability_filter,
                limit=max(10, top_k * 3),
            ),
            Prefetch(
                query=models.Document(text=question, model=PROPERTY_QA_SPARSE_MODEL),
                using=PROPERTY_QA_SPARSE_VECTOR_NAME,
                filter=availability_filter,
                limit=max(10, top_k * 3),
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        with_payload=True,
        limit=top_k,
    ).points

    hits = []
    for point in response:
        payload = point.payload or {}
        hits.append(
            {
                "id": str(payload.get("prop_id", point.id)),
                "name": payload.get("name", ""),
                "city": payload.get("city", ""),
                "score": float(point.score or 0.0),
                "qa_document": payload.get("qa_document", ""),
                "payload": payload,
            }
        )
    return hits


def answer_property_question(question: str, top_k: int = _DEFAULT_TOP_K) -> dict:
    hits = retrieve_property_context(question, top_k=top_k)
    if not hits:
        return {"answer": "I couldn't find relevant properties for that question.", "sources": []}

    context_parts = []
    for i, hit in enumerate(hits, start=1):
        context_parts.append(
            f"[Source {i}] {hit['name']} ({hit['city']})\n{hit['qa_document']}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a hotel/property assistant at Booking com\n"
        "Answer the question using ONLY the provided sources.\n"
        "If the sources are insufficient, say what is missing.\n"
        "Keep the answer concise and practical.\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n{context}"
    )

    completion = _get_openai_client().chat.completions.create(
        model=INTENT_MODEL,
        messages=[
            {"role": "system", "content": "Ground answers strictly in retrieved property data."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    answer = completion.choices[0].message.content or ""

    sources = [
        {
            "id": hit["id"],
            "name": hit["name"],
            "city": hit["city"],
            "score": hit["score"],
        }
        for hit in hits
    ]
    return {"answer": answer.strip(), "sources": sources}
