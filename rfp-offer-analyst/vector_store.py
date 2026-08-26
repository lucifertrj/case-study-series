import uuid
from typing import Any, Dict, List

from qdrant_client import models

from client import get_embedding_client, get_qdrant_client
from config import config


def _offer_filter(offer_id: int) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="offer_id", match=models.MatchValue(value=offer_id))]
    )


def is_offer_indexed(offer_id: int) -> bool:
    client = get_qdrant_client()
    count = client.count(
        collection_name=config.vector.collection_name,
        count_filter=_offer_filter(offer_id),
        exact=True,
    )
    return count.count > 0


def delete_offer_chunks(offer_id: int) -> None:
    client = get_qdrant_client()
    client.delete(
        collection_name=config.vector.collection_name,
        points_selector=models.FilterSelector(filter=_offer_filter(offer_id)),
    )


def index_offer_chunks(offer_id: int, pages: List[List[Any]]) -> None:
    client = get_qdrant_client()
    embedder = get_embedding_client()
    points = []

    for page_num, text in pages:
        if not text or not str(text).strip():
            continue

        clean_txt = str(text).strip()
        chunks = [
            clean_txt[i : i + config.vector.chunk_size]
            for i in range(0, len(clean_txt), config.vector.chunk_stride)
        ]

        for chunk_idx, chunk in enumerate(chunks):
            vectors = list(embedder.embed([chunk]))
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"offer:{offer_id}:page:{page_num}:chunk:{chunk_idx}")),
                    vector=vectors[0].tolist(),
                    payload={
                        "offer_id": offer_id,
                        "page": int(page_num),
                        "chunk": chunk_idx,
                        "text": chunk,
                    },
                )
            )

    if points:
        client.upsert(collection_name=config.vector.collection_name, points=points)


def reindex_offer_chunks(offer_id: int, pages: List[List[Any]]) -> None:
    delete_offer_chunks(offer_id)
    index_offer_chunks(offer_id, pages)


def search_offer(offer_id: int, query: str, limit: int | None = None) -> List[Dict[str, Any]]:
    client = get_qdrant_client()
    embedder = get_embedding_client()

    vectors = list(embedder.embed([query]))
    res = client.query_points(
        collection_name=config.vector.collection_name,
        query=vectors[0].tolist(),
        query_filter=_offer_filter(offer_id),
        limit=limit or config.vector.search_limit,
    )

    return [
        {
            "text": pt.payload.get("text", ""),
            "page": pt.payload.get("page", 1),
            "score": pt.score,
        }
        for pt in res.points
    ]
