from __future__ import annotations
from qdrant_client import QdrantClient
from config import QDRANT_ENDPOINT, QDRANT_API_KEY

_qdrant: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=QDRANT_ENDPOINT,
            api_key=QDRANT_API_KEY,
        )
    return _qdrant
