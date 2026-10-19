"""Shared Qdrant Cloud document-store factory.

Both the indexing pipeline and the live RAG app must point at the *same* Qdrant
Cloud collection, so the construction lives in one place.
"""

from functools import lru_cache

from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from .config import Settings

# text-embedding-3-small returns 1536-dim vectors.
EMBEDDING_DIM = 1536


@lru_cache(maxsize=None)
def _store_cached(
    url: str,
    api_key: str,
    collection_name: str,
    recreate: bool,
) -> QdrantDocumentStore:
    return QdrantDocumentStore(
        url=url,
        api_key=Secret.from_token(api_key) if api_key else None,
        index=collection_name,
        embedding_dim=EMBEDDING_DIM,
        recreate_index=recreate,
        return_embedding=True,
        wait_result_from_api=True,
    )


def get_document_store(
    settings: Settings, recreate: bool = False
) -> QdrantDocumentStore:
    """Open (or lazily reuse) the Qdrant Cloud collection for this project.

    Pass ``recreate=True`` only from the indexing entry point; the live app must
    always attach to the existing collection.
    """
    return _store_cached(
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.collection_name,
        recreate,
    )
