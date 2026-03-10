"""
Property QA RAG Indexer

Builds a dedicated hybrid-search index (dense + sparse) for Property QA.
Run:
  python -m scripts.index_property_qa_rag
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from qdrant_client import models
from qdrant_client.http.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from config import (
    PROPERTIES_FILE,
    PROPERTY_QA_COLLECTION_NAME,
    PROPERTY_QA_DENSE_DIM,
    PROPERTY_QA_DENSE_VECTOR_NAME,
    PROPERTY_QA_SPARSE_MODEL,
    PROPERTY_QA_SPARSE_VECTOR_NAME,
)
from embeddings import embed_batch
from property_qa_rag import build_property_qa_document
from qdrant_client_singleton import get_qdrant


def index_property_qa_rag() -> None:
    print(f"Loading properties from {PROPERTIES_FILE}")
    with open(PROPERTIES_FILE) as f:
        data = json.load(f)
    properties = data.get("properties", [])
    if not properties:
        raise ValueError("No properties found in data file")
    print(f"Loaded {len(properties)} properties")

    documents = [build_property_qa_document(prop) for prop in properties]
    print("Embedding QA documents with OpenAI dense embeddings...")
    dense_vectors = embed_batch(documents)
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if PROPERTY_QA_COLLECTION_NAME in existing:
        print(f"Dropping existing collection '{PROPERTY_QA_COLLECTION_NAME}'")
        client.delete_collection(PROPERTY_QA_COLLECTION_NAME)

    client.create_collection(
        collection_name=PROPERTY_QA_COLLECTION_NAME,
        vectors_config={
            PROPERTY_QA_DENSE_VECTOR_NAME: VectorParams(
                size=PROPERTY_QA_DENSE_DIM,
                distance=Distance.COSINE,
                on_disk=True,
            )
        },
        sparse_vectors_config={
            PROPERTY_QA_SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
        },
    )
    client.create_payload_index(
        collection_name=PROPERTY_QA_COLLECTION_NAME,
        field_name="available",
        field_schema=PayloadSchemaType.BOOL,
        wait=True,
    )
    print(f"Created collection '{PROPERTY_QA_COLLECTION_NAME}'")

    points: list[PointStruct] = []
    for idx, (prop, doc, dense_vector) in enumerate(zip(properties, documents, dense_vectors)):
        loc = prop.get("location", {})
        payload = {
            "prop_id": prop.get("id"),
            "name": prop.get("name"),
            "type": prop.get("type"),
            "city": loc.get("city", ""),
            "country": loc.get("country", ""),
            "neighborhood": loc.get("neighborhood", ""),
            "price_per_night": prop.get("price_per_night"),
            "rating": prop.get("rating"),
            "review_count": prop.get("review_count"),
            "amenities": prop.get("amenities", []),
            "vibe_tags": prop.get("vibe_tags", []),
            "description": prop.get("description", ""),
            "review_highlights": prop.get("review_highlights", []),
            "available": prop.get("available", True),
            "qa_document": doc,
        }
        points.append(
            PointStruct(
                id=idx,
                payload=payload,
                vector={
                    PROPERTY_QA_DENSE_VECTOR_NAME: dense_vector,
                PROPERTY_QA_SPARSE_VECTOR_NAME: models.Document(
                    text=doc,
                    model=PROPERTY_QA_SPARSE_MODEL,
                ),
                },
            )
        )

    batch_size = 10
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=PROPERTY_QA_COLLECTION_NAME,
            points=points[i : i + batch_size],
            wait=True,
        )

    count = client.count(collection_name=PROPERTY_QA_COLLECTION_NAME).count
    print(f"Indexed {count} QA points in '{PROPERTY_QA_COLLECTION_NAME}'")


if __name__ == "__main__":
    index_property_qa_rag()
