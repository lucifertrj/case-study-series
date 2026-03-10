"""
Property Indexing Script with Hybrid Search (Dense + Sparse BM25)
Loads data/properties.json, embeds each property, and upserts into Qdrant.
Run once: python -m scripts.index_properties
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from qdrant_client import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, SparseVectorParams, Modifier, PayloadSchemaType,
)
from config import (
    COLLECTION_NAME, PROPERTIES_FILE,
    EMBEDDING_DIM,
)
from embeddings import embed_batch
from qdrant_client_singleton import get_qdrant


def build_property_document(prop: dict) -> str:
    """Create a rich text representation of a property for embedding.
    
    Key insight: Review highlights carry implicit semantic signals 
    (e.g. 'watched sunset from the balcony' → matches 'scenic views').
    """
    loc = prop.get("location", {})
    reviews = " | ".join(prop.get("review_highlights", []))
    amenities_str = ", ".join(prop.get("amenities", []))
    vibe_str = ", ".join(prop.get("vibe_tags", []))

    return (
        f"Property: {prop['name']} — {prop['type']} in {loc.get('city', '')} ({loc.get('neighborhood', '')})\n"
        f"Location vibe: {loc.get('location_vibe', '')}\n"
        f"Description: {prop['description']}\n"
        f"Amenities: {amenities_str}\n"
        f"Vibe: {vibe_str}\n"
        f"Price: ${prop['price_per_night']}/night | Rating: {prop['rating']}/10 | Reviews: {prop['review_count']}\n"
        f"What guests love: {reviews}"
    )


def index_properties():
    print(f"📂 Loading properties from {PROPERTIES_FILE}")
    with open(PROPERTIES_FILE) as f:
        data = json.load(f)
    properties = data["properties"]
    print(f"✅ Loaded {len(properties)} properties")

    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"🗑  Dropping existing collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)

    # Create collection with hybrid search support (dense + sparse BM25)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
                on_disk=True,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF)
        },
    )
    print(f"✅ Created Qdrant collection '{COLLECTION_NAME}' with hybrid search (dense + BM25)")

    docs = [build_property_document(p) for p in properties]
    print(f"📝 Built {len(docs)} property documents")

    BATCH_SIZE = 20
    all_vectors = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1}/{(len(docs) - 1) // BATCH_SIZE + 1} ({len(batch)} docs)...")
        vectors = embed_batch(batch)
        all_vectors.extend(vectors)
        if i + BATCH_SIZE < len(docs):
            time.sleep(0.5)  

    print(f"✅ Embedded {len(all_vectors)} property vectors")

    # ── Build Qdrant points with hybrid vectors (dense + sparse BM25) ──────
    points = []
    for idx, (prop, doc, vector) in enumerate(zip(properties, docs, all_vectors)):
        loc = prop.get("location", {})
        payload = {
            "prop_id": prop["id"],
            "name": prop["name"],
            "type": prop["type"],
            "city": loc.get("city", ""),
            "country": loc.get("country", ""),
            "neighborhood": loc.get("neighborhood", ""),
            "location_vibe": loc.get("location_vibe", ""),
            "price_per_night": prop["price_per_night"],
            "rating": prop["rating"],
            "review_count": prop["review_count"],
            "amenities": prop["amenities"],
            "vibe_tags": prop["vibe_tags"],
            "description": prop["description"],
            "review_highlights": prop["review_highlights"],
            "available": prop["available"],
        }
        points.append(
            PointStruct(
                id=idx,
                vector={
                    "dense": vector,
                    "sparse": models.Document(text=doc, model="Qdrant/bm25"),
                },
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Upserted {len(points)} points into Qdrant")

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"✅ Validation: {count} points indexed")

    print("\n🎉 Indexing complete! Run the backend: cd backend && uvicorn main:app --reload")


if __name__ == "__main__":
    index_properties()
