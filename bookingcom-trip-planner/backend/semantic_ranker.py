from __future__ import annotations
import logging
from qdrant_client.http.models import (
    Filter, FieldCondition, Range, MatchValue,
)
from schemas import FilterSchema
from embeddings import embed
from qdrant_client_singleton import get_qdrant
from config import COLLECTION_NAME, QDRANT_TOP_K

logger = logging.getLogger(__name__)

# Known cities in the database with common variations
CITY_MAPPINGS = {
    "amsterdam": "Amsterdam",
    "paris": "Paris",
    "bali": "Bali",
    "tokyo": "Tokyo",
    "kyoto": "Kyoto",
    "barcelona": "Barcelona",
    "rome": "Rome",
    "new york": "New York",
    "ny": "New York",
    "nyc": "New York",
    "cape town": "Cape Town",
    "amalfi": "Amalfi",
    "santorini": "Santorini",
    "manali": "Manali",
}



def _build_query_text(intent: FilterSchema) -> str:
    parts = []
    if intent.destination:
        parts.append(f"in {intent.destination}")
    if intent.location_vibe:
        parts.append(intent.location_vibe)
    if intent.property_type:
        parts.append(" or ".join(intent.property_type) + " property")
    if intent.vibe:
        parts.append(", ".join(intent.vibe) + " atmosphere")
    if intent.amenities:
        parts.append("with " + ", ".join(intent.amenities))
    if intent.special_needs:
        parts.append(", ".join(intent.special_needs))
    return " ".join(parts) if parts else "comfortable hotel with great amenities"


def _build_qdrant_filters(intent: FilterSchema) -> Filter | None:
    """Build Qdrant filters with only hard constraints."""
    must_conditions = []

    # Always filter for availability
    must_conditions.append(
        FieldCondition(key="available", match=MatchValue(value=True))
    )

    # Location filter - strict match on city
    if intent.destination:
        # Normalize destination using mappings
        dest_lower = intent.destination.strip().lower()
        city = CITY_MAPPINGS.get(dest_lower)

        # Fallback to title case if no mapping found
        if not city:
            city = intent.destination.strip().title()

        must_conditions.append(
            FieldCondition(key="city", match=MatchValue(value=city))
        )
        logger.info(f"Applied city filter: '{city}' (from input: '{intent.destination}')")

    # Budget range - strict
    if intent.budget_range:
        min_p, max_p = intent.budget_range
        # Add 20% buffer to budget for flexibility
        buffer = (max_p - min_p) * 0.2
        must_conditions.append(
            FieldCondition(
                key="price_per_night",
                range=Range(gte=max(0, min_p - buffer), lte=max_p + buffer),
            )
        )

    return Filter(must=must_conditions)



def semantic_search(intent: FilterSchema) -> list[dict]:
    """Dense search with query fallback if filters fail."""
    query_text = _build_query_text(intent)
    qdrant_filter = _build_qdrant_filters(intent)
    client = get_qdrant()
    dense_query = embed(query_text)

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_query,
            query_filter=qdrant_filter,
            limit=QDRANT_TOP_K,
            with_payload=True,
            using="dense",
        ).points
    except Exception as e:
        logger.error(f"Dense search failed: {e}")
        raise

    output = []
    for point in results:
        output.append({
            "payload": point.payload,
            "semantic_score": point.score,
            "id": point.id,
        })

    return output
