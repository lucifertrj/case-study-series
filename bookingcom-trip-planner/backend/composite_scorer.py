from __future__ import annotations
import os
import logging
from openai import OpenAI
from schemas import FilterSchema, PropertyResult
from config import WEIGHTS, FINAL_TOP_K, INTENT_MODEL

logger = logging.getLogger(__name__)
_client: OpenAI | None = None


def _filter_score(payload: dict, intent: FilterSchema) -> float:
    scores = []

    if intent.amenities:
        prop_amenities = set(a.lower() for a in payload.get("amenities", []))
        query_amenities = set(a.lower() for a in intent.amenities)
        matched = sum(
            1 for qa in query_amenities
            if any(qa in pa or pa in qa for pa in prop_amenities)
        )
        scores.append(matched / max(len(query_amenities), 1))

    if intent.vibe:
        prop_vibes = set(v.lower() for v in payload.get("vibe_tags", []))
        query_vibes = set(v.lower() for v in intent.vibe)
        matched = len(query_vibes & prop_vibes)
        scores.append(matched / max(len(query_vibes), 1))

    if intent.location_vibe:
        prop_loc_vibe = payload.get("location_vibe", "").lower()
        query_loc = intent.location_vibe.lower()
        scores.append(1.0 if query_loc in prop_loc_vibe or prop_loc_vibe in query_loc else 0.0)

    return sum(scores) / max(len(scores), 1)


def _quality_score(payload: dict) -> float:
    rating = payload.get("rating", 0.0) / 10.0
    review_count = payload.get("review_count", 0)
    import math
    review_boost = min(math.log10(max(review_count, 1)) / 4, 0.1)
    return min(rating + review_boost, 1.0)


def _memory_alignment_score(payload: dict, user_memory: str) -> float:
    if not user_memory.strip():
        return 0.5

    memory_lower = user_memory.lower()
    score_factors = []

    prop_type = payload.get("type", "")
    if prop_type and prop_type.lower() in memory_lower:
        score_factors.append(1.0)
    else:
        score_factors.append(0.3)

    prop_vibes = payload.get("vibe_tags", [])
    vibe_hits = sum(1 for v in prop_vibes if v.lower() in memory_lower)
    score_factors.append(min(vibe_hits / max(len(prop_vibes), 1), 1.0))

    prop_amenities = payload.get("amenities", [])[:5]
    amenity_hits = sum(1 for a in prop_amenities if a.lower() in memory_lower)
    score_factors.append(min(amenity_hits / max(len(prop_amenities), 1), 1.0))

    return sum(score_factors) / len(score_factors)


def _generate_match_reason(payload: dict, intent: FilterSchema, composite_score: float) -> str:
    reasons = []

    matching_vibes = set(v.lower() for v in intent.vibe) & set(v.lower() for v in payload.get("vibe_tags", []))
    if matching_vibes:
        reasons.append(f"matches your {' & '.join(matching_vibes)} vibe")

    prop_amenities = set(a.lower() for a in payload.get("amenities", []))
    query_amenities = set(a.lower() for a in intent.amenities)
    matched_amenities = [
        qa for qa in query_amenities
        if any(qa in pa or pa in qa for pa in prop_amenities)
    ]
    if matched_amenities:
        reasons.append(f"has {', '.join(matched_amenities[:3])}")

    rating = payload.get("rating", 0)
    if rating >= 9.0:
        reasons.append(f"exceptional {rating}/10 rating")
    elif rating >= 8.5:
        reasons.append(f"highly rated at {rating}/10")

    if intent.location_vibe and intent.location_vibe.lower() in payload.get("location_vibe", "").lower():
        reasons.append(f"{payload.get('location_vibe')} location")

    if intent.destination:
        city = payload.get("city", "")
        if city and intent.destination.lower() in city.lower():
            reasons.append(f"in {city}")

    if not reasons:
        reasons.append(f"strong semantic match ({composite_score:.0%} relevance)")

    return " · ".join(reasons[:4]).capitalize()


def composite_score_and_rank(
    candidates: list[dict],
    intent: FilterSchema,
    user_memory: str = "",
) -> list[PropertyResult]:
    scored = []

    for candidate in candidates:
        payload = candidate["payload"]
        semantic_s = float(candidate.get("semantic_score", 0))

        filter_s = _filter_score(payload, intent)
        quality_s = _quality_score(payload)
        memory_s = _memory_alignment_score(payload, user_memory)

        final_score = (
            WEIGHTS["semantic"] * semantic_s
            + WEIGHTS["filter"]  * filter_s
            + WEIGHTS["quality"] * quality_s
            + WEIGHTS["memory"]  * memory_s
        )

        match_reason = _generate_match_reason(payload, intent, final_score)

        result = PropertyResult(
            id=payload.get("prop_id", str(candidate.get("id", ""))),
            name=payload.get("name", ""),
            type=payload.get("type", ""),
            city=payload.get("city", ""),
            country=payload.get("country", ""),
            neighborhood=payload.get("neighborhood", ""),
            location_vibe=payload.get("location_vibe", ""),
            price_per_night=payload.get("price_per_night", 0),
            rating=payload.get("rating", 0),
            review_count=payload.get("review_count", 0),
            amenities=payload.get("amenities", []),
            vibe_tags=payload.get("vibe_tags", []),
            description=payload.get("description", ""),
            review_highlights=payload.get("review_highlights", [])[:3],
            available=payload.get("available", True),
            composite_score=round(final_score, 4),
            semantic_score=round(semantic_s, 4),
            filter_score=round(filter_s, 4),
            quality_score=round(quality_s, 4),
            memory_score=round(memory_s, 4),
            match_reason=match_reason,
        )
        scored.append(result)

    scored.sort(key=lambda x: x.composite_score, reverse=True)
    return scored[:FINAL_TOP_K]
