from __future__ import annotations
import os
import logging
import json
from openai import OpenAI
from schemas import FilterSchema
from config import INTENT_MODEL

logger = logging.getLogger(__name__)
_client: OpenAI | None = None


def _coerce_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        preferred = value.get("search_result")
        if preferred is not None and preferred is not value:
            return _coerce_to_text(preferred)
        try:
            return json.dumps(value, ensure_ascii=True)
        except Exception:
            return str(value)
    if isinstance(value, list):
        parts = [_coerce_to_text(item).strip() for item in value]
        parts = [p for p in parts if p]
        return "\n".join(parts)
    return str(value)

async def get_user_dna(user_id: str) -> str:
    if user_id == "anonymous":
        return ""
    try:
        import cognee
        answers = await cognee.search(
            query_text=f"What does user {user_id} prefer when traveling? "
                       f"What types of properties, vibes, amenities, and budgets do they gravitate toward?",
            query_type=cognee.SearchType.GRAPH_COMPLETION,
        )
        if answers:
            result = answers[0]
            # Handle if result is a list
            if isinstance(result, list):
                if len(result) > 0:
                    result = result[0]
                else:
                    return ""
            return _coerce_to_text(result)
    except Exception as e:
        logger.warning(f"Cognee retrieval failed for user {user_id}: {e}")
    return ""



async def record_signal(user_id: str, signal_type: str, data: dict) -> None:
    if user_id == "anonymous":
        return
    try:
        import cognee
        if signal_type == "onboarding":
            text = (
                f"User {user_id} onboarding preferences: "
                f"travel_style={data.get('travel_style', [])}; "
                f"property_preferences={data.get('property_preferences', [])}; "
                f"typical_budget={data.get('typical_budget')}; "
                f"must_have_amenities={data.get('must_have_amenities', [])}; "
                f"preferred_vibes={data.get('preferred_vibes', [])}; "
                f"typical_companions={data.get('typical_companions', [])}; "
                f"accessibility_needs={data.get('accessibility_needs', [])}; "
                f"location_preferences={data.get('location_preferences', [])}; "
                f"custom_interests={data.get('custom_interests')}; "
                f"dietary_preferences={data.get('dietary_preferences', [])}."
            )
            node_sets = [f"user_{user_id}", "preferences"]
        elif signal_type == "click":
            text = (
                f"User {user_id} showed interest in: {data.get('name')} "
                f"({data.get('type')}) in {data.get('location', {}).get('city', '')}. "
                f"Amenities: {data.get('amenities')}. Vibe: {data.get('vibe_tags')}. "
                f"Price: ${data.get('price_per_night')}/night. Rating: {data.get('rating')}."
            )
            node_sets = [f"user_{user_id}", "behavior"]
        elif signal_type == "book":
            text = (
                f"User {user_id} BOOKED (strong preference signal): {data.get('name')} "
                f"({data.get('type')}) in {data.get('location', {}).get('city', '')}. "
                f"Price: ${data.get('price_per_night')}/night. Rating: {data.get('rating')}. "
                f"Amenities: {data.get('amenities')}. Vibe: {data.get('vibe_tags')}."
            )
            node_sets = [f"user_{user_id}", "bookings"]
        else:
            return

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await cognee.add(text, node_set=node_sets)
        await cognee.cognify()
        logger.info(f"Recorded {signal_type} signal for user {user_id}")
    except Exception as e:
        logger.warning(f"Cognee signal recording failed: {e}")


async def enrich_with_memory(
    intent: FilterSchema,
    user_id: str,
) -> tuple[FilterSchema, str]:
    user_dna = _coerce_to_text(await get_user_dna(user_id)).strip()

    if not user_dna:
        return intent, ""

    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    enrichment_prompt = f"""The user submitted a travel search with the following structured intent:
{intent.model_dump_json(indent=2)}

The user's known travel preferences from their history:
{user_dna}

Task: Enhance the filter intent by incorporating the user's preferences WHERE they don't conflict with the explicit current search intent.
- Add implicit amenity preferences from history if not already specified
- Add vibe tags that match the user's pattern if not already specified  
- If budget_range is not set and user has a budget pattern, set it
- Do NOT override explicitly stated preferences
- Do NOT add preferences that contradict the current query

Return an enriched FilterSchema."""

    try:
        response = _client.beta.chat.completions.parse(
            model=INTENT_MODEL,
            messages=[{"role": "user", "content": enrichment_prompt}],
            response_format=FilterSchema,
            temperature=0.1,
        )
        enriched = response.choices[0].message.parsed
        summary = f"Personalized based on your travel history: {user_dna[:200]}..."
        return enriched, summary
    except Exception as e:
        logger.warning(f"Memory enrichment failed: {e}")
        return intent, ""
