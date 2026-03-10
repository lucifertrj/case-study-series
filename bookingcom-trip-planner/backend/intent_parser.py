from __future__ import annotations
import os
from openai import OpenAI
from schemas import FilterSchema
from config import INTENT_MODEL

_client: OpenAI | None = None

_SYSTEM_PROMPT = """You are a travel intent extractor for a hotel/property search engine.

Parse the user's natural language travel query into a structured set of search filters.

Rules:
- destination: extract city/region if mentioned (e.g. "Paris", "Bali coast", "near Amalfi")
- vibe: pick from [romantic, adventure, family, wellness, party, luxury, budget, cultural, quiet, scenic] — can be multiple
- amenities: extract specific amenities, including IMPLICIT ones:
    "sunset views" → ["scenic views", "balcony or terrace"]
    "good gym" → ["gym", "fitness center"]
    "infinity pool" → ["swimming pool", "infinity pool"]
    "pet friendly" → ["pet friendly"]
- property_type: pick from [hotel, boutique, villa, resort] — can be multiple
- budget_range: (min, max) per night in USD if budget is mentioned (e.g. "under $200" → (0, 200), "around $300" → (250, 350))
- special_needs: any accessibility, crib, parking, pet, or specific requirements
- location_vibe: one of [near beach, city center, countryside, waterfront, riverside, hilltop, old town, beachfront] if mentioned

Be generous with vibe and amenity inference — users rarely say exactly what they want."""


def extract_intent(user_query: str) -> FilterSchema:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = _client.beta.chat.completions.parse(
        model=INTENT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        response_format=FilterSchema,
        temperature=0.1,
    )
    return response.choices[0].message.parsed
