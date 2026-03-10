from __future__ import annotations

import os

from openai import OpenAI

from config import INTENT_MODEL

_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _format_intent(intent: dict) -> str:
    if not intent:
        return "None"
    parts = []
    for key, value in intent.items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"- {key}: {value}")
    return "\n".join(parts) if parts else "None"


def _format_properties(top_properties: list[dict]) -> str:
    if not top_properties:
        return "No shortlisted properties were provided."

    lines = []
    for idx, prop in enumerate(top_properties[:5], start=1):
        price = prop.get("price_per_night")
        price_text = f"${price}/night" if price is not None else "price unavailable"
        lines.append(
            f"{idx}. {prop.get('name', 'Unknown property')} in {prop.get('city', 'Unknown city')}, "
            f"{prop.get('country', '')} | {price_text} | rating {prop.get('rating', 'n/a')} | "
            f"match reason: {prop.get('match_reason', 'n/a')}"
        )
    return "\n".join(lines)


def build_itinerary_system_prompt(
    *,
    search_query: str,
    parsed_intent: dict,
    memory_summary: str | None,
    preferences_summary: str | None,
    top_properties: list[dict],
) -> str:
    return (
        "You are Booking.com's itinerary planner.\n"
        "Your job is to create practical, day-by-day travel itineraries grounded in the provided search context.\n"
        "Do not claim you looked up live data, transit times, opening hours, or reservations.\n"
        "Use the memory and preference context exactly as given; do not infer hidden memories or rerun retrieval.\n"
        "Prefer the shortlisted properties when recommending where the traveler should stay.\n"
        "When the user asks for an itinerary, provide a structured plan with morning, afternoon, evening, food guidance, and why it fits them.\n"
        "If dietary preferences or custom interests are present, incorporate them concretely into meal/activity suggestions.\n"
        "If something important is missing, ask a concise follow-up question instead of inventing specifics.\n\n"
        "Search query:\n"
        f"{search_query}\n\n"
        "Parsed search intent:\n"
        f"{_format_intent(parsed_intent)}\n\n"
        "Memory context extracted during search:\n"
        f"{memory_summary or 'None'}\n\n"
        "Saved onboarding preferences:\n"
        f"{preferences_summary or 'None'}\n\n"
        "Top matched properties from search:\n"
        f"{_format_properties(top_properties)}"
    )


def generate_itinerary_reply(
    *,
    message: str,
    search_query: str,
    parsed_intent: dict,
    memory_summary: str | None,
    preferences_summary: str | None,
    top_properties: list[dict],
    history: list[dict],
) -> str:
    system_prompt = build_itinerary_system_prompt(
        search_query=search_query,
        parsed_intent=parsed_intent,
        memory_summary=memory_summary,
        preferences_summary=preferences_summary,
        top_properties=top_properties,
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": message})

    completion = _get_openai_client().chat.completions.create(
        model=INTENT_MODEL,
        messages=messages,
        temperature=0.4,
    )
    return (completion.choices[0].message.content or "").strip()
