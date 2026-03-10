from __future__ import annotations
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    SearchRequest,
    SearchResponse,
    SignalRequest,
    PropertyQARequest,
    PropertyQAResponse,
    OnboardingRequest,
    OnboardingResponse,
    PreferencesResponse,
    ItineraryChatRequest,
    ItineraryChatResponse,
)
from intent_parser import extract_intent
from memory_enricher import enrich_with_memory, record_signal
from semantic_ranker import semantic_search
from composite_scorer import composite_score_and_rank
from qdrant_client_singleton import get_qdrant
from config import COLLECTION_NAME, PROPERTIES_FILE
from property_qa import answer_property_question
from itinerary_planner import generate_itinerary_reply
from travel_preferences import (
    TravelPreferences,
    save_preferences,
    load_preferences,
    build_preference_summary,
    preferences_to_filter_boost,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client = get_qdrant()
        count = client.count(collection_name=COLLECTION_NAME).count
        logger.info(f"✅ Qdrant collection '{COLLECTION_NAME}' has {count} properties indexed")
        if count == 0:
            logger.warning("⚠️  Qdrant collection is empty — run: python -m scripts.index_properties")
    except Exception as e:
        logger.warning(f"⚠️  Qdrant not ready: {e}. Run: python -m scripts.index_properties")
    yield


app = FastAPI(
    title="Booking.com Smart Filter PoC",
    description="AI-powered hotel search with 4-stage pipeline: Intent Parser → Memory Enricher → Semantic Ranker → Composite Scorer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/search", response_model=SearchResponse)
async def smart_search(req: SearchRequest):
    logger.info(f"Search request: query='{req.query}', user='{req.user_id}'")

    try:
        intent = extract_intent(req.query)
        logger.info(f"Parsed intent: {intent.model_dump()}")
    except Exception as e:
        logger.error(f"Intent parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Intent parsing failed: {str(e)}")

    # Load user preferences if available
    user_prefs = None
    prefs_summary = None
    if req.user_id and req.user_id != "anonymous":
        user_prefs = load_preferences(req.user_id)
        if user_prefs:
            prefs_summary = build_preference_summary(user_prefs)
            logger.info(f"Loaded preferences for {req.user_id}: {prefs_summary}")

            # Apply preference boosts to intent
            boost_params = preferences_to_filter_boost(user_prefs)

            # Merge preferences with intent (prefer explicit intent over preferences)
            if not intent.amenities and boost_params.get("amenities"):
                intent.amenities.extend(boost_params["amenities"][:2])  # Add top 2 preferred amenities

            if not intent.vibe and boost_params.get("vibes"):
                intent.vibe.extend(boost_params["vibes"][:2])

            if not intent.budget_range and boost_params.get("budget_range"):
                intent.budget_range = list(boost_params["budget_range"])

    enriched_intent, memory_summary = await enrich_with_memory(intent, req.user_id)

    # Combine preference summary with memory summary
    if prefs_summary and not memory_summary:
        memory_summary = prefs_summary
    elif prefs_summary and memory_summary:
        memory_summary = f"{prefs_summary} | {memory_summary}"

    memory_used = bool(memory_summary)

    candidates = semantic_search(enriched_intent)
    logger.info(f"Semantic search returned {len(candidates)} candidates")

    if not candidates:
        return SearchResponse(
            query=req.query,
            parsed_intent=intent,
            memory_used=memory_used,
            memory_summary=memory_summary or None,
            results=[],
            total_found=0,
        )

    user_memory = memory_summary
    ranked_results = composite_score_and_rank(candidates, enriched_intent, user_memory)

    return SearchResponse(
        query=req.query,
        parsed_intent=intent,
        memory_used=memory_used,
        memory_summary=memory_summary or None,
        results=ranked_results,
        total_found=len(ranked_results),
    )


@app.post("/signal")
async def record_behavioral_signal(req: SignalRequest):
    if req.signal_type not in ("click", "book"):
        raise HTTPException(status_code=400, detail="signal_type must be 'click' or 'book'")

    await record_signal(req.user_id, req.signal_type, req.property_data)
    return {"status": "ok", "signal": req.signal_type, "user": req.user_id}



@app.post("/property-qa", response_model=PropertyQAResponse)
async def property_qa(req: PropertyQARequest):
    try:
        result = answer_property_question(req.question, top_k=req.top_k)
    except Exception as e:
        logger.error(f"Property QA failed: {e}")
        raise HTTPException(status_code=500, detail=f"Property QA failed: {str(e)}")

    return PropertyQAResponse(
        question=req.question,
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
    )



@app.get("/health")
async def health():
    try:
        client = get_qdrant()
        count = client.count(collection_name=COLLECTION_NAME).count
        return {
            "status": "healthy",
            "qdrant": "connected",
            "properties_indexed": count,
            "needs_indexing": count == 0,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "qdrant": "error",
            "error": str(e),
            "needs_indexing": True,
        }



@app.get("/cities")
async def get_cities():
    try:
        with open(PROPERTIES_FILE) as f:
            data = json.load(f)
        cities = sorted(set(
            p["location"]["city"]
            for p in data["properties"]
            if p.get("location", {}).get("city")
        ))
        return {"cities": cities}
    except Exception as e:
        return {"cities": [], "error": str(e)}


@app.post("/onboarding", response_model=OnboardingResponse)
async def save_user_onboarding(req: OnboardingRequest):
    """Save user travel preferences from onboarding flow."""
    try:
        preferences = TravelPreferences(
            user_id=req.user_id,
            travel_style=req.travel_style,
            property_preferences=req.property_preferences,
            typical_budget=req.typical_budget,
            must_have_amenities=req.must_have_amenities,
            preferred_vibes=req.preferred_vibes,
            typical_companions=req.typical_companions,
            accessibility_needs=req.accessibility_needs,
            location_preferences=req.location_preferences,
            custom_interests=req.custom_interests,
            dietary_preferences=req.dietary_preferences,
        )

        success = save_preferences(preferences)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save preferences")

        try:
            await record_signal(req.user_id, "onboarding", preferences.model_dump())
        except Exception as memory_error:
            logger.warning(f"Onboarding memory setup failed for {req.user_id}: {memory_error}")

        summary = build_preference_summary(preferences)

        return OnboardingResponse(
            success=True,
            user_id=req.user_id,
            message="Your travel preferences have been saved! We'll use these to personalize your hotel recommendations.",
            preferences_summary=summary,
            preferences=preferences.model_dump(),
        )
    except Exception as e:
        logger.error(f"Onboarding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Onboarding failed: {str(e)}")


@app.post("/itinerary-chat", response_model=ItineraryChatResponse)
async def itinerary_chat(req: ItineraryChatRequest):
    try:
        reply = generate_itinerary_reply(
            message=req.message,
            search_query=req.search_query,
            parsed_intent=req.parsed_intent,
            memory_summary=req.memory_summary,
            preferences_summary=req.preferences_summary,
            top_properties=req.top_properties,
            history=[msg.model_dump() for msg in req.history],
        )
        return ItineraryChatResponse(reply=reply)
    except Exception as e:
        logger.error(f"Itinerary planner failed: {e}")
        raise HTTPException(status_code=500, detail=f"Itinerary planner failed: {str(e)}")


@app.get("/preferences/{user_id}", response_model=PreferencesResponse)
async def get_user_preferences(user_id: str):
    """Get user preferences."""
    try:
        preferences = load_preferences(user_id)

        if not preferences:
            return PreferencesResponse(
                user_id=user_id,
                has_preferences=False,
                preferences=None,
                summary=None,
            )

        return PreferencesResponse(
            user_id=user_id,
            has_preferences=True,
            preferences=preferences.model_dump(),
            summary=build_preference_summary(preferences),
        )
    except Exception as e:
        logger.error(f"Failed to get preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")
