"""
Travel Preferences Module
Handles user onboarding and preference storage for personalized recommendations.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Preferences storage path
PREFERENCES_DIR = Path(__file__).parent.parent / "data" / "user_preferences"
PREFERENCES_DIR.mkdir(parents=True, exist_ok=True)


class TravelPreferences(BaseModel):
    """User travel preferences from onboarding."""
    user_id: str

    # Travel style
    travel_style: list[str] = Field(default_factory=list)  # e.g., ["adventure", "relaxation", "cultural"]

    # Accommodation preferences
    property_preferences: list[str] = Field(default_factory=list)  # e.g., ["boutique", "resort", "hostel"]

    # Budget
    typical_budget: Optional[str] = None  # e.g., "budget", "mid-range", "luxury"

    # Amenities importance
    must_have_amenities: list[str] = Field(default_factory=list)  # e.g., ["wifi", "pool", "gym"]

    # Atmosphere
    preferred_vibes: list[str] = Field(default_factory=list)  # e.g., ["quiet", "romantic", "social"]

    # Travel companions
    typical_companions: list[str] = Field(default_factory=list)  # e.g., ["solo", "couple", "family"]

    # Special needs
    accessibility_needs: list[str] = Field(default_factory=list)  # e.g., ["wheelchair accessible", "quiet environment"]

    # Location preferences
    location_preferences: list[str] = Field(default_factory=list)  # e.g., ["beach", "city center", "mountains"]

    # Free-form interests useful for trip planning
    custom_interests: Optional[str] = None

    # Food or dietary preferences for itinerary planning
    dietary_preferences: list[str] = Field(default_factory=list)

    # Timestamp
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def save_preferences(preferences: TravelPreferences) -> bool:
    """Save user preferences to disk."""
    try:
        preferences.updated_at = datetime.utcnow().isoformat()
        file_path = PREFERENCES_DIR / f"{preferences.user_id}.json"
        with open(file_path, "w") as f:
            json.dump(preferences.model_dump(), f, indent=2)
        logger.info(f"Saved preferences for user {preferences.user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save preferences: {e}")
        return False


def load_preferences(user_id: str) -> Optional[TravelPreferences]:
    """Load user preferences from disk."""
    try:
        file_path = PREFERENCES_DIR / f"{user_id}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            data = json.load(f)
        return TravelPreferences(**data)
    except Exception as e:
        logger.error(f"Failed to load preferences for {user_id}: {e}")
        return None


def preferences_to_filter_boost(preferences: TravelPreferences) -> dict:
    """Convert preferences to filter/boost parameters for search."""
    boost_params = {
        "amenities": preferences.must_have_amenities,
        "vibes": preferences.preferred_vibes,
        "property_types": preferences.property_preferences,
        "location_vibes": preferences.location_preferences,
    }

    # Budget mapping
    budget_map = {
        "budget": (0, 150),
        "mid-range": (100, 300),
        "luxury": (250, 1000),
    }
    if preferences.typical_budget:
        boost_params["budget_range"] = budget_map.get(preferences.typical_budget, (0, 500))

    return boost_params


def build_preference_summary(preferences: TravelPreferences) -> str:
    """Build a human-readable summary of user preferences."""
    parts = []

    if preferences.travel_style:
        parts.append(f"Travel style: {', '.join(preferences.travel_style)}")

    if preferences.property_preferences:
        parts.append(f"Prefers {', '.join(preferences.property_preferences)} properties")

    if preferences.typical_budget:
        parts.append(f"{preferences.typical_budget} budget")

    if preferences.must_have_amenities:
        parts.append(f"Must have: {', '.join(preferences.must_have_amenities)}")

    if preferences.preferred_vibes:
        parts.append(f"Enjoys {', '.join(preferences.preferred_vibes)} atmosphere")

    if preferences.typical_companions:
        parts.append(f"Usually travels {'/'.join(preferences.typical_companions)}")

    if preferences.custom_interests:
        parts.append(f"Special interests: {preferences.custom_interests}")

    if preferences.dietary_preferences:
        parts.append(f"Dietary preferences: {', '.join(preferences.dietary_preferences)}")

    return " | ".join(parts) if parts else "No preferences set yet"
