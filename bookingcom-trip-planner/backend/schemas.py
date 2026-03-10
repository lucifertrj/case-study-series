from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

class FilterSchema(BaseModel):
    destination: Optional[str] = Field(None)
    vibe: list[str] = Field(default_factory=list)
    amenities: list[str] = Field(default_factory=list)
    property_type: list[str] = Field(default_factory=list)
    budget_range: Optional[list[int]] = Field(None)
    special_needs: list[str] = Field(default_factory=list)
    location_vibe: Optional[str] = Field(None)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    user_id: str = Field(default="anonymous")
    session_id: str = Field(default="default")


class PropertyResult(BaseModel):
    id: str
    name: str
    type: str
    city: str
    country: str
    neighborhood: str
    location_vibe: str
    price_per_night: float
    rating: float
    review_count: int
    amenities: list[str]
    vibe_tags: list[str]
    description: str
    review_highlights: list[str]
    available: bool
    composite_score: float
    semantic_score: float
    filter_score: float
    quality_score: float
    memory_score: float
    match_reason: str


class SearchResponse(BaseModel):
    query: str
    parsed_intent: FilterSchema
    memory_used: bool
    memory_summary: Optional[str]
    results: list[PropertyResult]
    total_found: int


class SignalRequest(BaseModel):
    user_id: str
    session_id: str
    property_id: str
    signal_type: str
    property_data: dict


class PropertyQASource(BaseModel):
    id: str
    name: str
    city: str
    score: float


class PropertyQARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=600)
    top_k: int = Field(default=5, ge=1, le=15)


class PropertyQAResponse(BaseModel):
    question: str
    answer: str
    sources: list[PropertyQASource]


class OnboardingRequest(BaseModel):
    """User onboarding request with travel preferences."""
    user_id: str = Field(..., min_length=1)
    travel_style: list[str] = Field(default_factory=list)
    property_preferences: list[str] = Field(default_factory=list)
    typical_budget: Optional[str] = None
    must_have_amenities: list[str] = Field(default_factory=list)
    preferred_vibes: list[str] = Field(default_factory=list)
    typical_companions: list[str] = Field(default_factory=list)
    accessibility_needs: list[str] = Field(default_factory=list)
    location_preferences: list[str] = Field(default_factory=list)
    custom_interests: Optional[str] = Field(default=None, max_length=400)
    dietary_preferences: list[str] = Field(default_factory=list)


class OnboardingResponse(BaseModel):
    """Response after saving preferences."""
    success: bool
    user_id: str
    message: str
    preferences_summary: str
    preferences: Optional[dict] = None


class PreferencesResponse(BaseModel):
    """Response for getting user preferences."""
    user_id: str
    has_preferences: bool
    preferences: Optional[dict] = None
    summary: Optional[str] = None


class ItineraryChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class ItineraryChatRequest(BaseModel):
    user_id: str = Field(default="anonymous")
    message: str = Field(..., min_length=3, max_length=2000)
    search_query: str = Field(..., min_length=2, max_length=500)
    parsed_intent: dict = Field(default_factory=dict)
    memory_summary: Optional[str] = None
    preferences_summary: Optional[str] = None
    top_properties: list[dict] = Field(default_factory=list)
    history: list[ItineraryChatMessage] = Field(default_factory=list)


class ItineraryChatResponse(BaseModel):
    reply: str
