from __future__ import annotations

def build_property_qa_document(prop: dict) -> str:
    location = prop.get("location", {})
    review_highlights = prop.get("review_highlights", [])
    amenities = prop.get("amenities", [])
    vibe_tags = prop.get("vibe_tags", [])

    reviews_joined = " | ".join(review_highlights)
    amenities_joined = ", ".join(amenities)
    vibe_joined = ", ".join(vibe_tags)

    return (
        f"Property: {prop.get('name', '')}\n"
        f"Type: {prop.get('type', '')}\n"
        f"City: {location.get('city', '')}\n"
        f"Neighborhood: {location.get('neighborhood', '')}\n"
        f"Location vibe: {location.get('location_vibe', '')}\n"
        f"Price per night (USD): {prop.get('price_per_night', '')}\n"
        f"Rating: {prop.get('rating', '')}/10 from {prop.get('review_count', '')} reviews\n"
        f"Amenities: {amenities_joined}\n"
        f"Vibe tags: {vibe_joined}\n"
        f"Description: {prop.get('description', '')}\n"
        f"Guest review highlights: {reviews_joined}"
    )