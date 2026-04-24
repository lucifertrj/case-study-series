import json

from backend.config import PREFERENCES_FILE


class StorageService:
    """File-based storage service for preferences."""

    @staticmethod
    def load_preferences() -> list[dict]:
        """Load existing preferences from file."""
        if PREFERENCES_FILE.exists():
            with open(PREFERENCES_FILE, "r") as f:
                return json.load(f)
        return []

    @staticmethod
    def save_preferences(preferences: list[dict]) -> None:
        """Save preferences to file."""
        with open(PREFERENCES_FILE, "w") as f:
            json.dump(preferences, f, indent=2)

    @staticmethod
    def clear_preferences() -> None:
        """Clear all preferences."""
        StorageService.save_preferences([])
