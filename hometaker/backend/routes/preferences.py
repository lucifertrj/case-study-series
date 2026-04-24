from datetime import datetime

from fastapi import APIRouter, HTTPException

from backend.models import APIResponse, PreferencesRequest
from backend.services import CogneeService, StorageService
from backend.utils import format_profile_for_cognee

router = APIRouter(prefix="/api/preferences", tags=["preferences"])


@router.post("", response_model=APIResponse)
async def create_preferences(request: PreferencesRequest):
    """Save user preferences and ingest into Cognee memory."""
    try:
        # Save to file as backup
        all_preferences = StorageService.load_preferences()
        new_entry = {
            "id": len(all_preferences) + 1,
            "userType": request.userType,
            "preferences": request.preferences,
            "createdAt": request.createdAt or datetime.now().isoformat(),
            "updatedAt": datetime.now().isoformat(),
        }
        all_preferences.append(new_entry)
        StorageService.save_preferences(all_preferences)

        # Format and ingest into Cognee
        profile_text = format_profile_for_cognee(request.userType, request.preferences)
        await CogneeService.add_and_cognify(profile_text)

        return APIResponse(
            success=True,
            message="Profile saved to memory successfully",
            data=new_entry,
        )

    except Exception as e:
        print(f"Error saving preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=APIResponse)
async def get_preferences():
    """Get all saved preferences from file."""
    try:
        preferences = StorageService.load_preferences()
        return APIResponse(
            success=True,
            message="Preferences retrieved successfully",
            data={"preferences": preferences},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest", response_model=APIResponse)
async def get_latest_preferences():
    """Get the most recent preference entry."""
    try:
        preferences = StorageService.load_preferences()
        if not preferences:
            return APIResponse(
                success=True,
                message="No preferences found",
                data=None,
            )

        latest = preferences[-1]
        return APIResponse(
            success=True,
            message="Latest preferences retrieved successfully",
            data=latest,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("", response_model=APIResponse)
async def clear_preferences():
    """Clear all saved preferences and reset Cognee memory."""
    try:
        StorageService.clear_preferences()
        await CogneeService.reset_memory()

        return APIResponse(
            success=True,
            message="All preferences and memory cleared",
            data=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
