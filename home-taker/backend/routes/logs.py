from datetime import datetime

from fastapi import APIRouter, HTTPException

from backend.models import APIResponse, LogRequest
from backend.services import CogneeService
from backend.utils.formatters import format_log_entry

router = APIRouter(prefix="/api", tags=["logs"])


@router.post("/log", response_model=APIResponse)
async def add_log(request: LogRequest):
    """Add a log entry to Cognee memory."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_text = format_log_entry(request.type, request.title, request.details, timestamp)

        await CogneeService.add_and_cognify(log_text)

        return APIResponse(
            success=True,
            message="Log entry saved to memory",
            data={
                "type": request.type,
                "title": request.title,
                "details": request.details,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        print(f"Error saving log: {e}")
        raise HTTPException(status_code=500, detail=str(e))
