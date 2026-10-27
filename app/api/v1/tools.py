"""规划表单提交接口。流程本身在 app/services/trip_form.py。"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_current_user
from app.models.user import User
from app.schemas.tools import ToolResultRequest
from app.services.trip_form import ToolResultError, open_tool_result_stream


router = APIRouter(prefix="/chat/tools", tags=["chat tools"])


@router.post("/{call_id}/result")
async def submit_tool_result(
    call_id: str,
    data: ToolResultRequest,
    user: User = Depends(get_current_user),
):
    try:
        stream = await open_tool_result_stream(call_id, data, str(user.id))
    except ToolResultError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
