"""任务、事件、审批、偏好和正式行程 API。"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import factory as agent_factory
from app.api.dependencies import ConversationId, get_current_user
from app.core.checkpointer import get_checkpointer
from app.governance.approvals import ApprovalService
from app.governance.drafts import PostgresDraftRepository, save_trip_draft
from app.governance.events import TaskEventService
from app.governance.itineraries import ItineraryGovernanceService
from app.governance.postgres import (
    PostgresApprovalRepository,
    PostgresEventRepository,
    PostgresItineraryRepository,
    PostgresPreferenceRepository,
    PostgresTripHistoryRepository,
)
from app.memory.defaults import apply_preference_defaults, resolve_preference_defaults
from app.memory.service import MemoryGovernanceService
from app.models.user import User
from app.models.base import get_db
from app.models.conversation import Conversation
from app.schemas.governance import ApprovalDecisionRequest, ItinerarySaveRequest, PreferenceProposalRequest
from app.schemas.planning import TravelRequirement
from app.utils.logger import app_logger

router = APIRouter(tags=["旅行规划任务"])


async def run_travel_planning(*args, **kwargs):
    """Compatibility seam for tests and callers replacing the planning runner."""
    return await agent_factory.run_travel_planning(*args, **kwargs)


@router.post("/tasks")
async def create_planning_task(
    requirement: TravelRequirement,
    user: User = Depends(get_current_user),
    request: Request = None,
    conversation_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    task_id = uuid4().hex
    event_service = TaskEventService(PostgresEventRepository())
    if conversation_id:
        result = await db.execute(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .where(Conversation.user_id == user.id)
        )
        if result.scalar_one_or_none() is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="conversation not found")
    try:
        defaults = await resolve_preference_defaults(str(user.id), PostgresPreferenceRepository())
    except Exception as exc:
        app_logger.warning(f"读取长期偏好失败，按无偏好处理: task_id={task_id} error={exc}", exc_info=True)
        defaults = {}
    requirement = apply_preference_defaults(requirement, defaults)
    app_state = getattr(getattr(request, "app", None), "state", None)
    draft = await run_travel_planning(
        requirement,
        checkpointer=await get_checkpointer(),
        event_service=event_service,
        task_id=task_id,
        user_id=str(user.id),
        registry=getattr(app_state, "planning_registry", None),
        fallback_reason=getattr(app_state, "planning_fallback_reason", None),
        conversation_id=conversation_id,
    )
    draft_record = None
    if conversation_id:
        draft_record = await save_trip_draft(
            PostgresDraftRepository(),
            str(user.id),
            conversation_id,
            draft,
        )
    return {
        "task_id": task_id,
        "status": "degraded" if draft.status == "degraded" else "completed",
        "draft": draft.model_dump(mode="json"),
        "draft_version": draft_record.version if draft_record else None,
    }


@router.get("/tasks/{task_id}")
async def get_planning_task(task_id: str, user: User = Depends(get_current_user)):
    events = await PostgresEventRepository().list(task_id, str(user.id))
    if not events:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    last_event = events[-1]
    if last_event.event_type == "task_failed":
        task_status = "failed"
    elif last_event.event_type == "task_completed":
        task_status = last_event.payload.get("status", "completed")
    else:
        task_status = "running"
    return {
        "task_id": task_id,
        "status": task_status,
        "last_event": last_event.model_dump(mode="json"),
    }


@router.get("/drafts/{conversation_id}")
async def get_trip_draft(
    conversation_id: ConversationId,
    user: User = Depends(get_current_user),
):
    record = await PostgresDraftRepository().get(str(user.id), conversation_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="draft not found")
    return record.model_dump(mode="json")


@router.get("/tasks/{task_id}/events")
async def get_task_events(task_id: str, user: User = Depends(get_current_user)):
    events = await PostgresEventRepository().list(task_id, str(user.id))
    return {"task_id": task_id, "events": [event.model_dump(mode="json") for event in events]}


@router.post("/preferences/proposals")
async def propose_preference(data: PreferenceProposalRequest, user: User = Depends(get_current_user)):
    service = MemoryGovernanceService(
        ApprovalService(PostgresApprovalRepository()),
        PostgresPreferenceRepository(),
    )
    approval = await service.request_upsert(data.task_id, str(user.id), data.key, data.value)
    return approval.model_dump(mode="json")


@router.post("/itineraries/{conversation_id}/save")
async def request_itinerary_save(
    conversation_id: ConversationId,
    data: ItinerarySaveRequest,
    user: User = Depends(get_current_user),
):
    service = ItineraryGovernanceService(
        ApprovalService(PostgresApprovalRepository()),
        PostgresItineraryRepository(),
    )
    approval = await service.request_save(
        data.task_id, str(user.id), conversation_id, data.title, data.content
    )
    return approval.model_dump(mode="json")


@router.get("/itineraries/{conversation_id}")
async def get_saved_itinerary(conversation_id: ConversationId, user: User = Depends(get_current_user)):
    record = await PostgresItineraryRepository().get(str(user.id), conversation_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="正式行程不存在")
    return record


@router.post("/approvals/{approval_id}/decision")
async def decide_approval(
    approval_id: str,
    data: ApprovalDecisionRequest,
    user: User = Depends(get_current_user),
):
    approval_repository = PostgresApprovalRepository()
    approvals = ApprovalService(approval_repository)
    try:
        record = await approvals.decide(approval_id, str(user.id), data.decision, data.payload)
        applied = None
        if record.status in {"approved", "edited"} and record.action.startswith("memory."):
            applied = await MemoryGovernanceService(
                approvals, PostgresPreferenceRepository()
            ).apply(record.id, str(user.id))
        elif record.status in {"approved", "edited"} and record.action.startswith("itinerary."):
            applied = await ItineraryGovernanceService(
                approvals, PostgresItineraryRepository(), PostgresTripHistoryRepository()
            ).apply(record.id, str(user.id))
        return {
            "approval": record.model_dump(mode="json"),
            "applied": applied.model_dump(mode="json") if hasattr(applied, "model_dump") else applied,
        }
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
