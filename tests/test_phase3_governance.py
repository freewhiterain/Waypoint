from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.supervisor import create_supervisor_graph, run_travel_planning
from app.governance.approvals import ApprovalService, InMemoryApprovalRepository
from app.governance.events import InMemoryEventRepository, TaskEventService
from app.governance.itineraries import InMemoryItineraryRepository, ItineraryGovernanceService
from app.memory.service import InMemoryPreferenceRepository, MemoryGovernanceService
from app.schemas.planning import TravelRequirement
from app.models.base import Base
import app.models  # noqa: F401


@pytest.mark.asyncio
async def test_memory_is_not_written_before_owner_approval():
    approvals = ApprovalService(InMemoryApprovalRepository())
    preferences = InMemoryPreferenceRepository()
    service = MemoryGovernanceService(approvals, preferences)
    pending = await service.request_upsert("task-1", "user-1", "diet", ["不吃花生"])

    assert await preferences.list("user-1") == []
    with pytest.raises(PermissionError):
        await service.apply(pending.id, "user-1")
    with pytest.raises(PermissionError):
        await approvals.decide(pending.id, "user-2", "approve")

    await approvals.decide(pending.id, "user-1", "approve")
    stored = await service.apply(pending.id, "user-1")

    assert stored.key == "diet"
    assert stored.value == ["不吃花生"]


@pytest.mark.asyncio
async def test_edit_and_delete_memory_require_separate_approvals():
    approvals = ApprovalService(InMemoryApprovalRepository())
    preferences = InMemoryPreferenceRepository()
    service = MemoryGovernanceService(approvals, preferences)
    create = await service.request_upsert("t", "u", "style", "文化")
    await approvals.decide(create.id, "u", "edit", {"key": "style", "value": "美食"})
    await service.apply(create.id, "u")
    delete = await service.request_delete("t", "u", "style")

    assert (await preferences.list("u"))[0].value == "美食"
    with pytest.raises(PermissionError):
        await service.apply(delete.id, "u")
    await approvals.decide(delete.id, "u", "approve")
    assert await service.apply(delete.id, "u") is True


@pytest.mark.asyncio
async def test_formal_itinerary_save_and_overwrite_are_approved_and_versioned():
    approvals = ApprovalService(InMemoryApprovalRepository())
    repository = InMemoryItineraryRepository()
    service = ItineraryGovernanceService(approvals, repository)
    first = await service.request_save("t", "u", "c", "成都", {"days": 5})
    await approvals.decide(first.id, "u", "approve")
    saved = await service.apply(first.id, "u")
    second = await service.request_save("t", "u", "c", "成都修改版", {"days": 4})

    assert saved["version"] == 1
    assert second.action == "itinerary.overwrite"
    await approvals.decide(second.id, "u", "approve")
    assert (await service.apply(second.id, "u"))["version"] == 2



@pytest.mark.asyncio
async def test_supervisor_persists_checkpoint_and_ordered_events():
    saver = InMemorySaver()
    event_repository = InMemoryEventRepository()
    event_service = TaskEventService(event_repository)
    requirement = TravelRequirement(
        origin="上海",
        destination="成都",
        departure_date=date(2026, 8, 1),
        days=2,
    )

    await run_travel_planning(
        requirement,
        checkpointer=saver,
        event_service=event_service,
        task_id="task-checkpoint",
        user_id="user-1",
        conversation_id="conversation-1",
    )
    graph = create_supervisor_graph(checkpointer=saver)
    snapshot = await graph.aget_state({"configurable": {"thread_id": "task-checkpoint"}})
    events = await event_repository.list("task-checkpoint", "user-1")

    assert snapshot.values["status"] == "completed"
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].event_type == "task_created"
    assert events[-1].event_type == "task_completed"
    assert await event_repository.list("task-checkpoint", "other-user") == []


def test_governance_tables_are_registered_for_database_initialization():
    assert {"taskevent", "approval", "userpreference", "saveditinerary"}.issubset(Base.metadata.tables)
