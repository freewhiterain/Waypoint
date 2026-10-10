from datetime import date

import pytest

from app.governance.drafts import (
    InMemoryDraftRepository,
    load_trip_draft_context,
    save_trip_draft,
)
from app.schemas.planning import TravelPlanDraft, TravelRequirement


def requirement() -> TravelRequirement:
    return TravelRequirement(
        destination="Chengdu",
        departure_date=date(2026, 8, 1),
        days=2,
    )


@pytest.mark.asyncio
async def test_save_trip_draft_persists_owned_workspace_and_increments_version():
    repository = InMemoryDraftRepository()
    draft = TravelPlanDraft(
        requirement=requirement(),
        itinerary=[],
        worker_results=[],
        evidence=[],
    )

    first = await save_trip_draft(repository, "user-1", "conversation-1", draft)
    second = await save_trip_draft(repository, "user-1", "conversation-1", draft)

    assert first.version == 1
    assert second.version == 2
    assert (await repository.get("user-1", "conversation-1")).version == 2
    assert await repository.get("user-2", "conversation-1") is None


@pytest.mark.asyncio
async def test_load_trip_draft_context_returns_only_the_owned_workspace():
    repository = InMemoryDraftRepository()
    draft = TravelPlanDraft(
        requirement=requirement(),
        itinerary=[],
        worker_results=[],
        evidence=[],
    )
    await save_trip_draft(repository, "user-1", "conversation-1", draft)

    context = await load_trip_draft_context(repository, "user-1", "conversation-1")

    assert context["version"] == 1
    assert context["requirement"]["destination"] == "Chengdu"
    assert await load_trip_draft_context(repository, "user-2", "conversation-1") is None
