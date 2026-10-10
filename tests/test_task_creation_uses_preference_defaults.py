from datetime import date

import pytest

import app.api.v1.planning as planning_api
from app.schemas.governance import ApprovalDecisionRequest, ApprovalRecord, PreferenceRecord
from app.schemas.planning import TravelPlanDraft, TravelRequirement


class _FakePreferenceRepository:
    def __init__(self, records):
        self._records = records

    async def list(self, user_id):
        return self._records


class _FailingPreferenceRepository:
    async def list(self, user_id):
        raise RuntimeError("db down")


class _FakeUser:
    def __init__(self, user_id):
        self.id = user_id


async def _noop_coro(value=None):
    return value


def _patch_task_creation_collaborators(monkeypatch, preference_repository, requirement_holder):
    monkeypatch.setattr(planning_api, "PostgresPreferenceRepository", lambda: preference_repository)
    monkeypatch.setattr(planning_api, "get_checkpointer", lambda: _noop_coro(None))
    monkeypatch.setattr(planning_api, "TaskEventService", lambda *args, **kwargs: None)
    monkeypatch.setattr(planning_api, "PostgresEventRepository", lambda: None)

    async def fake_run_travel_planning(requirement, **kwargs):
        requirement_holder["requirement"] = requirement
        return TravelPlanDraft(
            requirement=requirement, itinerary=[], worker_results=[], evidence=[]
        )

    monkeypatch.setattr(planning_api, "run_travel_planning", fake_run_travel_planning)


@pytest.mark.asyncio
async def test_create_planning_task_fills_empty_fields_from_confirmed_preferences(monkeypatch):
    records = [PreferenceRecord(user_id="u1", key="food_preferences", value=["清淡", "不吃辣"])]
    captured = {}
    _patch_task_creation_collaborators(monkeypatch, _FakePreferenceRepository(records), captured)
    requirement = TravelRequirement(destination="成都", departure_date=date(2026, 8, 1), days=3)

    await planning_api.create_planning_task(requirement, _FakeUser("u1"))

    assert captured["requirement"].food_preferences == ["清淡", "不吃辣"]


@pytest.mark.asyncio
async def test_create_planning_task_never_overrides_explicit_field(monkeypatch):
    records = [PreferenceRecord(user_id="u1", key="food_preferences", value=["清淡", "不吃辣"])]
    captured = {}
    _patch_task_creation_collaborators(monkeypatch, _FakePreferenceRepository(records), captured)
    requirement = TravelRequirement(
        destination="成都", departure_date=date(2026, 8, 1), days=3, food_preferences=["微辣"]
    )

    await planning_api.create_planning_task(requirement, _FakeUser("u1"))

    assert captured["requirement"].food_preferences == ["微辣"]


@pytest.mark.asyncio
async def test_create_planning_task_degrades_to_no_defaults_when_preference_lookup_fails(monkeypatch):
    captured = {}
    _patch_task_creation_collaborators(monkeypatch, _FailingPreferenceRepository(), captured)
    requirement = TravelRequirement(destination="成都", departure_date=date(2026, 8, 1), days=3)

    result = await planning_api.create_planning_task(requirement, _FakeUser("u1"))

    assert result["status"] == "completed"
    assert captured["requirement"].food_preferences == []


@pytest.mark.asyncio
async def test_decide_approval_wires_trip_history_repository_into_itinerary_apply(monkeypatch):
    captured_args = {}

    class _SpyItineraryGovernanceService:
        def __init__(self, approvals, repository, trip_history=None):
            captured_args["trip_history"] = trip_history

        async def apply(self, approval_id, user_id):
            return {"id": "itin-1", "version": 1}

    class _FakeApprovalService:
        def __init__(self, repository):
            self.repository = repository

        async def decide(self, approval_id, user_id, decision, payload):
            return ApprovalRecord(
                id=approval_id, task_id="t", user_id=user_id, action="itinerary.save",
                payload={}, status="approved",
            )

    monkeypatch.setattr(planning_api, "ItineraryGovernanceService", _SpyItineraryGovernanceService)
    monkeypatch.setattr(planning_api, "ApprovalService", _FakeApprovalService)
    monkeypatch.setattr(planning_api, "PostgresApprovalRepository", lambda: None)
    monkeypatch.setattr(planning_api, "PostgresItineraryRepository", lambda: None)
    monkeypatch.setattr(planning_api, "PostgresTripHistoryRepository", lambda: "trip-history-repo-instance")

    await planning_api.decide_approval(
        "approval-1", ApprovalDecisionRequest(decision="approve"), _FakeUser("u1")
    )

    assert captured_args["trip_history"] == "trip-history-repo-instance"
