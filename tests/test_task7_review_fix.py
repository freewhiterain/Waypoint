from datetime import date
from types import SimpleNamespace

import pytest

from app.agents import factory
from app.agents.subagents.registry import SubagentRegistry
from app.agents.workers.registry import WorkerRegistry
from app.api.v1 import planning as planning_api
from app.api.v1 import tools as tools_api
from app.governance.events import InMemoryEventRepository
from app.schemas.governance import TaskEventRecord
from app.schemas.planning import TripDraftRecord, TravelPlanDraft, TravelRequirement

from tests.test_trip_form_tool_flow import (
    FakeDraft,
    InMemoryInvocationRepository,
    configure_endpoint,
    endpoint_client,
    invocation,
    parse_sse,
)


def requirement() -> TravelRequirement:
    return TravelRequirement(
        destination="Chengdu",
        departure_date=date(2026, 9, 1),
        days=3,
    )


def draft_for(requirement: TravelRequirement) -> TravelPlanDraft:
    return TravelPlanDraft(
        requirement=requirement,
        itinerary=[],
        worker_results=[],
        evidence=[],
    )


@pytest.mark.asyncio
async def test_planning_task_status_reports_failed_and_degraded_terminal_events(monkeypatch):
    repository = InMemoryEventRepository()
    await repository.append(
        TaskEventRecord(
            task_id="failed-task",
            user_id="user-1",
            event_type="task_failed",
            payload={"error": "ProviderError"},
            sequence=1,
        )
    )
    await repository.append(
        TaskEventRecord(
            task_id="degraded-task",
            user_id="user-1",
            event_type="task_completed",
            payload={"status": "degraded"},
            sequence=1,
        )
    )
    monkeypatch.setattr(planning_api, "PostgresEventRepository", lambda: repository)
    user = SimpleNamespace(id="user-1")

    failed = await planning_api.get_planning_task("failed-task", user)
    degraded = await planning_api.get_planning_task("degraded-task", user)

    assert failed["status"] == "failed"
    assert degraded["status"] == "degraded"


def test_factory_selects_legacy_registry_for_supervisor_mode(monkeypatch):
    monkeypatch.setattr(factory.settings, "travel_agent_mode", "supervisor")

    registry, fallback_reason = factory.create_planning_registry()

    assert isinstance(registry, WorkerRegistry)
    assert fallback_reason is None


def test_factory_selects_subagent_registry_for_subagent_mode(monkeypatch):
    monkeypatch.setattr(factory.settings, "travel_agent_mode", "supervisor_subagents")
    monkeypatch.setattr(factory.settings, "llm_api_key", "configured")

    registry, fallback_reason = factory.create_planning_registry()

    assert isinstance(registry, SubagentRegistry)
    assert fallback_reason is None


@pytest.mark.asyncio
async def test_chat_agent_graph_receives_mode_selected_registry(monkeypatch):
    from app.agents import supervisor

    registry = object()
    captured = {}

    def fake_registry():
        return registry, None

    def fake_graph(*, registry):
        captured["registry"] = registry
        return "graph"

    monkeypatch.setattr(factory, "create_planning_registry", fake_registry)
    monkeypatch.setattr(supervisor, "create_supervisor_graph", fake_graph)

    assert await factory.create_chat_agent() == "graph"
    assert captured["registry"] is registry


@pytest.mark.asyncio
async def test_factory_fallback_marks_no_llm_planning_as_degraded(monkeypatch):
    monkeypatch.setattr(factory.settings, "travel_agent_mode", "supervisor_subagents")
    monkeypatch.setattr(factory.settings, "llm_api_key", "")
    monkeypatch.setattr(factory.settings, "allow_legacy_fallback", True)
    captured = {}

    async def fake_run(requirement, **kwargs):
        captured["registry"] = kwargs["registry"]
        return draft_for(requirement)

    monkeypatch.setattr("app.agents.supervisor.run_travel_planning", fake_run)

    result = await factory.run_travel_planning(requirement())

    assert isinstance(captured["registry"], WorkerRegistry)
    assert result.status == "degraded"
    assert "planning_degraded:no_llm_or_provider" in result.warnings


@pytest.mark.asyncio
async def test_planning_endpoint_uses_configured_factory_runner(monkeypatch):
    captured = {}

    async def fake_run(requirement, **kwargs):
        captured["requirement"] = requirement
        return draft_for(requirement)

    monkeypatch.setattr(factory, "run_travel_planning", fake_run)
    monkeypatch.setattr(planning_api, "PostgresPreferenceRepository", lambda: _EmptyPreferences())
    monkeypatch.setattr(planning_api, "PostgresEventRepository", lambda: None)
    monkeypatch.setattr(planning_api, "TaskEventService", lambda *_args: None)
    monkeypatch.setattr(planning_api, "get_checkpointer", lambda: _none())

    result = await planning_api.create_planning_task(requirement(), SimpleNamespace(id="user-1"))

    assert captured["requirement"].destination == "Chengdu"
    assert result["draft"]["status"] == "draft"


@pytest.mark.asyncio
async def test_planning_endpoint_persists_a_draft_only_for_an_owned_conversation(monkeypatch):
    captured = {}

    class QueryResult:
        def scalar_one_or_none(self):
            return object()

    class FakeDB:
        async def execute(self, _statement):
            return QueryResult()

    async def fake_run(requirement, **kwargs):
        captured.update(kwargs)
        return draft_for(requirement)

    async def fake_save(repository, user_id, conversation_id, draft):
        captured["draft_save"] = (user_id, conversation_id, draft)
        return TripDraftRecord(
            user_id=user_id,
            conversation_id=conversation_id,
            version=3,
            requirement=draft.requirement.model_dump(mode="json"),
            content=draft.model_dump(mode="json"),
        )

    monkeypatch.setattr(planning_api, "run_travel_planning", fake_run)
    monkeypatch.setattr(planning_api, "save_trip_draft", fake_save)
    async def empty_defaults(*_args):
        return {}

    monkeypatch.setattr(planning_api, "resolve_preference_defaults", empty_defaults)
    monkeypatch.setattr(planning_api, "get_checkpointer", _none)

    user = SimpleNamespace(id="user-1")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    result = await planning_api.create_planning_task(
        requirement(),
        user,
        request,
        "conversation-1",
        FakeDB(),
    )

    assert result["status"] == "completed"
    assert result["draft_version"] == 3
    assert captured["conversation_id"] == "conversation-1"
    assert captured["draft_save"][:2] == ("user-1", "conversation-1")


@pytest.mark.asyncio
async def test_tool_endpoint_uses_configured_factory_runner(monkeypatch):
    user = SimpleNamespace(id="user-1")
    repository = InMemoryInvocationRepository([invocation(user_id=user.id)])
    configure_endpoint(monkeypatch, repository)
    captured = {}

    async def fake_run(requirement, **kwargs):
        captured["requirement"] = requirement
        return FakeDraft(requirement.destination)

    monkeypatch.setattr(factory, "run_travel_planning", fake_run)

    async with endpoint_client(user) as client:
        response = await client.post(
            "/api/v1/chat/tools/call-1/result",
            json={
                "tool": "collect_trip_requirements",
                "status": "completed",
                "result": {
                    "destination": "Kyoto",
                    "departure_date": "2026-08-03",
                    "days": 4,
                },
            },
        )

    events = parse_sse(response)
    assert captured["requirement"].destination == "Kyoto"
    assert [event["type"] for event in events][-3:] == ["result", "token", "done"]


class _EmptyPreferences:
    async def list(self, _user_id):
        return []


async def _none():
    return None
