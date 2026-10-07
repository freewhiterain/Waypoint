import asyncio
from datetime import date

import pytest

from app.agents.subagents.registry import SubagentRegistry
from app.agents.subagents.registry import create_default_subagent_registry
from app.agents.supervisor import run_travel_planning
from app.governance.events import InMemoryEventRepository, TaskEventService
from app.schemas.planning import Evidence, ResearchTask, TravelRequirement
from app.schemas.research import Claim, EvidenceBoundCandidate, SubagentResponse


WORKERS = {"attractions", "weather", "transport", "hotel", "food"}


def chengdu_requirement() -> TravelRequirement:
    return TravelRequirement(
        origin="Shanghai",
        destination="Chengdu",
        departure_date=date(2026, 8, 1),
        days=3,
    )


class ParallelRecordingSubagent:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.started: list[str] = []
        self.completed: list[str] = []
        self._lock = asyncio.Lock()

    async def run(
        self,
        task: ResearchTask,
        requirement: TravelRequirement,
    ) -> SubagentResponse:
        async with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.started.append(task.task_type)
        await asyncio.sleep(0.02)
        async with self._lock:
            self.active -= 1
            self.completed.append(task.task_type)

        evidence_id = f"{task.id}-evidence"
        fact = f"{task.task_type} option for {requirement.destination}"
        return SubagentResponse(
            task_id=task.id,
            worker=task.task_type,
            status="completed",
            summary=fact,
            claims=[Claim(text=fact, evidence_ids=[evidence_id])],
            candidates=[
                EvidenceBoundCandidate(
                    id=f"{task.id}-candidate",
                    name=fact,
                    category=task.task_type,
                    description=fact,
                    evidence_ids=[evidence_id],
                )
            ],
            evidence=[
                Evidence(
                    id=evidence_id,
                    content=fact,
                    source="official",
                )
            ],
        )


@pytest.mark.asyncio
async def test_confirmed_trip_runs_four_subagents_in_parallel_then_transport_and_generates_draft():
    subagent = ParallelRecordingSubagent()
    registry = SubagentRegistry({worker: subagent for worker in WORKERS})

    result = await run_travel_planning(chengdu_requirement(), registry=registry)

    assert {item.worker for item in result.worker_results} == WORKERS
    assert result.itinerary
    assert result.warnings == []
    assert set(subagent.started) == WORKERS
    assert set(subagent.completed) == WORKERS
    # 第一批景点/住宿/美食/天气并行；交通依赖前三者的候选，最后单独跑。
    assert subagent.max_active == 4
    assert subagent.started[-1] == "transport"


@pytest.mark.asyncio
async def test_subagent_end_to_end_result_keeps_traceable_task_and_evidence_events():
    subagent = ParallelRecordingSubagent()
    registry = SubagentRegistry({worker: subagent for worker in WORKERS})
    events = InMemoryEventRepository()
    event_service = TaskEventService(events)

    result = await run_travel_planning(
        chengdu_requirement(),
        registry=registry,
        event_service=event_service,
        task_id="task-trace",
        user_id="user-trace",
        conversation_id="conversation-trace",
    )

    event_records = await events.list("task-trace", "user-trace")
    started = [event.payload for event in event_records if event.event_type == "worker_started"]
    completed = [event.payload for event in event_records if event.event_type == "worker_completed"]

    assert {payload["worker"] for payload in started} == WORKERS
    assert {payload["worker"] for payload in completed} == WORKERS
    assert {payload["task_id"] for payload in started} == {
        item.task_id for item in result.worker_results
    }
    evidence_ids = {item.id for item in result.evidence}
    for worker_result in result.worker_results:
        assert worker_result.task_id
        assert worker_result.evidence
        for option in worker_result.options:
            assert set(option.evidence_ids).issubset(evidence_ids)


@pytest.mark.asyncio
async def test_no_llm_or_provider_absence_returns_deterministic_degraded_draft(monkeypatch):
    monkeypatch.setattr("app.agents.supervisor.settings.llm_api_key", "")
    registry = create_default_subagent_registry(build_tools=lambda _worker: [], llm=None)

    result = await run_travel_planning(chengdu_requirement(), registry=registry)

    worker_results = {item.worker: item for item in result.worker_results}
    assert set(worker_results) == WORKERS
    assert result.itinerary
    assert result.status == "degraded"
    assert result.degraded_reason == "worker_unavailable"
    assert result.warnings
    assert all(item.status == "unavailable" for item in worker_results.values())
    assert all(item.warnings for item in worker_results.values())
    assert any("unavailable" in warning.lower() for warning in result.warnings)


@pytest.mark.asyncio
async def test_factory_accepts_supervisor_subagents_mode_and_builds_supervisor_graph(monkeypatch):
    from app.agents import factory
    from app.agents import supervisor

    calls = []

    class FakeRegistry:
        async def run(self, task, requirement):  # pragma: no cover - compile-time dependency only
            raise AssertionError("factory test should not execute graph workers")

    def create_registry():
        calls.append("registry")
        return FakeRegistry(), None

    monkeypatch.setattr(factory.settings, "travel_agent_mode", "supervisor_subagents")
    monkeypatch.setattr(factory, "create_planning_registry", create_registry)

    graph = await factory.create_chat_agent()

    assert graph is not None
    assert calls == ["registry"]


@pytest.mark.asyncio
async def test_factory_preserves_supervisor_mode_compatibility(monkeypatch):
    from app.agents import factory
    from app.agents import supervisor

    sentinel = object()
    captured = {}

    def create_graph(*, registry):
        captured["registry"] = registry
        return sentinel

    class FakeRegistry:
        async def run(self, task, requirement):
            raise AssertionError("compatibility test should not execute graph workers")

    monkeypatch.setattr(factory.settings, "travel_agent_mode", "supervisor")
    monkeypatch.setattr(factory, "create_planning_registry", lambda: (FakeRegistry(), None))
    monkeypatch.setattr(supervisor, "create_supervisor_graph", create_graph)

    assert await factory.create_chat_agent() is sentinel
    assert isinstance(captured["registry"], FakeRegistry)


@pytest.mark.asyncio
async def test_factory_still_rejects_unsupported_modes(monkeypatch):
    from app.agents import factory

    monkeypatch.setattr(factory.settings, "travel_agent_mode", "handoffs")

    with pytest.raises(ValueError, match="TRAVEL_AGENT_MODE=handoffs"):
        await factory.create_chat_agent()
