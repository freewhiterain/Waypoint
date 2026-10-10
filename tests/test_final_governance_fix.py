from datetime import date, datetime, timezone

import pytest

from app.agents.subagents.registry import SubagentRegistry
from app.agents.supervisor import run_travel_planning
from app.governance.evidence import EvidenceGovernanceService
from app.schemas.planning import Evidence, ResearchTask, TravelPlanDraft, TravelRequirement
from app.schemas.research import Claim, EvidenceBoundCandidate, SubagentResponse


WORKERS = ("attractions", "weather", "transport", "hotel", "food")


def _requirement() -> TravelRequirement:
    return TravelRequirement(
        origin="Shanghai",
        destination="Chengdu",
        departure_date=date(2026, 8, 1),
        days=3,
    )


def _task(worker: str) -> ResearchTask:
    return ResearchTask(id=f"{worker}-task", task_type=worker, query=f"Find {worker}")


class _CrossWorkerSubagent:
    async def run(self, task: ResearchTask, requirement: TravelRequirement) -> SubagentResponse:
        if task.task_type in {"attractions", "food"}:
            evidence = Evidence(
                id=f"{task.task_type}-shared",
                content="Shared place is open.",
                source="official" if task.task_type == "food" else "search",
                source_url="https://example.test/shared",
            )
        else:
            value = "open" if task.task_type == "weather" else "closed"
            evidence = Evidence(
                id=f"{task.task_type}-evidence",
                content=f"Cross-worker status is {value} for {task.task_type}.",
                source="official",
                source_url=f"https://example.test/{task.task_type}",
                metadata={"fact_key": "cross_worker_status", "fact_value": value},
            )
        return SubagentResponse(
            task_id=task.id,
            worker=task.task_type,
            status="completed",
            claims=[Claim(text=evidence.content, evidence_ids=[evidence.id])],
            candidates=[
                EvidenceBoundCandidate(
                    name=evidence.content,
                    category=task.task_type,
                    evidence_ids=[evidence.id],
                )
            ],
            evidence=[evidence],
        )


@pytest.mark.asyncio
async def test_supervisor_governs_all_five_responses_after_fan_in_with_traceability():
    worker = _CrossWorkerSubagent()
    registry = SubagentRegistry({name: worker for name in WORKERS})

    draft = await run_travel_planning(_requirement(), registry=registry)
    by_worker = {result.worker: result for result in draft.worker_results}

    assert {result.worker for result in draft.worker_results} == set(WORKERS)
    assert [item.id for item in draft.evidence if item.source_url == "https://example.test/shared"] == [
        "food-shared"
    ]
    assert by_worker["attractions"].options[0].evidence_ids == ["food-shared"]
    assert by_worker["food"].options[0].evidence_ids == ["food-shared"]
    assert all(result.task_id for result in draft.worker_results)
    assert "evidence_conflict:unresolved" in draft.warnings


@pytest.mark.asyncio
async def test_registry_and_supervisor_exception_warnings_are_stable_and_persisted():
    class ExplodingWorker:
        async def run(self, task, requirement):
            raise RuntimeError("secret provider payload")

    registry_result = await SubagentRegistry({"weather": ExplodingWorker()}).run(
        _task("weather"), _requirement()
    )
    assert registry_result.warnings == ["subagent_error:worker_execution_failed"]

    class ExplodingRegistry:
        async def run(self, task, requirement):
            raise RuntimeError("another secret provider payload")

    class RecordingEvents:
        def __init__(self):
            self.events = []

        async def emit(self, **event):
            self.events.append(event)

    events = RecordingEvents()
    await run_travel_planning(_requirement(), registry=ExplodingRegistry(), event_service=events)
    completed = [event for event in events.events if event["event_type"] == "worker_completed"]

    assert completed
    assert all(
        event["payload"]["warnings"] == ["subagent_error:supervisor_worker_failed"]
        for event in completed
    )
    assert "secret provider payload" not in repr(completed)


class _UnavailableSubagent:
    async def run(self, task, requirement):
        return SubagentResponse(
            task_id=task.id,
            worker=task.task_type,
            status="unavailable",
            warnings=["provider_unavailable"],
        )


@pytest.mark.asyncio
async def test_runtime_unavailability_degrades_draft_even_when_llm_is_configured(monkeypatch):
    worker = _UnavailableSubagent()
    registry = SubagentRegistry({name: worker for name in WORKERS})

    async def deterministic_synthesis(_requirement, _results, template):
        return template

    monkeypatch.setattr("app.agents.supervisor.settings.llm_api_key", "configured")
    monkeypatch.setattr("app.agents.supervisor.synthesize_itinerary_with_llm", deterministic_synthesis)

    draft = await run_travel_planning(_requirement(), registry=registry)

    assert draft.status == "degraded"
    assert draft.degraded_reason == "worker_unavailable"
    assert "planning_degraded:worker_unavailable" in draft.warnings


class _PartialSubagent:
    def __init__(self, status: str):
        self.status = status

    async def run(self, task, requirement):
        return SubagentResponse(
            task_id=task.id,
            worker=task.task_type,
            status=self.status,
            evidence=[Evidence(id=f"{task.id}-evidence", content="Local fact.", source="local")],
        )


@pytest.mark.asyncio
async def test_partial_runtime_provider_result_marks_draft_provider_degraded():
    registry = SubagentRegistry(
        {name: _PartialSubagent("partial" if name == "weather" else "completed") for name in WORKERS}
    )

    draft = await run_travel_planning(_requirement(), registry=registry)

    assert draft.status == "degraded"
    assert draft.degraded_reason == "provider_degraded"
    assert "planning_degraded:provider_degraded" in draft.warnings


def test_governance_rejects_external_evidence_without_url_and_future_valid_from():
    now = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
    evidence = [
        Evidence(
            id="no-url",
            content="External without URL.",
            source="official",
            metadata={"source_type": "external"},
        ),
        Evidence(
            id="future",
            content="Future evidence.",
            source="official",
            source_url="https://example.test/future",
            valid_from=datetime(2026, 8, 1, 13),
        ),
        Evidence(
            id="current",
            content="Current evidence.",
            source="official",
            source_url="https://example.test/current",
            valid_from=datetime(2026, 8, 1, 11),
        ),
        Evidence(id="local", content="Local evidence.", source="local"),
    ]
    response = SubagentResponse(
        task_id="task",
        worker="attractions",
        status="completed",
        claims=[Claim(text=item.content, evidence_ids=[item.id]) for item in evidence],
        evidence=evidence,
    )

    reviewed = EvidenceGovernanceService(now=now).review([response])

    assert {item.id for item in reviewed.evidence} == {"current", "local"}
    current = next(item for item in reviewed.evidence if item.id == "current")
    assert current.valid_from.tzinfo is not None
    assert {claim.evidence_ids[0] for claim in reviewed.claims} == {"current", "local"}


@pytest.mark.asyncio
async def test_factory_preserves_runtime_degraded_reason_when_configuration_also_degrades(monkeypatch):
    from app.agents import factory

    requirement = _requirement()
    runtime_draft = TravelPlanDraft(
        requirement=requirement,
        itinerary=[],
        worker_results=[],
        evidence=[],
        status="degraded",
        degraded_reason="worker_unavailable",
        warnings=["planning_degraded:worker_unavailable"],
    )

    monkeypatch.setattr(
        factory,
        "create_planning_registry",
        lambda: (object(), "no_llm_or_provider"),
    )

    async def fake_supervisor(_requirement, **kwargs):
        return runtime_draft

    monkeypatch.setattr("app.agents.supervisor.run_travel_planning", fake_supervisor)

    result = await factory.run_travel_planning(requirement)

    assert result.degraded_reason == "worker_unavailable"
    assert "planning_degraded:worker_unavailable" in result.warnings
    assert "planning_degraded:no_llm_or_provider" not in result.warnings
