from datetime import date

import pytest

from app.agents.planner import create_research_plan, parallel_groups
from app.agents.subagents.registry import SubagentRegistry
from app.agents.supervisor import (
    _subagent_response_to_worker_result,
    merge_worker_results,
    run_travel_planning,
)
from app.schemas.planning import Evidence, ResearchTask, TravelRequirement
from app.schemas.research import Claim, EvidenceBoundCandidate, SubagentResponse


def _requirement() -> TravelRequirement:
    return TravelRequirement(
        origin="Shanghai",
        destination="Chengdu",
        departure_date=date(2026, 8, 1),
        days=3,
    )


def _response(task_id: str, worker: str, *, status: str = "completed") -> dict:
    return SubagentResponse(
        task_id=task_id,
        worker=worker,
        status=status,
        summary=f"{worker} done",
        claims=[Claim(text=f"{worker} claim", evidence_ids=[f"{task_id}-ev"])],
        candidates=[EvidenceBoundCandidate(name=f"{worker} option", evidence_ids=[f"{task_id}-ev"])],
        evidence=[Evidence(id=f"{task_id}-ev", content=f"{worker} claim", source="official")],
    ).model_dump(mode="json")


def test_supervisor_merges_parallel_results_by_task_id():
    merged = merge_worker_results({}, {"task-a": _response("task-a", "attractions")})
    merged = merge_worker_results(merged, {"task-b": _response("task-b", "weather")})

    assert set(merged) == {"task-a", "task-b"}


def test_supervisor_replaces_retry_result_for_same_task_id():
    merged = merge_worker_results({"task-a": _response("task-a", "attractions", status="failed")}, {})
    merged = merge_worker_results(merged, {"task-a": _response("task-a", "attractions", status="completed")})

    assert set(merged) == {"task-a"}
    assert merged["task-a"]["status"] == "completed"


def test_worker_result_summary_uses_only_governed_claims():
    response = SubagentResponse(
        task_id="task-a",
        worker="attractions",
        status="completed",
        summary="Unsupported summary should not reach synthesis.",
        claims=[
            Claim(text="Supported museum fact.", evidence_ids=["ev-supported"]),
            Claim(text="Unsupported museum fact.", evidence_ids=["missing"]),
        ],
        evidence=[Evidence(id="ev-supported", content="Museum source text.", source="official")],
    )

    result = _subagent_response_to_worker_result(response)

    assert result.summary == "Supported museum fact."
    assert "Unsupported" not in result.summary


def test_failed_worker_result_summary_uses_fixed_non_factual_text():
    response = SubagentResponse(
        task_id="task-a",
        worker="attractions",
        status="failed",
        summary="Unsupported factual summary: Panda Base tickets are sold out today.",
        warnings=["TimeoutError: attractions provider crashed"],
    )

    result = _subagent_response_to_worker_result(response)

    assert result.summary == "Domain subagent execution failed."
    assert "Panda Base tickets" not in result.summary
    assert result.warnings == ["TimeoutError: attractions provider crashed"]


def test_confirmed_destination_plan_creates_two_task_groups():
    tasks = create_research_plan(_requirement())

    assert len(tasks) == 5
    assert [task.task_type for task in tasks if task.dependencies] == ["transport"]
    assert len(parallel_groups(tasks)) == 2


class FailingWeatherSubagent:
    async def run(self, task: ResearchTask, requirement: TravelRequirement) -> SubagentResponse:
        raise RuntimeError("weather provider crashed")


class MalformedAttractionsSubagent:
    async def run(self, task: ResearchTask, requirement: TravelRequirement) -> SubagentResponse:
        return SubagentResponse.model_construct(
            task_id=task.id,
            worker=task.task_type,
            status="completed",
            summary="malformed attractions response",
            claims=[object()],
            candidates=[],
            evidence=[],
            research_report=None,
            warnings=[],
        )


class StaticSubagent:
    async def run(self, task: ResearchTask, requirement: TravelRequirement) -> SubagentResponse:
        return SubagentResponse(
            task_id=task.id,
            worker=task.task_type,
            status="completed",
            summary=f"{task.task_type} ok",
            claims=[Claim(text=f"{task.task_type} evidence", evidence_ids=[f"{task.id}-ev"])],
            candidates=[EvidenceBoundCandidate(name=f"{task.task_type} option", evidence_ids=[f"{task.id}-ev"])],
            evidence=[Evidence(id=f"{task.id}-ev", content=f"{task.task_type} evidence", source="official")],
        )


@pytest.mark.asyncio
async def test_subagent_failure_does_not_stop_other_branches():
    shared = StaticSubagent()
    registry = SubagentRegistry(
        {
            "attractions": shared,
            "transport": shared,
            "hotel": shared,
            "food": shared,
            "weather": FailingWeatherSubagent(),
        }
    )

    draft = await run_travel_planning(_requirement(), registry)
    results = {result.worker: result for result in draft.worker_results}

    assert set(results) == {"attractions", "transport", "hotel", "food", "weather"}
    assert results["weather"].status == "failed"
    assert all(results[worker].status == "completed" for worker in {"attractions", "transport", "hotel", "food"})


@pytest.mark.asyncio
async def test_governance_failure_does_not_stop_other_branches():
    shared = StaticSubagent()
    registry = SubagentRegistry(
        {
            "attractions": MalformedAttractionsSubagent(),
            "transport": shared,
            "hotel": shared,
            "food": shared,
            "weather": shared,
        }
    )

    draft = await run_travel_planning(_requirement(), registry)
    results = {result.worker: result for result in draft.worker_results}

    assert set(results) == {"attractions", "transport", "hotel", "food", "weather"}
    assert results["attractions"].status == "failed"
    assert all(results[worker].status == "completed" for worker in {"transport", "hotel", "food", "weather"})
