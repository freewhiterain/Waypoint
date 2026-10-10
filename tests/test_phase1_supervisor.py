import asyncio
from datetime import date

import pytest

from app.agents.supervisor import build_itinerary, run_travel_planning
from app.agents.workers.base import TravelWorker
from app.agents.workers.registry import WorkerRegistry
from app.schemas.planning import CandidateOption, ResearchTask, TravelRequirement, WorkerResult


def requirement():
    return TravelRequirement(
        origin="Shanghai",
        destination="Chengdu",
        departure_date=date(2026, 8, 1),
        days=5,
        adults=2,
        styles=["culture", "food"],
    )


@pytest.mark.asyncio
async def test_supervisor_returns_full_draft_without_fake_realtime_facts():
    draft = await run_travel_planning(requirement())

    assert len(draft.worker_results) == 5
    assert {result.worker for result in draft.worker_results} == {
        "attractions",
        "transport",
        "hotel",
        "food",
        "weather",
    }
    assert len(draft.itinerary) == 5
    assert [slot.period for slot in draft.itinerary[0].slots] == [
        "morning",
        "afternoon",
        "evening",
    ]
    assert "order" not in draft.model_dump_json().lower()
    assert "payment" not in draft.model_dump_json().lower()


def test_itinerary_reads_candidates_from_attractions_worker():
    attractions = WorkerResult(
        task_id="attractions-1",
        worker="attractions",
        status="completed",
        summary="ok",
        options=[CandidateOption(name="Wuhou Shrine", category="attractions")],
    )

    itinerary = build_itinerary(requirement(), [attractions])

    assert itinerary[0].slots[0].title == "Wuhou Shrine"


class CountingWorker(TravelWorker):
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def run(self, task: ResearchTask, requirement: TravelRequirement) -> WorkerResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return WorkerResult(
            task_id=task.id,
            worker=task.task_type,
            status="completed",
            summary="ok",
        )


@pytest.mark.asyncio
async def test_supervisor_executes_independent_workers_in_parallel():
    worker = CountingWorker()
    registry = WorkerRegistry(
        {
            "attractions": worker,
            "transport": worker,
            "hotel": worker,
            "food": worker,
            "weather": worker,
        }
    )

    draft = await run_travel_planning(requirement(), registry)

    assert len(draft.worker_results) == 5
    assert worker.max_active >= 2
