from datetime import date

from app.agents.scheduling import schedule_itinerary
from app.schemas.planning import (
    CandidateOption,
    Evidence,
    TravelRequirement,
    WorkerResult,
)


def requirement() -> TravelRequirement:
    return TravelRequirement(
        destination="Chengdu",
        departure_date=date(2026, 8, 1),
        days=2,
    )


def result(worker: str, options: list[CandidateOption]) -> WorkerResult:
    return WorkerResult(
        task_id=f"{worker}-task",
        worker=worker,
        status="completed",
        summary="grounded result",
        options=options,
        evidence=[Evidence(id=f"{worker}-evidence", content="supported", source="official")],
    )


def option(name: str, category: str, cost: float, location: str) -> CandidateOption:
    return CandidateOption(
        name=name,
        category=category,
        estimated_cost=cost,
        attributes={"location": location, "travel_minutes": 20, "opening_hours": "09:00-17:00"},
        evidence_ids=[f"{category}-evidence"],
    )


def test_scheduler_uses_distinct_grounded_places_and_preserves_constraints():
    results = [
        result("attractions", [option("Panda Base", "attractions", 100, "Chenghua") , option("Wenshu", "attractions", 80, "Qingyang")]),
        result("food", [option("Sichuan dinner", "food", 50, "Jinjiang")]),
    ]

    itinerary, warnings = schedule_itinerary(requirement(), results)

    assert len(itinerary) == 2
    assert itinerary[0].slots[0].title == "Panda Base"
    assert itinerary[0].slots[1].title == "Wenshu"
    assert itinerary[0].slots[0].location == "Chenghua"
    assert itinerary[0].slots[0].travel_minutes == 20
    assert itinerary[0].slots[0].opening_window == "09:00-17:00"
    assert itinerary[1].slots[0].title not in {"Panda Base", "Wenshu"}
    assert itinerary[1].notes == ["第2天已无未使用的景点候选。"]
    assert warnings == []


