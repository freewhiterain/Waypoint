from datetime import date

from app.agents.scheduling import calculate_budget, schedule_itinerary
from app.schemas.planning import (
    BudgetSummary,
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
        budget=800,
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


def test_budget_sums_grounded_prices_and_marks_missing_categories():
    results = [
        result("attractions", [option("Panda Base", "attractions", 100, "Chenghua"), option("Wenshu", "attractions", 80, "Qingyang")]),
        result("food", [option("Sichuan dinner", "food", 50, "Jinjiang")]),
        result("hotel", [
            CandidateOption(
                name="City hotel",
                category="hotel",
                estimated_cost=200,
                attributes={"pricing_unit": "per_night"},
                evidence_ids=["hotel-evidence"],
            )
        ]),
        result("transport", [option("Rail", "transport", 20, "Chengdu")]),
    ]

    budget = calculate_budget(requirement(), results)

    # 2 天行程住 1 晚；两顿晚餐各 50；两个景点都排进了第 1 天。
    assert budget.categories == {
        "transport": 20.0,
        "accommodation": 200.0,
        "food": 100.0,
        "attractions": 180.0,
        "misc": None,
    }
    assert budget.total_estimate == 500.0
    # 文案面向终端用户，必须和 render_plan_markdown 的中文正文同语言。
    assert any("预算" in note for note in budget.notes)
    assert any("800.00 元" in note for note in budget.notes)


def test_budget_counts_only_scheduled_items_for_every_traveller_and_one_hotel():
    party = requirement().model_copy(update={"days": 1, "adults": 2, "children": 1, "budget": None})
    results = [
        result("attractions", [
            option("Panda Base", "attractions", 100, "Chenghua"),
            option("Wenshu", "attractions", 80, "Qingyang"),
            option("Not scheduled", "attractions", 999, "Far"),
        ]),
        result("food", [option("Sichuan dinner", "food", 50, "Jinjiang")]),
        result("hotel", [
            CandidateOption(name="Pricey", category="hotel", estimated_cost=900,
                            attributes={"pricing_unit": "per_night"}, evidence_ids=["hotel-evidence"]),
        ]),
    ]

    budget = calculate_budget(party, results)

    assert budget.categories["attractions"] == 540.0  # (100 + 80) × 3 人，第 3 个景点没排进行程
    assert budget.categories["food"] == 150.0
    assert budget.categories["accommodation"] == 0.0  # 当天往返
    assert any("共 3 人" in note for note in budget.notes)


def test_budget_uses_cheapest_hotel_with_nights_and_rooms_and_flags_overspend():
    trip = requirement().model_copy(update={"days": 3, "adults": 3, "budget": 500})
    results = [
        result("hotel", [
            CandidateOption(name="A", category="hotel", estimated_cost=300,
                            attributes={"pricing_unit": "per_night"}, evidence_ids=["hotel-evidence"]),
            CandidateOption(name="B", category="hotel", estimated_cost=200,
                            attributes={"pricing_unit": "per_night"}, evidence_ids=["hotel-evidence"]),
        ]),
    ]

    budget = calculate_budget(trip, results)

    assert budget.categories["accommodation"] == 800.0  # 200 × 2 晚 × 2 间房
    assert any("超出预算 300.00 元" in note for note in budget.notes)
