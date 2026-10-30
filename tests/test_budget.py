"""总预算：要有说预算的话才记；候选按预算筛酒店和餐厅；超了反馈给模型并告诉用户。"""

from datetime import date

import pytest

from app.agents.itinerary_planner import PlaceSelection, SelectedPlace, plan_itinerary
from app.agents.trip_cost import estimate_trip_cost, hotel_price
from app.evaluation.plan_checks import check_plan
from app.places.catalog import PlaceCatalog
from app.schemas.planning import CandidateOption, ItineraryDay, TimeSlot, TravelRequirement, WorkerResult
from app.schemas.slots import SlotUpdate, TripSlots
from app.services.slots import apply_slot_update
from tests.test_itinerary_planner import LEGS, FakeLLM, chengdu_results

TODAY = date(2026, 10, 30)
SLOTS = TripSlots(destination="成都", departure_date=date(2026, 11, 2), days=2)


@pytest.mark.parametrize(
    ("text", "kept"),
    [("预算 3000", True), ("两个人两千块以内", True), ("人均 1.5k", True), ("帮我规划成都两天", False)],
)
def test_budget_needs_a_budget_cue(text, kept):
    slots, dropped = apply_slot_update(SLOTS, SlotUpdate(budget=3000), user_text=text, today=TODAY)

    assert (slots.budget == 3000) is kept
    assert ("budget" in dropped) is not kept


def test_changing_city_clears_the_budget():
    slots = SLOTS.model_copy(update={"budget": 3000})

    slots, _ = apply_slot_update(slots, SlotUpdate(destination="杭州"), user_text="改去杭州", today=TODAY)

    assert slots.budget is None


def test_cost_counts_tickets_and_meals_per_person_and_hotel_per_night():
    requirement = TravelRequirement(destination="成都", departure_date=TODAY, days=2, adults=2)
    itinerary = [
        ItineraryDay(day=1, date=TODAY, slots=[TimeSlot(period="morning", title="武侯祠", estimated_cost=50),
                                              TimeSlot(period="evening", title="火锅", estimated_cost=100)]),
        ItineraryDay(day=2, date=TODAY, slots=[TimeSlot(period="morning", title="锦里"),
                                              TimeSlot(period="evening", title="自由安排晚餐")]),
    ]

    cost = estimate_trip_cost(requirement, itinerary, hotel_price=300)

    assert (cost.tickets, cost.meals, cost.hotel, cost.total, cost.unknown) == (100, 200, 300, 600, 1)


def test_hotel_price_is_read_from_the_hotel_note():
    day = ItineraryDay(day=1, date=TODAY, slots=[], notes=["天气：晴", "住宿：桔子酒店。近地铁"])
    hotels = WorkerResult(task_id="h", worker="hotel", status="completed", summary="", options=[
        CandidateOption(id="h1", name="桔子酒店", category="hotel", description="", estimated_cost=388, evidence_ids=["e"])])

    assert hotel_price([day], [hotels]) == 388


def test_tight_budget_shortlists_only_affordable_hotels_and_cheaper_food():
    catalog = PlaceCatalog()
    loose = TravelRequirement(destination="上海", departure_date=TODAY, days=3)
    tight = loose.model_copy(update={"budget": 1500})

    hotels = catalog.shortlist(tight, "hotel")
    food = catalog.shortlist(tight, "food")

    assert all(record.price <= 1500 * 0.45 / 2 for record in hotels)
    assert max(r.price or 0 for r in food) <= max(r.price or 0 for r in catalog.shortlist(loose, "food"))
    assert all(record.price is None or record.price <= 1500 * 0.3 / 3 for record in food)


@pytest.mark.asyncio
async def test_over_budget_plan_is_sent_back_and_reported():
    results = chengdu_results(LEGS)
    food = next(item for item in results if item.worker == "food")
    food.options[0] = food.options[0].model_copy(update={"estimated_cost": 5000})
    requirement = TravelRequirement(destination="成都", departure_date=date(2026, 11, 14), days=1, budget=500)
    pricey = PlaceSelection(hotel_id="武侯酒店", attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里")],
                            dinners=[SelectedPlace(id="火锅")])
    llm = FakeLLM([pricey, pricey, pricey])

    _, warnings = await plan_itinerary(requirement, results, llm=llm)

    assert len(llm.prompts) == 3 and "超出预算" in llm.prompts[-1]  # 每轮都把超预算反馈给模型
    assert any(w.startswith("itinerary_constraint:budget:") and "超出预算 500 元" in w for w in warnings)


def test_plan_check_flags_an_over_budget_plan():
    requirement = TravelRequirement(destination="大理", departure_date=TODAY, days=1, budget=100)
    day = ItineraryDay(day=1, date=TODAY, slots=[TimeSlot(period="evening", title="饭馆", estimated_cost=150)])

    assert any("超出预算" in issue.message for issue in check_plan(requirement, [day]))
