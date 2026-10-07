"""LLM 选点 → 算法排程 → 约束检查（LLM 全部用假对象，不访问网络）。"""

from datetime import date

import pytest

from app.agents.itinerary_planner import (
    PlaceSelection,
    SelectedPlace,
    TravelCosts,
    assign_days,
    check_itinerary,
    plan_itinerary,
)
from app.schemas.planning import CandidateOption, TravelRequirement, WorkerResult


def requirement(**updates) -> TravelRequirement:
    values = {"destination": "成都", "departure_date": date(2026, 11, 14), "days": 2}
    values.update(updates)
    return TravelRequirement(**values)


def option(option_id, category, cost=None, **attributes) -> CandidateOption:
    return CandidateOption(id=option_id, name=option_id, category=category, description=f"{option_id}的介绍",
                           estimated_cost=cost, attributes=attributes, evidence_ids=["e"])


def leg(a, b, minutes=None, proximity="unknown") -> CandidateOption:
    return option(f"{a}-{b}", "travel_leg", from_id=a, to_id=b, minutes=minutes, proximity=proximity)


def result(worker, options, summary="") -> WorkerResult:
    return WorkerResult(task_id=f"{worker}-t", worker=worker, status="completed", summary=summary, options=options)


def chengdu_results(transport_legs):
    return [
        result("attractions", [option("熊猫基地", "attractions"), option("武侯祠", "attractions"),
                               option("宽窄巷子", "attractions"), option("锦里", "attractions")]),
        result("food", [option("火锅", "food"), option("小吃", "food")]),
        result("hotel", [option("武侯酒店", "hotel")]),
        result("transport", transport_legs),
    ]


# 武侯祠、锦里、武侯酒店同在城南；熊猫基地在城北、离宽窄巷子较近。
LEGS = [
    leg("武侯祠", "锦里", 8), leg("武侯酒店", "武侯祠", 5), leg("武侯酒店", "锦里", 10),
    leg("熊猫基地", "宽窄巷子", 30), leg("武侯酒店", "熊猫基地", 50), leg("武侯酒店", "宽窄巷子", 25),
    leg("熊猫基地", "武侯祠", 55), leg("熊猫基地", "锦里", 55), leg("宽窄巷子", "武侯祠", 20),
    leg("宽窄巷子", "锦里", 22), leg("武侯祠", "火锅", 10), leg("宽窄巷子", "小吃", 5),
]


def test_days_group_nearby_places_starting_from_the_hotel():
    costs = TravelCosts(chengdu_results(LEGS))

    days = assign_days(["熊猫基地", "武侯祠", "宽窄巷子", "锦里"], {}, 2, "武侯酒店", costs)

    assert days == [["武侯祠", "锦里"], ["宽窄巷子", "熊猫基地"]]


def test_preferred_day_is_respected():
    costs = TravelCosts(chengdu_results(LEGS))

    days = assign_days(["熊猫基地", "武侯祠", "宽窄巷子", "锦里"], {"熊猫基地": 1}, 2, "武侯酒店", costs)

    assert "熊猫基地" in days[0]


class FakeLLM:
    def __init__(self, selections):
        self.selections = list(selections)
        self.prompts = []

    def with_structured_output(self, schema):
        assert schema is PlaceSelection
        return self

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1]["content"])
        return self.selections.pop(0)


@pytest.mark.asyncio
async def test_llm_selection_is_sanitized_scheduled_and_given_real_travel_times():
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="锦里"), SelectedPlace(id="不存在的景点"), SelectedPlace(id="武侯祠")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, warnings = await plan_itinerary(requirement(days=1), chengdu_results(LEGS), llm=llm)

    morning, afternoon, evening = itinerary[0].slots
    assert (morning.title, afternoon.title, evening.title) == ("武侯祠", "锦里", "火锅")
    assert (morning.travel_minutes, afternoon.travel_minutes) == (5, 8)
    assert "itinerary_selection_dropped:attractions" in warnings


@pytest.mark.asyncio
async def test_violations_are_fed_back_to_the_llm_for_a_second_round():
    expensive = [
        result("attractions", [option("贵景点", "attractions", 3000), option("便宜景点", "attractions", 50)]),
        result("food", [option("火锅", "food", 100)]),
    ]
    llm = FakeLLM([
        PlaceSelection(attractions=[SelectedPlace(id="贵景点")], dinners=[SelectedPlace(id="火锅")]),
        PlaceSelection(attractions=[SelectedPlace(id="便宜景点")], dinners=[SelectedPlace(id="火锅")]),
    ])

    itinerary, warnings = await plan_itinerary(requirement(days=1, budget=500), expensive, llm=llm)

    assert itinerary[0].slots[0].title == "便宜景点"
    assert len(llm.prompts) == 2 and "超出预算" in llm.prompts[1]
    assert not any(warning.startswith("itinerary_constraint") for warning in warnings)


@pytest.mark.asyncio
async def test_rule_fallback_keeps_cutting_until_within_budget():
    pricey = [result("attractions", [option("A", "attractions", 300), option("B", "attractions", 300)])]

    itinerary, warnings = await plan_itinerary(requirement(days=1, budget=400), pricey)

    titles = [slot.title for slot in itinerary[0].slots]
    assert sum(title in {"A", "B"} for title in titles) == 1
    assert not any(warning.startswith("itinerary_constraint") for warning in warnings)


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_rules():
    class BrokenLLM(FakeLLM):
        async def ainvoke(self, messages):
            raise TimeoutError("model down")

    itinerary, warnings = await plan_itinerary(requirement(days=1), chengdu_results(LEGS), llm=BrokenLLM([]))

    assert itinerary[0].slots[0].title in {"熊猫基地", "武侯祠", "宽窄巷子", "锦里"}
    assert "itinerary_selection_fallback:llm_failed" in warnings


def test_checker_flags_opening_hours_and_long_travel_days():
    results = [result("attractions", [option("夜市", "attractions", opening_hours="18:00-23:00")])]
    selection_itinerary = [
        day.model_copy(update={"slots": [
            day.slots[0].model_copy(update={"opening_window": "18:00-23:00", "travel_minutes": 200}),
            *day.slots[1:],
        ]})
        for day in __import__("app.agents.scheduling", fromlist=["x"]).schedule_itinerary(requirement(days=1), results)[0]
    ]

    codes = {violation.code for violation in check_itinerary(requirement(days=1), results, selection_itinerary)}

    assert codes == {"opening_hours", "daily_travel"}


@pytest.mark.asyncio
async def test_district_estimates_order_places_without_showing_invented_minutes():
    legs = [leg("A", "B", proximity="different_district"), leg("A", "C", proximity="same_district"),
            leg("B", "C", proximity="different_district")]
    results = [result("attractions", [option("A", "attractions"), option("B", "attractions"),
                                      option("C", "attractions")]), result("transport", legs)]

    itinerary, _ = await plan_itinerary(requirement(days=2), results)

    assert [slot.title for slot in itinerary[0].slots[:2]] == ["A", "C"]  # 同区的排在一天
    assert all(slot.travel_minutes is None for day in itinerary for slot in day.slots)
