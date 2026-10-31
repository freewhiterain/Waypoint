"""看过行程后的修改意见：结构化成"不去的地点 + 轻松的天"，作为重新排程的硬约束。"""

from datetime import date

import pytest

from app.agents.itinerary_planner import PlaceSelection, SelectedPlace, day_capacities, plan_itinerary
from app.schemas.planning import TravelRequirement
from app.schemas.slots import SlotUpdate, TripSlots
from app.services.slots import apply_slot_update
from app.services.trip_form import build_requirement
from tests.test_itinerary_planner import LEGS, FakeLLM, chengdu_results

TODAY = date(2026, 10, 28)
PLANNED = TripSlots(destination="成都", departure_date=date(2026, 11, 2), days=2)


def test_avoided_place_must_be_named_by_the_user():
    slots, dropped = apply_slot_update(
        PLANNED,
        SlotUpdate(avoid_places=["熊猫基地", "武侯祠"]),
        user_text="把熊猫基地换成别的",
        today=TODAY,
    )

    assert slots.avoid_places == ["熊猫基地"]
    assert "avoid_places" in dropped


def test_relaxed_days_need_a_pace_cue_and_accumulate():
    slots, _ = apply_slot_update(PLANNED, SlotUpdate(relaxed_days=[2]), user_text="第二天轻松一点", today=TODAY)
    slots, _ = apply_slot_update(slots, SlotUpdate(relaxed_days=[1]), user_text="第一天也少安排点", today=TODAY)
    unchanged, dropped = apply_slot_update(PLANNED, SlotUpdate(relaxed_days=[1]), user_text="第一天去哪", today=TODAY)

    assert slots.relaxed_days == [1, 2]
    assert unchanged.relaxed_days == [] and "relaxed_days" in dropped


def test_changing_city_clears_previous_revisions():
    revised = PLANNED.model_copy(update={"avoid_places": ["熊猫基地"], "relaxed_days": [2]})

    slots, _ = apply_slot_update(revised, SlotUpdate(destination="杭州"), user_text="算了改去杭州", today=TODAY)

    assert (slots.avoid_places, slots.relaxed_days) == ([], [])


def test_revisions_travel_with_the_form_into_the_requirement():
    revised = PLANNED.model_copy(update={"avoid_places": ["熊猫基地"], "relaxed_days": [2]})
    form = {"destination": "成都", "departure_date": date(2026, 11, 2), "days": 2}

    requirement = build_requirement(form, {"requirement_context": revised.requirement_context()})

    assert requirement.avoid_places == ["熊猫基地"]
    assert requirement.relaxed_days == [2]
    assert day_capacities(requirement) == [2, 1]


@pytest.mark.asyncio
async def test_avoided_place_never_appears_and_relaxed_day_gets_one_stop():
    # 模型仍然选了用户说不想去的熊猫基地：必须被丢掉，也不能被补空位补回来。
    requirement = TravelRequirement(
        destination="成都", departure_date=date(2026, 11, 14), days=2, avoid_places=["熊猫基地"], relaxed_days=[2]
    )
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="熊猫基地"), SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里"),
                     SelectedPlace(id="宽窄巷子")],
        dinners=[SelectedPlace(id="火锅"), SelectedPlace(id="小吃")],
    )])

    itinerary, _ = await plan_itinerary(requirement, chengdu_results(LEGS), llm=llm)

    titles = [slot.title for day in itinerary for slot in day.slots]
    assert "熊猫基地" not in titles
    day_two = [slot for slot in itinerary[1].slots if slot.period in ("morning", "afternoon")]
    assert sum(slot.title != "自由活动" for slot in day_two) == 1
    assert "轻松" in next(slot.description for slot in day_two if slot.title == "自由活动")


def test_full_place_name_completed_from_context_still_counts_as_mentioned():
    slots, dropped = apply_slot_update(
        PLANNED,
        SlotUpdate(avoid_places=["成都大熊猫繁育研究基地", "都江堰景区"]),
        user_text="把熊猫基地换成别的",
        today=TODAY,
    )

    assert slots.avoid_places == ["成都大熊猫繁育研究基地"]
    assert "avoid_places" in dropped


def test_travelling_with_older_parents_means_a_relaxed_pace_even_when_summarised():
    for styles in (["带父母"], ["爸妈年纪大了，别太赶"], ["长辈同行"]):
        requirement = TravelRequirement(destination="北京", departure_date=date(2026, 11, 2), days=3, styles=styles)
        assert day_capacities(requirement) == [1, 1, 1], styles
    assert day_capacities(TravelRequirement(destination="北京", departure_date=date(2026, 11, 2), days=2,
                                            styles=["历史"])) == [2, 2]


def test_other_ways_of_asking_for_a_lighter_day_count():
    for text in ("第二天别排太满", "第二天别太折腾", "第二天别那么赶"):
        slots, dropped = apply_slot_update(PLANNED, SlotUpdate(relaxed_days=[2]), user_text=text, today=TODAY)
        assert slots.relaxed_days == [2], text
