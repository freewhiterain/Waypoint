"""用户点名要去的地点：必须是用户说过的，候选里有就一定排进行程，没资料要说明。"""

from datetime import date

import pytest

from app.agents.itinerary_planner import PlaceSelection, SelectedPlace, check_itinerary, plan_itinerary
from app.evaluation.plan_checks import check_plan
from app.places.catalog import PlaceCatalog, named_by
from app.schemas.planning import TravelRequirement
from app.schemas.slots import SlotUpdate, TripSlots
from app.services.slots import apply_slot_update
from tests.test_itinerary_planner import LEGS, FakeLLM, chengdu_results

TODAY = date(2026, 10, 28)
SLOTS = TripSlots(destination="成都", departure_date=date(2026, 11, 2), days=2)


def test_must_visit_needs_the_user_to_name_it_and_yields_to_avoid():
    slots, dropped = apply_slot_update(
        SLOTS, SlotUpdate(must_visit=["宽窄巷子", "武侯祠"]), user_text="一定要去宽窄巷子", today=TODAY
    )
    assert slots.must_visit == ["宽窄巷子"] and "must_visit" in dropped

    slots, _ = apply_slot_update(slots, SlotUpdate(avoid_places=["宽窄巷子"]), user_text="宽窄巷子算了不去了", today=TODAY)
    assert slots.must_visit == [] and slots.avoid_places == ["宽窄巷子"]


def test_changing_city_clears_must_visit():
    slots = SLOTS.model_copy(update={"must_visit": ["宽窄巷子"]})

    slots, _ = apply_slot_update(slots, SlotUpdate(destination="重庆"), user_text="改去重庆", today=TODAY)

    assert slots.must_visit == []


def test_named_by_matches_short_and_full_names_but_ignores_branch_names():
    assert named_by("洪崖洞民俗风貌区", ["洪崖洞"])
    assert named_by("成都大熊猫繁育研究基地", ["成都大熊猫繁育研究基地"])
    assert not named_by("四季民福烤鸭店(故宫店)", ["故宫"])
    assert not named_by("锦里", ["宽"])
    assert named_by("秦始皇帝陵博物院(兵马俑)", ["兵马俑"])  # 括号里是别名
    assert not named_by("盛天毛血旺(国泰广场店)", ["国泰广场"])


def test_catalog_shortlist_puts_named_places_first_even_if_unpopular():
    catalog = PlaceCatalog()
    city = catalog.cities()[0]
    places = catalog.places(city, "attractions")
    obscure = places[-1].name
    requirement = TravelRequirement(destination=city, departure_date=TODAY, days=1, must_visit=[obscure])

    shortlist = catalog.shortlist(requirement, "attractions")

    assert shortlist[0].name == obscure
    assert obscure not in [record.name for record in catalog.shortlist(
        requirement.model_copy(update={"must_visit": []}), "attractions")]


@pytest.mark.asyncio
async def test_planner_adds_the_named_place_when_the_model_leaves_it_out():
    requirement = TravelRequirement(destination="成都", departure_date=date(2026, 11, 14), days=1, must_visit=["宽窄巷子"])
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="熊猫基地"), SelectedPlace(id="武侯祠")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, warnings = await plan_itinerary(requirement, chengdu_results(LEGS), llm=llm)

    titles = [slot.title for day in itinerary for slot in day.slots]
    assert "宽窄巷子" in titles
    assert not any("must_visit" in warning for warning in warnings)


def test_unknown_named_place_is_reported_instead_of_silently_dropped():
    requirement = TravelRequirement(destination="成都", departure_date=date(2026, 11, 14), days=1, must_visit=["不存在的景点"])

    violations = check_itinerary(requirement, chengdu_results(LEGS), [])
    report = check_plan(requirement, [], catalog=None)

    assert any(item.code == "must_visit" and "没查到" in item.message for item in violations)
    assert any("不存在的景点" in issue.message for issue in report)


def test_a_place_mentioned_with_a_negation_is_not_a_must_visit():
    slots, dropped = apply_slot_update(
        SLOTS, SlotUpdate(must_visit=["成都大熊猫繁育研究基地"]),
        user_text="行程里没安排熊猫基地，不过第二天想轻松点", today=TODAY,
    )
    assert slots.must_visit == [] and "must_visit" in dropped

    slots, _ = apply_slot_update(SLOTS, SlotUpdate(must_visit=["成都大熊猫繁育研究基地"]),
                                 user_text="熊猫基地一定要去，不想太累", today=TODAY)
    assert slots.must_visit == ["成都大熊猫繁育研究基地"]
