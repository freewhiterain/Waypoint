"""餐饮候选：当地菜优先；不吃辣时连冬阴功这类东南亚菜也排除。"""

from collections import Counter
from datetime import date

from app.places.catalog import PlaceCatalog, PlaceRecord, _suits_dinner
from app.schemas.planning import TravelRequirement

DAY = date(2026, 11, 5)


def test_local_cuisine_leads_the_shortlist():
    catalog = PlaceCatalog()
    for city, local in (("上海", {"本帮菜", "江浙菜"}), ("广州", {"粤菜"}), ("北京", {"北京菜"})):
        food = catalog.shortlist(TravelRequirement(destination=city, departure_date=DAY, days=2), "food")
        types = Counter(record.type for record in food)
        assert sum(types[name] for name in (*local, "小吃")) * 2 > len(food), (city, types)


def test_a_named_cuisine_still_beats_local_food():
    catalog = PlaceCatalog()
    requirement = TravelRequirement(destination="上海", departure_date=DAY, days=2, food_preferences=["日本料理"])

    assert "日本料理" in {record.type for record in catalog.shortlist(requirement, "food")[:4]}


def test_thai_style_food_is_spicy_for_people_who_avoid_spice():
    thai = PlaceRecord(city="上海", category="food", rank=1, name="某泰餐", type="东南亚菜",
                       recommended_food="冬阴功面,菠萝炒饭,芒果糯米饭")
    requirement = TravelRequirement(destination="上海", departure_date=DAY, days=1, food_preferences=["不吃辣"])

    assert not _suits_dinner(thai, requirement)
    assert _suits_dinner(thai, requirement.model_copy(update={"food_preferences": []}))


def test_vegetarian_restaurants_only_for_people_who_asked():
    veg = PlaceRecord(city="上海", category="food", rank=1, name="宁国素斋", type="素食")
    requirement = TravelRequirement(destination="上海", departure_date=DAY, days=1)

    assert not _suits_dinner(veg, requirement)
    assert _suits_dinner(veg, requirement.model_copy(update={"food_preferences": ["吃素"]}))


def test_sichuan_snacks_with_dandan_noodles_count_as_spicy():
    snacks = PlaceRecord(city="成都", category="food", rank=1, name="某小吃店", type="小吃",
                         recommended_food="担担面,钵钵鸡,伤心凉粉,糖油果子")
    requirement = TravelRequirement(destination="成都", departure_date=DAY, days=1, food_preferences=["不吃辣"])

    assert not _suits_dinner(snacks, requirement)


def test_lunch_prefers_local_food_within_reach():
    from app.agents.itinerary_planner import TravelCosts, _pick_lunch
    from app.schemas.planning import CandidateOption, WorkerResult

    def food(name, cuisine):
        return CandidateOption(id=name, name=name, category="food", description="", evidence_ids=["e"],
                               attributes={"type": cuisine})

    def leg(a, b, minutes):
        return CandidateOption(id=f"{a}-{b}", name=f"{a}-{b}", category="travel_leg", description="",
                               evidence_ids=["e"], attributes={"from_id": a, "to_id": b, "minutes": minutes})

    options = {"粤菜馆": food("粤菜馆", "粤菜"), "川菜馆": food("川菜馆", "川菜"), "远川菜": food("远川菜", "川菜")}
    costs = TravelCosts([WorkerResult(task_id="t", worker="transport", status="completed", summary="", options=[
        leg("武侯祠", "粤菜馆", 5), leg("武侯祠", "川菜馆", 20), leg("武侯祠", "远川菜", 50)])])

    assert _pick_lunch(list(options), set(), "武侯祠", costs, options, "成都") == "川菜馆"
