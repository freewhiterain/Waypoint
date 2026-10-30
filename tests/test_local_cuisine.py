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
