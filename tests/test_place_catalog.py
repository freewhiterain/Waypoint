"""地点库：读取、筛选候选、接到子 Agent 和排日程上（不访问网络和数据库）。"""

from datetime import date

import pytest

from app.agents.itinerary_planner import TravelCosts, check_itinerary, plan_itinerary
from app.agents.subagents.attractions import AttractionsSubagent
from app.agents.subagents.food import FoodSubagent
from app.agents.travel_time import Place, StraightLineProvider, describe_leg
from app.places.catalog import PlaceCatalog, get_place_catalog, straight_line_km
from app.schemas.planning import (
    CandidateOption,
    ItineraryDay,
    ResearchTask,
    TimeSlot,
    TravelRequirement,
    WorkerResult,
)

ATTRACTIONS = """city,rank,name,type,latitude,longitude,open_time,close_time,price,min_hours,max_hours
成都,0,熊猫基地,游乐园/体育娱乐,30.747,104.146,07:30,17:00,55.0,0.5,2.0
成都,1,都江堰,自然风光,31.008,103.618,08:00,18:00,80.0,0.0,1.5
成都,2,武侯祠,历史古迹,30.653,104.055,08:30,19:00,50.0,2.0,3.5
成都,3,人民公园,公园,30.657,104.058,00:00,24:00,0.0,1.0,2.0
成都,4,冷门古迹甲,历史古迹,30.60,104.00,09:00,17:00,10.0,1.0,2.0
成都,5,冷门古迹乙,历史古迹,30.61,104.01,09:00,17:00,10.0,1.0,2.0
成都,6,冷门古迹丙,历史古迹,30.62,104.02,09:00,17:00,10.0,1.0,2.0
成都,7,冷门古迹丁,历史古迹,30.63,104.03,09:00,17:00,10.0,1.0,2.0
成都,8,冷门古迹戊,历史古迹,30.64,104.04,09:00,17:00,10.0,1.0,2.0
成都,9,春熙路,商业街区,30.655,104.08,00:00,24:00,0.0,1.0,2.0
"""
RESTAURANTS = """city,rank,name,type,latitude,longitude,open_time,close_time,price,recommended_food
成都,0,鹤鸣茶社,茶馆/茶室,30.657,104.059,08:00,22:00,45.0,
成都,1,马旺子,川菜,30.651,104.084,11:30,20:30,140.0,"口水鸡,夫妻肺片"
成都,2,早午餐店,西餐,30.65,104.08,08:00,15:00,90.0,
成都,3,粤菜馆,粤菜,30.66,104.07,11:00,21:30,160.0,"烧鹅,虾饺"
成都,4,深夜烧烤,烧烤,30.66,104.06,18:00,02:00,80.0,
成都,5,泰式海鲜火锅,东南亚菜,30.66,104.06,11:30,22:00,120.0,
"""
HOTELS = """city,rank,name,type,latitude,longitude,price,beds
成都,0,城市名人酒店,桑拿,30.654,104.065,353,2
"""


@pytest.fixture
def catalog(tmp_path):
    (tmp_path / "attractions.csv").write_text(ATTRACTIONS, encoding="utf-8")
    (tmp_path / "restaurants.csv").write_text(RESTAURANTS, encoding="utf-8")
    (tmp_path / "accommodations.csv").write_text(HOTELS, encoding="utf-8")
    return PlaceCatalog(tmp_path)


def requirement(**updates) -> TravelRequirement:
    values = {"destination": "成都", "departure_date": date(2026, 11, 14), "days": 2}
    values.update(updates)
    return TravelRequirement(**values)


def task(worker):
    return ResearchTask(id=f"{worker}-t", task_type=worker, query="q")


# ---------------------------------------------------------------- 读取和筛选


def test_places_come_back_in_popularity_order_and_unknown_city_is_empty(catalog):
    assert [record.name for record in catalog.places("成都市", "attractions")][:3] == ["熊猫基地", "都江堰", "武侯祠"]
    assert catalog.places("漠河", "attractions") == []
    assert catalog.shortlist(requirement(destination="漠河"), "attractions") == []


def test_record_text_and_attributes_carry_the_structured_facts(catalog):
    panda, _, _, park = catalog.places("成都", "attractions")[:4]

    assert panda.describe() == "游乐园/体育娱乐，开放时间 07:30-17:00，门票 55 元。"
    assert panda.attributes()["opening_hours"] == "07:30-17:00"
    assert (panda.attributes()["latitude"], panda.attributes()["longitude"]) == (30.747, 104.146)
    assert park.describe() == "公园，全天开放，免费。"
    assert catalog.places("成都", "food")[1].describe() == "川菜，人均 140 元，营业时间 11:30-20:30，推荐菜：口水鸡、夫妻肺片。"
    assert catalog.places("成都", "hotel")[0].describe() == "住宿，特色：桑拿，参考价 353 元/晚。"


def test_preference_matches_take_at_most_half_the_shortlist(catalog):
    names = [record.name for record in catalog.shortlist(requirement(styles=["历史"]), "attractions")]

    # 8 个名额：命中"历史古迹"的最多 4 个排在前面，其余按热门程度补。
    assert names[:4] == ["武侯祠", "冷门古迹甲", "冷门古迹乙", "冷门古迹丙"]
    assert names[4:] == ["熊猫基地", "都江堰", "人民公园", "冷门古迹丁"]


def test_dinner_shortlist_drops_teahouses_early_closers_and_spicy_food_when_asked(catalog):
    names = [record.name for record in catalog.shortlist(requirement(), "food")]
    assert names == ["马旺子", "粤菜馆", "深夜烧烤", "泰式海鲜火锅"]

    mild = [record.name for record in catalog.shortlist(requirement(food_preferences=["不吃辣"]), "food")]
    assert mild == ["粤菜馆", "深夜烧烤"]


def test_bundled_data_covers_ten_cities_with_usable_fields():
    bundled = get_place_catalog()

    for city in ("北京", "上海", "广州", "深圳", "成都", "重庆", "杭州", "南京", "苏州", "武汉"):
        assert len(bundled.places(city, "attractions")) > 300
        assert len(bundled.places(city, "food")) > 400
        assert len(bundled.places(city, "hotel")) > 250
    panda = bundled.places("成都", "attractions")[0]
    assert panda.name == "成都大熊猫繁育研究基地" and panda.latitude and panda.opening_hours == "07:30-17:00"


# ---------------------------------------------------------------- 子 Agent


class _MustNotBeCalled:
    def with_structured_output(self, schema):
        raise AssertionError("地点库覆盖的城市不应该调用模型提取候选")


async def _no_tools(*args, **kwargs):
    raise AssertionError("地点库覆盖的城市不应该走检索")


@pytest.mark.asyncio
async def test_covered_city_returns_catalog_candidates_without_retrieval_or_llm(catalog):
    events = []

    async def on_event(kind, payload):
        events.append((kind, payload["tool_name"]))

    agent = AttractionsSubagent(catalog=catalog, llm=_MustNotBeCalled(), tool_builder=_no_tools)
    response = await agent.run(task("attractions"), requirement(), event_callback=on_event)

    assert response.status == "completed"
    assert [candidate.name for candidate in response.candidates][:2] == ["熊猫基地", "都江堰"]
    first = response.candidates[0]
    assert first.estimated_cost == 55.0 and first.attributes["opening_hours"] == "07:30-17:00"
    assert first.evidence_ids == [response.evidence[0].id]
    assert response.evidence[0].content == "熊猫基地：游乐园/体育娱乐，开放时间 07:30-17:00，门票 55 元。"
    assert events == [("subagent_tool_call", "place_catalog"), ("subagent_tool_completed", "place_catalog")]


@pytest.mark.asyncio
async def test_uncovered_city_falls_back_to_the_retrieval_flow(catalog):
    called = []

    async def tools(*args, **kwargs):
        called.append(True)
        return []

    response = await FoodSubagent(catalog=catalog, tool_builder=tools).run(task("food"), requirement(destination="漠河"))

    assert called and response.status == "unavailable"


# ---------------------------------------------------------------- 交通和排日程


@pytest.mark.asyncio
async def test_straight_line_provider_gives_distance_but_never_minutes():
    near = Place(id="a", name="武侯祠", category="attractions", latitude=30.653, longitude=104.055)
    far = Place(id="b", name="都江堰", category="attractions", latitude=31.008, longitude=103.618)
    unknown = Place(id="c", name="某地", category="attractions")

    legs = await StraightLineProvider().legs("成都", [(near, far), (near, unknown)])

    assert len(legs) == 1 and legs[0].minutes is None
    assert 55 < legs[0].distance_km < 60
    assert "直线距离约" in describe_leg(legs[0], near, far) and "分钟" not in describe_leg(legs[0], near, far)
    assert straight_line_km((30.0, 104.0), (30.0, 104.0)) == 0


def _option(name, worker, latitude, longitude, **attributes):
    return CandidateOption(id=name, name=name, category=worker, description=f"{name}的介绍。", evidence_ids=["e"],
                           attributes={"latitude": latitude, "longitude": longitude, **attributes})


def _results():
    return [
        WorkerResult(task_id="a", worker="attractions", status="completed", summary="", options=[
            _option("都江堰", "attractions", 31.008, 103.618),
            _option("武侯祠", "attractions", 30.653, 104.055),
            _option("人民公园", "attractions", 30.657, 104.058),
            _option("青城山", "attractions", 30.90, 103.57),
        ]),
        WorkerResult(task_id="f", worker="food", status="completed", summary="", options=[
            _option("城里的餐厅", "food", 30.655, 104.06), _option("都江堰的餐厅", "food", 31.0, 103.62),
        ]),
        WorkerResult(task_id="h", worker="hotel", status="completed", summary="", options=[
            _option("市中心酒店", "hotel", 30.654, 104.065),
        ]),
    ]


def test_costs_use_coordinates_when_there_is_no_route_data():
    costs = TravelCosts(_results())

    assert costs.minutes("武侯祠", "人民公园") is None  # 展示给用户的耗时仍然只来自真实数据
    assert costs.cost("武侯祠", "人民公园") < 10 < 150 < costs.cost("武侯祠", "都江堰")


@pytest.mark.asyncio
async def test_rule_planning_groups_nearby_places_and_notes_the_hotel():
    itinerary, _warnings = await plan_itinerary(requirement(days=2), _results())

    day_one = [slot.title for slot in itinerary[0].slots]
    day_two = [slot.title for slot in itinerary[1].slots]
    assert set(day_one[:2]) == {"武侯祠", "人民公园"} and day_one[2] == "城里的餐厅"
    assert set(day_two[:2]) == {"都江堰", "青城山"} and day_two[2] == "都江堰的餐厅"
    assert itinerary[0].notes[0] == "住宿：市中心酒店。市中心酒店的介绍。"


def test_checker_flags_a_day_whose_stops_are_far_apart_by_coordinates():
    day = ItineraryDay(day=1, date=date(2026, 11, 14), slots=[
        TimeSlot(period="morning", title="武侯祠"),
        TimeSlot(period="afternoon", title="都江堰"),
        TimeSlot(period="evening", title="城里的餐厅"),
    ])

    violations = check_itinerary(requirement(days=1), _results(), [day])

    assert [violation.code for violation in violations] == ["daily_travel"]
    assert "相距太远" in violations[0].message


def test_checker_allows_a_day_trip_out_of_town_followed_by_dinner_back_in_the_city():
    day = ItineraryDay(day=1, date=date(2026, 11, 14), slots=[
        TimeSlot(period="morning", title="都江堰"),
        TimeSlot(period="afternoon", title="青城山"),
        TimeSlot(period="evening", title="城里的餐厅"),
    ])

    assert check_itinerary(requirement(days=1), _results(), [day]) == []


def _plan(names, **windows):
    from app.agents.itinerary_planner import PlaceSelection, SelectedPlace, build_itinerary

    results = _results()
    # 这几条只测已选景点怎么排；候选里只留选中的，避免空时段被 top_up_days 补上。
    results[0].options = [option for option in results[0].options if option.name in names]
    for option in results[0].options:
        if option.name in windows:
            option.attributes["opening_hours"] = windows[option.name]
    selection = PlaceSelection(hotel_id="市中心酒店", attractions=[SelectedPlace(id=name) for name in names])
    return build_itinerary(requirement(days=len(names)), results, selection, TravelCosts(results))


def test_far_apart_places_are_not_forced_into_the_same_day():
    itinerary = _plan(["武侯祠", "都江堰"])

    assert [slot.title for slot in itinerary[0].slots[:2]] == ["武侯祠", "自由活动"]
    assert [slot.title for slot in itinerary[1].slots[:2]] == ["都江堰", "自由活动"]


def test_late_opening_place_is_moved_to_the_afternoon():
    swapped = _plan(["武侯祠", "人民公园"], 武侯祠="12:00-22:00")
    assert {slot.period: slot.title for slot in swapped[0].slots[:2]} == {"morning": "人民公园", "afternoon": "武侯祠"}

    alone = _plan(["武侯祠"], 武侯祠="12:00-22:00")
    assert [(slot.period, slot.title) for slot in alone[0].slots[:2]] == [("morning", "自由活动"), ("afternoon", "武侯祠")]


def test_dinner_shortlist_leaves_out_luxury_restaurants_unless_asked(tmp_path):
    rows = ["city,rank,name,type,latitude,longitude,open_time,close_time,price,recommended_food"]
    rows += [f"成都,{i},酒店中餐厅{i},粤菜,30.65,104.07,11:30,21:30,{900 + i},烧鹅" for i in range(3)]
    rows += [f"成都,{i + 3},家常馆{i},川菜,30.65,104.07,11:00,21:30,{60 + i * 10},回锅肉" for i in range(8)]
    (tmp_path / "restaurants.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    catalog = PlaceCatalog(tmp_path)

    everyday = [record.name for record in catalog.shortlist(requirement(), "food")]
    upscale = [record.name for record in catalog.shortlist(requirement(styles=["米其林"]), "food")]

    assert not any(name.startswith("酒店中餐厅") for name in everyday)
    assert upscale[0] == "酒店中餐厅0"
