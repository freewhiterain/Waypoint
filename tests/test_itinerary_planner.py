"""LLM 选点 → 算法排程 → 约束检查（LLM 全部用假对象，不访问网络）。"""

import re
from datetime import date

import pytest

from app.agents.itinerary_planner import (
    PlaceSelection,
    SelectedPlace,
    TravelCosts,
    assign_days,
    check_itinerary,
    plan_itinerary,
    schedule_by_rules,
    top_up_days,
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
        for day in schedule_by_rules(requirement(days=1), results)[0]
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


def weather_result(*days):
    from app.schemas.planning import Evidence

    evidence = [
        Evidence(id=f"w{index}", content=f"成都 {day}：{sky} 15~20℃", source="高德开放平台天气服务",
                 metadata={"provider": "amap"})
        for index, (day, sky) in enumerate(days)
    ]
    # 总结为空：模拟子 Agent 的总结被证据校验丢掉的常见情况。
    return WorkerResult(task_id="weather-t", worker="weather", status="partial", summary="", evidence=evidence)


@pytest.mark.asyncio
async def test_weather_evidence_reaches_the_selection_prompt_and_the_day_notes():
    llm = FakeLLM([PlaceSelection(hotel_id="武侯酒店", attractions=[SelectedPlace(id="武侯祠")])])
    results = [*chengdu_results(LEGS), weather_result(("2026-11-13", "晴"), ("2026-11-14", "小雨"))]

    itinerary, _warnings = await plan_itinerary(requirement(days=2), results, llm=llm)

    # 只带上行程覆盖到的日期；出发前一天的预报不出现。
    assert '"weather": ["第1天：小雨 15~20℃"]' in llm.prompts[0]
    assert itinerary[0].notes[0] == "天气：小雨 15~20℃"
    assert not any(note.startswith("天气：") for note in itinerary[1].notes)


def test_api_evidence_sharing_one_doc_url_is_not_collapsed_into_one():
    from app.governance.evidence import EvidenceGovernanceService
    from app.schemas.planning import Evidence

    url = "https://lbs.amap.com/api/webservice/guide/api/weatherinfo"
    days = [Evidence(content=f"成都 2026-11-1{n}：晴", source="高德", source_url=url, metadata={"provider": "amap"}) for n in (4, 5)]
    pages = [Evidence(content=text, source="某网站", source_url="https://example.com/a") for text in ("上半段", "下半段")]

    assert len({EvidenceGovernanceService._dedupe_key(item) for item in days}) == 2
    assert len({EvidenceGovernanceService._dedupe_key(item) for item in pages}) == 1


@pytest.mark.asyncio
async def test_empty_llm_selection_is_retried_before_falling_back_to_rules():
    # 强制工具调用下 DeepSeek 偶尔返回空结果，langchain 解析成 None。
    llm = FakeLLM([None, PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, warnings = await plan_itinerary(requirement(days=1), chengdu_results(LEGS), llm=llm)

    assert [slot.title for slot in itinerary[0].slots] == ["武侯祠", "锦里", "火锅"]
    assert "itinerary_selection_fallback:llm_failed" not in warnings


def test_selected_place_blocked_as_too_far_on_the_last_day_is_not_dropped():
    # 熊猫基地离其余景点都超过 MAX_HOP_MINUTES：贪心会让它独占一天，
    # 剩下的景点不能因为"最后一天没位置"被丢掉。
    legs = [
        leg("武侯酒店", "武侯祠", 5), leg("武侯酒店", "锦里", 10), leg("武侯祠", "锦里", 8),
        leg("武侯酒店", "熊猫基地", 50), leg("熊猫基地", "武侯祠", 90), leg("熊猫基地", "锦里", 90),
        leg("武侯酒店", "宽窄巷子", 25), leg("宽窄巷子", "武侯祠", 20), leg("宽窄巷子", "锦里", 22),
        leg("熊猫基地", "宽窄巷子", 80),
    ]
    costs = TravelCosts(chengdu_results(legs))

    days = assign_days(["武侯祠", "锦里", "熊猫基地", "宽窄巷子"], {}, 2, "武侯酒店", costs)

    assert sorted(place for day in days for place in day) == ["宽窄巷子", "武侯祠", "熊猫基地", "锦里"]


@pytest.mark.asyncio
async def test_half_day_left_empty_by_the_llm_is_filled_with_a_nearby_candidate():
    # 模型只选了 3 个景点，两天还空一个时段；武侯祠旁边的锦里没被选中，应补进来。
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="宽窄巷子"), SelectedPlace(id="熊猫基地")],
        dinners=[SelectedPlace(id="火锅"), SelectedPlace(id="小吃")],
    )])

    itinerary, _ = await plan_itinerary(requirement(days=2), chengdu_results(LEGS), llm=llm)

    titles = [slot.title for day in itinerary for slot in day.slots if slot.period != "evening"]
    assert sorted(titles) == ["宽窄巷子", "武侯祠", "熊猫基地", "锦里"]
    assert "自由活动" not in titles


def test_top_up_leaves_the_slot_empty_when_nothing_is_close_enough():
    costs = TravelCosts(chengdu_results([leg("武侯酒店", "熊猫基地", 50), leg("熊猫基地", "锦里", 120)]))

    days = top_up_days([["熊猫基地"]], ["熊猫基地", "锦里"], "武侯酒店", costs)

    assert days == [["熊猫基地"]]


def _lunch_results():
    results = chengdu_results(LEGS + [leg("武侯祠", "小吃", 6), leg("锦里", "小吃", 7)])
    results[1].options[1].attributes["opening_hours"] = "10:00-21:00"
    return results


@pytest.mark.asyncio
async def test_food_lovers_get_a_lunch_near_the_morning_stop():
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, _ = await plan_itinerary(requirement(days=1, styles=["美食"]), _lunch_results(), llm=llm)

    assert [(slot.period, slot.title) for slot in itinerary[0].slots] == [
        ("morning", "武侯祠"), ("lunch", "小吃"), ("afternoon", "锦里"), ("evening", "火锅"),
    ]
    assert itinerary[0].slots[1].travel_minutes == 6
    assert itinerary[0].slots[2].travel_minutes == 7


@pytest.mark.asyncio
async def test_no_lunch_slot_without_food_interest():
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, _ = await plan_itinerary(requirement(days=1), _lunch_results(), llm=llm)

    assert [slot.period for slot in itinerary[0].slots] == ["morning", "afternoon", "evening"]


def test_not_eating_spicy_food_is_not_read_as_food_interest():
    from app.agents.itinerary_planner import wants_lunch

    assert wants_lunch(requirement(styles=["美食"]))
    assert not wants_lunch(requirement(food_preferences=["不吃辣"]))


@pytest.mark.asyncio
async def test_origin_city_is_mentioned_on_the_first_and_last_day_without_invented_schedules():
    itinerary, _ = await plan_itinerary(requirement(days=2, origin="上海"), chengdu_results(LEGS), llm=None)

    assert any(note.startswith("交通：从上海出发") for note in itinerary[0].notes)
    assert any("返回上海" in note for note in itinerary[-1].notes)
    transport = " ".join(note for day in itinerary for note in day.notes if note.startswith("交通："))
    assert not re.search(r"\d{1,2}:\d{2}|[GDKZT]\d{2,4}", transport)  # 不编车次和发车时间


@pytest.mark.asyncio
async def test_the_same_restaurant_is_not_repeated_on_later_evenings():
    results = chengdu_results(LEGS)
    results[1].options = results[1].options[:1]  # 只有一家餐厅
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里"), SelectedPlace(id="宽窄巷子")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, _ = await plan_itinerary(requirement(days=2), results, llm=llm)

    evenings = [day.slots[-1] for day in itinerary]
    assert evenings[0].title == "火锅"
    assert evenings[1].title == "自由安排晚餐" and "已经安排过" in evenings[1].description


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("灵隐寺", "灵隐飞来峰景区", True),
        ("故宫", "故宫博物院", True),
        ("南京博物院", "南京总统府", False),  # 主干只剩城市名时不算同一处
        ("成都博物馆", "成都大熊猫繁育研究基地", False),
        ("武侯祠", "锦里", False),
        ("楼外楼(西湖景区孤山路店)", "西湖风景名胜区", False),  # 括号里的分店名不算
        ("四季民福烤鸭店(故宫店)", "故宫博物院", False),
        ("南京大牌档(夫子庙平江府店)", "夫子庙", False),
    ],
)
def test_same_place_recognises_different_names_for_one_spot(first, second, expected):
    from app.agents.itinerary_planner import same_place

    city = "南京" if first.startswith("南京") else "成都" if first.startswith("成都") else "杭州"
    assert same_place(first, second, city) is expected


@pytest.mark.asyncio
async def test_two_names_for_one_spot_are_not_both_scheduled():
    results = chengdu_results(LEGS)
    results[0].options.append(option("武侯祠博物馆", "attractions"))
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="武侯祠博物馆"), SelectedPlace(id="锦里")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, _ = await plan_itinerary(requirement(days=2), results, llm=llm)

    titles = [slot.title for day in itinerary for slot in day.slots]
    assert "武侯祠" in titles and "武侯祠博物馆" not in titles


def test_a_relaxed_pace_style_makes_every_day_light():
    from app.agents.itinerary_planner import day_capacities

    assert day_capacities(requirement(days=3, styles=["适合老人", "节奏轻松"])) == [1, 1, 1]
    assert day_capacities(requirement(days=2, styles=["美食"])) == [2, 2]


def _night_results():
    results = chengdu_results(LEGS + [leg("火锅", "锦江夜游", 10), leg("武侯祠", "锦江夜游", 15)])
    results[0].options.append(option("锦江夜游", "attractions"))
    return results


@pytest.mark.asyncio
async def test_night_spots_go_after_dinner_for_night_view_lovers():
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店",
        attractions=[SelectedPlace(id="锦江夜游"), SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, _ = await plan_itinerary(requirement(days=1, styles=["夜景"]), _night_results(), llm=llm)

    periods = [(slot.period, slot.title) for slot in itinerary[0].slots]
    assert periods[-1] == ("night", "锦江夜游")
    assert ("morning", "锦江夜游") not in periods and ("afternoon", "锦江夜游") not in periods


@pytest.mark.asyncio
async def test_night_spots_are_never_scheduled_in_daytime():
    itinerary, _ = await plan_itinerary(requirement(days=2), _night_results(), llm=None)

    titles = [slot.title for day in itinerary for slot in day.slots]
    assert "锦江夜游" not in titles


def test_a_specific_dish_in_interests_counts_as_food_interest():
    from app.agents.itinerary_planner import wants_lunch

    assert wants_lunch(requirement(styles=["火锅", "夜景"]))
    assert not wants_lunch(requirement(styles=["夜景"], food_preferences=["不吃辣"]))
    # 模型把想吃的东西记进口味时同样算；忌口不算
    assert wants_lunch(requirement(styles=["夜景"], food_preferences=["火锅"]))
    assert not wants_lunch(requirement(food_preferences=["不要火锅", "少吃烧烤"]))


@pytest.mark.asyncio
async def test_two_names_for_the_same_night_activity_are_not_both_scheduled():
    results = _night_results()
    # 真实数据里是"重庆夜游"和"夜游两江"：去掉城市名后主干都是"夜游"。
    results[0].options.append(option("成都夜游", "attractions"))
    results[3].options.append(leg("小吃", "成都夜游", 5))

    itinerary, _ = await plan_itinerary(requirement(days=2, styles=["夜景"]), results, llm=None)

    nights = [slot.title for day in itinerary for slot in day.slots if slot.period == "night"]
    assert len(nights) == 1


@pytest.mark.asyncio
async def test_a_restaurant_that_closes_in_the_afternoon_is_never_dinner():
    results = chengdu_results(LEGS)
    food = next(item for item in results if item.worker == "food")
    food.options[0] = option("火锅", "food", opening_hours="11:30-16:30")
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店", attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里")],
        dinners=[SelectedPlace(id="火锅")],
    )])

    itinerary, _ = await plan_itinerary(requirement(days=1), results, llm=llm)

    assert itinerary[0].slots[-1].title == "小吃"


@pytest.mark.asyncio
async def test_two_branches_of_one_chain_are_not_both_scheduled():
    results = chengdu_results(LEGS)
    food = next(item for item in results if item.worker == "food")
    food.options[:] = [
        CandidateOption(id="a", name="盛天毛血旺(来福士店)", category="food", description="d", evidence_ids=["e"]),
        CandidateOption(id="b", name="盛天毛血旺(国泰广场店)", category="food", description="d", evidence_ids=["e"]),
        CandidateOption(id="c", name="杨记隆府", category="food", description="d", evidence_ids=["e"]),
    ]
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店", attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里")],
        dinners=[SelectedPlace(id="a"), SelectedPlace(id="b")],
    )])

    itinerary, _ = await plan_itinerary(requirement(days=2), results, llm=llm)

    dinners = [day.slots[-1].title for day in itinerary]
    assert dinners[0].startswith("盛天毛血旺") and dinners[1] == "杨记隆府"


@pytest.mark.asyncio
async def test_a_far_away_dinner_is_swapped_for_a_nearby_one():
    # 当天最后一站是锦里：模型选的"远店"要 90 分钟，备选"火锅"只要 12 分钟
    results = chengdu_results(LEGS + [leg("锦里", "远店", 90), leg("锦里", "火锅", 12)])
    food = next(item for item in results if item.worker == "food")
    food.options.append(option("远店", "food"))
    llm = FakeLLM([PlaceSelection(
        hotel_id="武侯酒店", attractions=[SelectedPlace(id="武侯祠"), SelectedPlace(id="锦里")],
        dinners=[SelectedPlace(id="远店")],
    )])

    itinerary, warnings = await plan_itinerary(requirement(days=1), results, llm=llm)

    assert itinerary[0].slots[-1].title == "火锅"
    assert not any("dinner_far" in warning for warning in warnings)


def test_a_far_dinner_with_no_nearby_backup_is_a_violation():
    from app.agents.itinerary_planner import check_itinerary
    from app.schemas.planning import ItineraryDay, TimeSlot

    day = ItineraryDay(day=1, date=date(2026, 11, 14), slots=[
        TimeSlot(period="evening", title="远店", travel_minutes=90)])

    violations = check_itinerary(requirement(days=1), chengdu_results(LEGS), [day])

    assert [item.code for item in violations] == ["dinner_far"]
