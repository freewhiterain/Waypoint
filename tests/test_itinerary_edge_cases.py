"""排程引擎的边界情况和性质测试找出来的回归用例（固定输入，不依赖随机生成）。"""

from __future__ import annotations

from datetime import date

from app.agents.itinerary.planner import schedule_by_rules
from app.agents.itinerary.scheduling import dodge_rain
from app.agents.itinerary.travel import TravelCosts
from app.schemas.planning import CandidateOption, Evidence, TravelRequirement, WorkerResult

CITY = "测试市"
START = date(2026, 11, 14)


def place(name, category, km_north=0.0, cost=None, **attributes) -> CandidateOption:
    """在 (30.6, 104.0) 正北 km_north 公里处放一个候选。"""
    attributes.setdefault("latitude", 30.6 + km_north / 111.0)
    attributes.setdefault("longitude", 104.0)
    return CandidateOption(id=name, name=name, category=category, description=f"{name}的介绍",
                           estimated_cost=cost, attributes=attributes, evidence_ids=["e"])


def result(worker, options) -> WorkerResult:
    return WorkerResult(task_id=f"{worker}-t", worker=worker, status="completed", summary="", options=options)


def requirement(**updates) -> TravelRequirement:
    return TravelRequirement(**{"destination": CITY, "departure_date": START, "days": 2, **updates})


def test_dinner_swapped_after_a_free_slot_keeps_its_leg_from_the_hotel():
    """没有景点的那天上午是"自由活动"：换成当地菜的晚饭要按从住处出发算路程，远的不换过去。"""
    results = [
        result("attractions", []),
        result("food", [place("川菜馆A", "food", 0.5, type="川菜"), place("川菜馆B", "food", 0.6, type="川菜"),
                        place("小吃店", "food", 2.0, type="小吃"), place("远郊小吃", "food", 60.0, type="小吃")]),
        result("hotel", [place("酒店", "hotel", 0.0)]),
    ]
    costs = TravelCosts(results)

    itinerary, _ = schedule_by_rules(requirement(), results, costs)

    dinners = [next(slot for slot in day.slots if slot.period == "evening") for day in itinerary]
    assert [slot.title for slot in dinners] == ["川菜馆A", "小吃店"]
    assert (dinners[1].travel_minutes, dinners[1].travel_mode) == costs.leg("酒店", "小吃店")
    assert dinners[1].travel_minutes is not None


def test_neighbouring_excursions_are_not_doubled_up_on_a_relaxed_day():
    """挨着的两处远郊平时可以同一天；整趟要求轻松时每天只排一个，各占一天。"""
    results = [
        result("attractions", [place("远郊甲", "attractions", 60.0), place("远郊乙", "attractions", 60.5),
                               place("市区景点", "attractions", 1.0)]),
        result("food", [place("饭馆", "food", 0.5)]),
        result("hotel", [place("酒店", "hotel", 0.0)]),
    ]

    relaxed, _ = schedule_by_rules(requirement(days=3, styles=["轻松"]), results)
    normal, _ = schedule_by_rules(requirement(days=3), results)

    def sights(itinerary):
        return [[slot.title for slot in day.slots if slot.period in ("morning", "afternoon")
                 and not slot.title.startswith("自由")] for day in itinerary]

    assert all(len(day) <= 1 for day in sights(relaxed)), sights(relaxed)
    assert {"远郊甲", "远郊乙"} <= {title for day in sights(relaxed) for title in day}
    assert {"远郊甲", "远郊乙"} in [set(day) for day in sights(normal)]


def weather(*chances) -> WorkerResult:
    evidence = [Evidence(content=f"{CITY} 2026-11-{14 + index}：小雨；降水概率 {chance}%", source="test")
                for index, chance in enumerate(chances)]
    return WorkerResult(task_id="weather-t", worker="weather", status="completed", summary="", evidence=evidence)


def test_rain_swap_does_not_move_a_theme_park_onto_day_one():
    """乐园排在中间那天、那天下大雨：可以和第三天对调，不能换到第一天（第一天要赶路）。"""
    options = {name: place(name, "attractions", type=kind)
               for name, kind in (("欢乐谷", "游乐园/体育娱乐"), ("博物馆甲", "博物馆"), ("博物馆乙", "博物馆"))}
    forecast = {1: "降水概率 10%", 2: "降水概率 90%", 3: "降水概率 10%"}
    plans = [["博物馆甲"], ["欢乐谷"], ["博物馆乙"]]

    assert dodge_rain(plans, options, forecast, [1, 1, 1], {"欢乐谷"}) == [["博物馆甲"], ["博物馆乙"], ["欢乐谷"]]
    # 只有第一天更干时宁可不换
    assert dodge_rain(plans, options, {1: "降水概率 10%", 2: "降水概率 90%"}, [1, 1, 1], {"欢乐谷"}) == plans
    # 不传 whole_day 时行为不变
    assert dodge_rain(plans, options, forecast, [1, 1, 1])[0] == ["欢乐谷"]


def test_relaxed_trip_with_rain_keeps_the_theme_park_off_day_one():
    results = [
        result("attractions", [place("欢乐谷", "attractions", 10.0, type="游乐园/体育娱乐"),
                               place("博物馆甲", "attractions", 1.0, type="博物馆"),
                               place("博物馆乙", "attractions", 1.5, type="博物馆")]),
        result("food", [place("饭馆", "food", 0.5)]),
        result("hotel", [place("酒店", "hotel", 0.0)]),
        weather(10, 90, 10),
    ]

    itinerary, _ = schedule_by_rules(requirement(days=3, styles=["轻松"]), results)

    assert "欢乐谷" not in [slot.title for slot in itinerary[0].slots]
    assert "欢乐谷" in [slot.title for day in itinerary for slot in day.slots]
