"""排程引擎的性质测试：随机生成候选池，检查无论候选怎么组合都该成立的不变量。

候选池里有市区景点、主题乐园、远郊景点、夜景、餐厅、酒店，坐标、价格、营业时间随机；
天数 1–7、预算、必去、轻松节奏、雨天也随机。没有交通子 Agent 的实测路线，
TravelCosts 只能按坐标直线距离估算，正好是"假的交通代价"。
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta

from hypothesis import HealthCheck, given, settings, strategies as st

from app.agents.itinerary.planner import plan_itinerary, schedule_by_rules
from app.agents.itinerary.repair import trim_to_budget
from app.agents.itinerary.rules import (
    HOTEL_NOTE_PREFIX,
    PlaceSelection,
    SelectedPlace,
    _DINNER_WINDOW,
    _LUNCH_WINDOW,
    _fits,
    _opening_minutes,
    day_capacities,
    is_excursion,
    is_full_day,
    outskirts,
)
from app.agents.itinerary.travel import WALK_KM, TravelCosts
from app.agents.trip_cost import estimate_trip_cost
from app.evaluation.plan_checks import check_plan
from app.places.catalog import named_by, straight_line_km
from app.schemas.planning import CandidateOption, Evidence, ItineraryDay, TravelRequirement, WorkerResult

CITY = "测试市"
START = date(2026, 11, 14)
CENTER = (30.60, 104.00)
_KM_PER_DEGREE = 111.0

SIGHT_HOURS = (None, "08:00-18:00", "09:00-17:00", "08:30-21:00", "13:00-21:00", "全天")
FOOD_HOURS = (None, "10:00-22:00", "11:00-14:00", "17:00-23:00", "全天")
SIGHT_TYPES = ("公园", "博物馆", "历史古迹", "自然风光")
FOOD_TYPES = ("小吃", "川菜", "西餐", "火锅")
PLACEHOLDER_PREFIXES = ("自由", f"{CITY}分区")


class EmptyCatalog:
    """库外城市：plan_checks 不做地点库对照，只检查逻辑层和描述一致性。"""

    def cities(self):
        return []

    def find(self, city, name):
        return None


@dataclass
class Pool:
    sights: list[CandidateOption] = field(default_factory=list)
    parks: list[CandidateOption] = field(default_factory=list)
    excursions: list[CandidateOption] = field(default_factory=list)
    nights: list[CandidateOption] = field(default_factory=list)
    food: list[CandidateOption] = field(default_factory=list)
    hotels: list[CandidateOption] = field(default_factory=list)
    rain: dict[int, int] = field(default_factory=dict)

    def results(self) -> list[WorkerResult]:
        evidence = [
            Evidence(content=f"{CITY} {(START + timedelta(days=day - 1)).isoformat()}：小雨 12~18℃；降水概率 {chance}%",
                     source="test")
            for day, chance in sorted(self.rain.items())
        ]
        return [
            _result("attractions", [*self.sights, *self.parks, *self.excursions, *self.nights]),
            _result("food", self.food),
            _result("hotel", self.hotels),
            WorkerResult(task_id="weather-t", worker="weather", status="completed", summary="", evidence=evidence),
        ]


def _result(worker: str, options: list[CandidateOption]) -> WorkerResult:
    return WorkerResult(task_id=f"{worker}-t", worker=worker, status="completed", summary="", options=options)


def _at(distance_km: float, bearing: float) -> dict[str, float]:
    """离中心 distance_km 公里、方位角 bearing（弧度）的坐标；经度按纬度修正，距离才准。"""
    north, east = distance_km * math.cos(bearing), distance_km * math.sin(bearing)
    return {"latitude": CENTER[0] + north / _KM_PER_DEGREE,
            "longitude": CENTER[1] + east / (_KM_PER_DEGREE * math.cos(math.radians(CENTER[0])))}


_bearing = st.floats(0, 2 * math.pi)
_price = st.one_of(st.none(), st.sampled_from([0.0, 20.0, 50.0, 120.0, 300.0]))


@st.composite
def _place(draw, name: str, category: str, min_km: float, max_km: float, hours, types, located=True):
    attributes: dict = {}
    if located:
        attributes.update(_at(draw(st.floats(min_km, max_km)), draw(_bearing)))
    opening = draw(st.sampled_from(hours))
    if opening:
        attributes["opening_hours"] = opening
    attributes["type"] = draw(st.sampled_from(types))
    return CandidateOption(id=name, name=name, category=category, description=f"{name}的介绍",
                           estimated_cost=draw(_price), attributes=attributes, evidence_ids=["e"])


@st.composite
def pools(draw, min_sights=0, max_sights=14, max_food=10, max_hotels=3, max_days=7):
    # 名字等宽编号：same_place 按子串比，"景点1" 会被当成 "景点11" 的一部分。
    pool = Pool()
    for i in range(draw(st.integers(min_sights, max_sights))):
        pool.sights.append(draw(_place(f"景点{i:02d}", "attractions", 0.2, 8.0, SIGHT_HOURS, SIGHT_TYPES,
                                       located=draw(st.integers(0, 9)) > 0)))  # 一成没有坐标（联网搜来的）
    for i in range(draw(st.integers(0, 2))):
        pool.parks.append(draw(_place(f"欢乐谷{i:02d}", "attractions", 5.0, 20.0, SIGHT_HOURS, ("游乐园/体育娱乐",))))
    for i in range(draw(st.integers(0, 3))):
        pool.excursions.append(draw(_place(f"远郊{i:02d}", "attractions", 45.0, 80.0, SIGHT_HOURS, SIGHT_TYPES)))
    for i in range(draw(st.integers(0, 2))):
        pool.nights.append(draw(_place(f"夜景{i:02d}", "attractions", 0.2, 6.0, (None, "18:00-23:00"), ("人文景观",))))
    for i in range(draw(st.integers(0, max_food))):
        pool.food.append(draw(_place(f"餐厅{i:02d}", "food", 0.1, 8.0, FOOD_HOURS, FOOD_TYPES)))
    for i in range(draw(st.integers(0, max_hotels))):
        pool.hotels.append(draw(_place(f"酒店{i:02d}", "hotel", 0.1, 5.0, (None,), ("经济型",))))
    days = draw(st.integers(1, max_days))
    rainy = draw(st.lists(st.integers(1, days), unique=True, max_size=days))
    pool.rain = {day: draw(st.sampled_from([10, 40, 80, 95])) for day in rainy}
    return pool, days


@st.composite
def scenarios(draw, **pool_kwargs):
    pool, days = draw(pools(**pool_kwargs))
    nameable = [option.name for option in [*pool.sights, *pool.parks, *pool.excursions, *pool.food]]
    must_visit = draw(st.lists(st.sampled_from(nameable), unique=True, max_size=4)) if nameable else []
    pace = draw(st.sampled_from(["normal", "relaxed_trip", "relaxed_days"]))
    styles = draw(st.lists(st.sampled_from(["美食", "夜景", "历史"]), unique=True, max_size=3))
    if pace == "relaxed_trip":
        styles.append("轻松")
    relaxed_days = draw(st.lists(st.integers(1, days), unique=True, max_size=days)) if pace == "relaxed_days" else []
    requirement = TravelRequirement(
        destination=CITY, departure_date=START, days=days,
        adults=draw(st.integers(1, 3)), children=draw(st.integers(0, 2)),
        styles=styles, must_visit=must_visit, relaxed_days=relaxed_days,
        budget=draw(st.one_of(st.none(), st.sampled_from([50.0, 300.0, 1000.0, 3000.0, 20000.0]))),
    )
    return pool, requirement


# ---------- 帮助函数 ----------

def _ids(pool: Pool) -> dict[str, CandidateOption]:
    return {option.name: option for option in [*pool.sights, *pool.parks, *pool.excursions, *pool.nights,
                                                *pool.food, *pool.hotels]}


def _placeholder(title: str) -> bool:
    return title.startswith(PLACEHOLDER_PREFIXES)


def _sights(day: ItineraryDay) -> list[str]:
    return [slot.title for slot in day.slots if slot.period in ("morning", "afternoon") and not _placeholder(slot.title)]


def _hotel(pool: Pool, itinerary: list[ItineraryDay], planner: bool) -> CandidateOption | None:
    """plan_itinerary 在第一天备注里写住哪家（LLM 选点时 closer_hotel 可能换过），没写就是没住处；
    schedule_by_rules 不写备注，规则选点固定用第一家（见 select_places_by_rules）。"""
    if not planner:
        return pool.hotels[0] if pool.hotels else None
    note = next((note for note in (itinerary[0].notes if itinerary else []) if note.startswith(HOTEL_NOTE_PREFIX)), None)
    if note is None:
        assert not pool.hotels, "有住宿候选却没安排住处"
        return None
    name = note.removeprefix(HOTEL_NOTE_PREFIX).split("。")[0]
    return next(option for option in pool.hotels if option.name == name)


def _whole(pool: Pool, results: list[WorkerResult]):
    remote = outskirts(results)
    return lambda option: is_full_day(option) or is_excursion(option, remote)


def _check_common(pool: Pool, requirement: TravelRequirement, itinerary: list[ItineraryDay], planner: bool) -> None:
    results = pool.results()
    by_name = _ids(pool)
    costs = TravelCosts(results)
    whole = _whole(pool, results)
    hotel = _hotel(pool, itinerary, planner)

    assert len(itinerary) == requirement.days
    assert [day.day for day in itinerary] == list(range(1, requirement.days + 1))

    # 每个标题要么是候选里的地点，要么是兜底文案，不能编造
    for day in itinerary:
        for slot in day.slots:
            assert slot.title in by_name or _placeholder(slot.title), slot.title

    # 同一处地方不去两次：景点、餐、夜景各自不重复
    for periods in (("morning", "afternoon"), ("lunch", "evening"), ("night",)):
        titles = [slot.title for day in itinerary for slot in day.slots
                  if slot.period in periods and not _placeholder(slot.title)]
        assert len(titles) == len(set(titles)), titles

    # 每天的时段顺序固定，晚饭每天恰好一顿
    order = ["morning", "lunch", "afternoon", "evening", "night"]
    for day in itinerary:
        periods = [slot.period for slot in day.slots]
        assert periods == sorted(periods, key=order.index), periods
        assert periods.count("evening") == 1
        assert periods.count("morning") <= 1 and periods.count("afternoon") <= 1

    capacities = day_capacities(requirement)
    for day, capacity in zip(itinerary, capacities):
        sights = _sights(day)
        # 轻松的天只排一个景点；普通的天不超过两个
        assert len(sights) <= capacity, (day.day, sights, capacity)
        # 乐园、远郊占一整天：当天不再有市区景点（同一片的远郊可以同一天）
        if any(whole(by_name[title]) for title in sights):
            assert all(whole(by_name[title]) for title in sights), sights
        if any(is_full_day(by_name[title]) for title in sights):
            assert len(sights) == 1, sights

    # 第一天要赶路：后面还有整天空着的日子时，乐园和远郊不排第一天。
    # （后面的日子都排了东西才算没办法——比如模型把唯一的市区景点指定在第二天，补位的乐园只能去第一天。）
    if requirement.days > 1 and any(whole(by_name[title]) for title in _sights(itinerary[0])):
        assert all(_sights(day) for day in itinerary[1:]), [_sights(day) for day in itinerary]

    # 时段和营业时间对得上
    for day in itinerary:
        for slot in day.slots:
            if _placeholder(slot.title):
                continue
            option = by_name[slot.title]
            # 例外：两个必去都下午才开门、只有一天可排时，两条硬约束没法同时满足。
            # 排程选择不丢必去，由 check_itinerary 报 opening_hours 告诉用户（planner 测试里核对有报）。
            if slot.period in ("morning", "afternoon") and not named_by(slot.title, requirement.must_visit):
                assert _fits(option, slot.period), (slot.period, slot.title, slot.opening_window)
            window = _opening_minutes(slot.opening_window) if slot.opening_window != "全天" else None
            if slot.period == "lunch" and window:
                assert window[0] <= _LUNCH_WINDOW[0] and window[1] >= _LUNCH_WINDOW[1], slot
            if slot.period == "evening" and window:
                assert window[0] <= _DINNER_WINDOW[0] and window[1] >= _DINNER_WINDOW[1], slot

    # 每段路的耗时和方式与真实的上一站一致；步行只在 1.2 公里以内
    for day in itinerary:
        previous = hotel.id if hotel else None
        for slot in day.slots:
            if _placeholder(slot.title):
                continue
            current = by_name[slot.title].id
            assert (slot.travel_minutes, slot.travel_mode) == costs.leg(previous, current), \
                (day.day, slot.period, previous, current, slot.travel_minutes, slot.travel_mode)
            if slot.travel_mode == "walk":
                first, second = by_name[previous].attributes, by_name[current].attributes
                km = straight_line_km((first["latitude"], first["longitude"]), (second["latitude"], second["longitude"]))
                assert km <= WALK_KM
            previous = current


def _expected_sight_days(pool: Pool, requirement: TravelRequirement) -> bool:
    """市区普通景点够填满每天所有时段时，每天都应该有景点。

    只够"每天一个"时不算：贪心分天先把前几天排满，最后几天可能整天自由活动
    （tests/test_scheduling.py 明确要求两天两个景点时都排第一天、第二天写"已无未使用的景点候选"），
    而 plan_checks 会把空着的那天记成逻辑违规。两边的取舍要由产品决定，这里不当作排程的错。
    """
    city_sights = [option for option in pool.sights if not _avoided_name(option.name, requirement)]
    return len(city_sights) >= sum(day_capacities(requirement))


def _avoided_name(name: str, requirement: TravelRequirement) -> bool:
    return any(term and (term in name or name in term) for term in requirement.avoid_places)


# ---------- 性质 ----------

# CI 里每个性质固定跑同一批 200 例（derandomize），结果可复现、不会某天突然红；
# 本地想多找找时 ITINERARY_EXAMPLES=5000 pytest tests/test_itinerary_properties.py，每次随机。
_EXPLORE = os.environ.get("ITINERARY_EXAMPLES")
SETTINGS = settings(max_examples=int(_EXPLORE or 200), derandomize=_EXPLORE is None, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(scenarios())
def test_rule_schedule_invariants(scenario):
    pool, requirement = scenario
    itinerary, _warnings = schedule_by_rules(requirement, pool.results())

    _check_common(pool, requirement, itinerary, False)
    if _expected_sight_days(pool, requirement):
        assert all(_sights(day) for day in itinerary), [_sights(day) for day in itinerary]


@SETTINGS
@given(scenarios())
def test_planner_invariants_and_plan_checks(scenario):
    pool, requirement = scenario
    results = pool.results()
    itinerary, warnings = asyncio.run(plan_itinerary(requirement, results))
    hotel = _hotel(pool, itinerary, True)

    _check_common(pool, requirement, itinerary, True)

    issues = [str(issue) for issue in check_plan(requirement, itinerary, EmptyCatalog(),
                                                  hotel.estimated_cost if hotel else None)]
    # 候选本身不够时有些违规躲不开：点名的地方容量放不下、预算本来就不够、附近只有一家很远的饭店
    # （去远郊玩一天回城吃晚饭，check_itinerary 也认为正常）。这些只要 planner 自己如实报了就算过；
    # 没报就是悄悄交了一份不合格的行程。
    reported = {item.split(":")[1] for item in warnings if item.startswith("itinerary_constraint:")}
    tolerated = {"用户点名要去": "must_visit", "超出预算": "budget", "吃饭路上": "dinner_far"}
    must_hours = [issue for issue in issues if "时段对不上" in issue and ("morning" in issue or "afternoon" in issue)
                  and named_by(issue.split("「")[1].split("」")[0], requirement.must_visit)]
    if must_hours:
        assert "opening_hours" in reported, (must_hours, warnings)
    issues = [issue for issue in issues if issue not in must_hours]
    for issue in issues:
        code = next((code for word, code in tolerated.items() if word in issue), None)
        if code is not None:
            assert code in reported, (issue, warnings)
    # 景点不够排满每天时的空白天：见 _expected_sight_days 的说明
    allowed = [] if _expected_sight_days(pool, requirement) else ["整天没有安排任何景点"]
    hard = [issue for issue in issues if not any(word in issue for word in [*tolerated, *allowed])]
    assert hard == [], hard


@SETTINGS
@given(scenarios())
def test_trim_to_budget_never_raises_cost(scenario):
    pool, requirement = scenario
    if not requirement.budget:
        return
    results = pool.results()
    costs = TravelCosts(results)
    itinerary, _ = schedule_by_rules(requirement, results, costs)
    hotel = _hotel(pool, itinerary, False)
    price = hotel.estimated_cost if hotel else None
    before = estimate_trip_cost(requirement, itinerary, price).total

    trimmed = trim_to_budget(requirement, results, itinerary, hotel, costs)
    after = estimate_trip_cost(requirement, trimmed, price).total

    assert after <= before
    if before <= requirement.budget:
        assert trimmed == itinerary
    # 点名要去的地点不会被换掉
    kept = [slot.title for day in itinerary for slot in day.slots if named_by(slot.title, requirement.must_visit)]
    titles = [slot.title for day in trimmed for slot in day.slots]
    assert all(title in titles for title in kept)
    _check_common(pool, requirement, trimmed, False)


@SETTINGS
@given(scenarios(min_sights=4, max_food=10))
def test_must_visit_within_capacity_is_kept(scenario):
    """点名的普通市区景点不超过每天容量的总和时，一个都不能漏。"""
    pool, requirement = scenario
    sights = {option.name for option in pool.sights}
    requirement = requirement.model_copy(update={
        "must_visit": [name for name in requirement.must_visit if name in sights][: sum(day_capacities(requirement))]
    })
    itinerary, _ = schedule_by_rules(requirement, pool.results())

    titles = [slot.title for day in itinerary for slot in day.slots]
    assert all(name in titles for name in requirement.must_visit), (requirement.must_visit, titles)


class FakeLLM:
    """按 id 随机选点的假模型：会选不存在的 id、重复的、越界的 preferred_day，也会直接出错。"""

    def __init__(self, selections: list[PlaceSelection | None]):
        self.selections = list(selections)

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        selection = self.selections.pop(0) if self.selections else None
        if selection is None:
            raise RuntimeError("模型没返回")
        return selection


@st.composite
def llm_selections(draw, pool: Pool, days: int):
    attraction_ids = [option.id for option in [*pool.sights, *pool.parks, *pool.excursions, *pool.nights]]
    food_ids = [option.id for option in pool.food]
    hotel_ids = [option.id for option in pool.hotels]

    def picks(ids):
        choices = st.sampled_from([*ids, "不存在的id"]) if ids else st.just("不存在的id")
        return st.lists(st.builds(SelectedPlace, id=choices,
                                  preferred_day=st.one_of(st.none(), st.integers(1, days + 2))), max_size=2 * days + 2)

    return PlaceSelection(
        hotel_id=draw(st.one_of(st.none(), st.sampled_from([*hotel_ids, "不存在的酒店"]))),
        attractions=draw(picks(attraction_ids)),
        dinners=draw(picks(food_ids)),
    )


@st.composite
def llm_scenarios(draw):
    pool, requirement = draw(scenarios())
    rounds = draw(st.lists(st.one_of(st.none(), llm_selections(pool, requirement.days)), min_size=1, max_size=6))
    return pool, requirement, rounds


@SETTINGS
@given(llm_scenarios())
def test_planner_with_random_llm_selections_keeps_invariants(scenario):
    """模型怎么乱选，清理、排程、兜底之后的行程都要守住同样的不变量。"""
    pool, requirement, rounds = scenario
    itinerary, warnings = asyncio.run(plan_itinerary(requirement, pool.results(), llm=FakeLLM(rounds)))

    _check_common(pool, requirement, itinerary, True)
    hotel = _hotel(pool, itinerary, True)
    issues = [str(issue) for issue in check_plan(requirement, itinerary, EmptyCatalog(),
                                                  hotel.estimated_cost if hotel else None)]
    # 模型选得少时空白天躲不开（top_up 只补顺路的）；这里只核对其余逻辑违规都报给了用户
    reported = {item.split(":")[1] for item in warnings if item.startswith("itinerary_constraint:")}
    for issue in issues:
        if "超出预算" in issue:
            assert "budget" in reported, (issue, warnings)
        if "用户点名要去" in issue:
            assert "must_visit" in reported, (issue, warnings)
        assert "已经去过" not in issue and "要求轻松" not in issue and "乐园" not in issue, issue
