"""行程编排：LLM 选点 → 算法排程 → 约束检查，不通过就带着违规原因让 LLM 重选。

分工参考 ItiNera（LLM 理解偏好、算法保证空间顺路）和 LLM-Modulo（外部检查器
把关、违规反馈给 LLM）：

1. 选点（LLM）：按偏好、天气、预算从已通过证据治理的候选里挑景点、晚餐和落脚酒店。
   只能引用候选 id，引用不存在的 id 直接丢弃；没有 LLM 或调用失败时按规则挑选。
2. 排程（算法）：以酒店为中心，按交通耗时把景点两两成组分到每天，再在每天内部
   选最省路的先后顺序；晚餐选离当天最后一站最近的餐厅。
3. 检查（代码）：开放时间、每天交通总耗时、预算。违规时把原因交回第 1 步，最多重试
   MAX_ROUNDS 次，仍不通过就采用最后一版并把违规写进 warnings。

交通耗时来自交通子 Agent 的 travel_leg 候选。没有真实分钟数时（只有区级粗估），
排程只用一个内部代价来比较远近，展示给用户的耗时保持为空，不编造数字。
"""

from __future__ import annotations

import json
import re
from itertools import permutations
from typing import Any

from pydantic import BaseModel, Field

from app.agents.scheduling import calculate_budget, schedule_itinerary
from app.schemas.planning import (
    CandidateOption,
    ItineraryDay,
    TimeSlot,
    TravelRequirement,
    WorkerResult,
)
from app.utils.logger import app_logger

MAX_ROUNDS = 3
ATTRACTIONS_PER_DAY = 2
MAX_DAILY_TRAVEL_MINUTES = 180
# 没有真实耗时时用于比较远近的内部代价（分钟量级），不会展示给用户。
_PROXIMITY_COST = {"same_district": 20, "unknown": 35, "different_district": 50}
_OPENING_PATTERN = re.compile(r"(\d{1,2}):(\d{2})\s*[-~至到]\s*(\d{1,2}):(\d{2})")
# 时段与开放时间的最低要求：上午 11 点前要开门，下午 15 点后才关门。
_PERIOD_WINDOWS = {"morning": (11 * 60, None), "afternoon": (None, 15 * 60)}


class SelectedPlace(BaseModel):
    id: str
    reason: str = ""
    preferred_day: int | None = Field(default=None, ge=1)


class PlaceSelection(BaseModel):
    """LLM 选点的结构化输出：只能引用候选 id。"""

    hotel_id: str | None = None
    attractions: list[SelectedPlace] = Field(default_factory=list)
    dinners: list[SelectedPlace] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Violation(BaseModel):
    code: str
    message: str


def _options(results: list[WorkerResult], worker: str) -> list[CandidateOption]:
    return [option for result in results if result.worker == worker for option in result.options]


def _summaries(results: list[WorkerResult], worker: str) -> list[str]:
    return [result.summary for result in results if result.worker == worker and result.summary]


class TravelCosts:
    """地点之间的交通代价：优先真实分钟数，其次按区粗估。"""

    def __init__(self, results: list[WorkerResult]):
        self._minutes: dict[frozenset[str], int] = {}
        self._estimate: dict[frozenset[str], int] = {}
        for leg in _options(results, "transport"):
            attributes = leg.attributes
            key = frozenset((str(attributes.get("from_id")), str(attributes.get("to_id"))))
            if isinstance(attributes.get("minutes"), int):
                self._minutes[key] = attributes["minutes"]
            self._estimate[key] = _PROXIMITY_COST.get(str(attributes.get("proximity")), _PROXIMITY_COST["unknown"])

    def minutes(self, first: str | None, second: str | None) -> int | None:
        if first is None or second is None or first == second:
            return None
        return self._minutes.get(frozenset((first, second)))

    def cost(self, first: str | None, second: str | None) -> int:
        if first is None or second is None or first == second:
            return 0
        key = frozenset((first, second))
        if key in self._minutes:
            return self._minutes[key]
        return self._estimate.get(key, _PROXIMITY_COST["unknown"])


# ---------------------------------------------------------------- 1. 选点


def _option_digest(option: CandidateOption) -> dict[str, Any]:
    digest = {"id": option.id, "name": option.name, "description": option.description[:160]}
    if option.estimated_cost is not None:
        digest["estimated_cost"] = option.estimated_cost
    for key in ("location", "district", "opening_hours"):
        if option.attributes.get(key):
            digest[key] = option.attributes[key]
    return digest


async def select_places_with_llm(
    llm: Any,
    requirement: TravelRequirement,
    results: list[WorkerResult],
    feedback: list[Violation],
) -> PlaceSelection:
    candidates = {
        "attractions": [_option_digest(option) for option in _options(results, "attractions")],
        "dinners": [_option_digest(option) for option in _options(results, "food")],
        "hotels": [_option_digest(option) for option in _options(results, "hotel")],
        "weather": _summaries(results, "weather"),
        "transport": [option.description for option in _options(results, "transport")][:30],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是行程选点助手，以 JSON 格式输出。只能从给定候选里按 id 选择，不得新增地点。"
                f"每天最多 {ATTRACTIONS_PER_DAY} 个景点、1 顿晚餐；景点总数不超过天数×{ATTRACTIONS_PER_DAY}。"
                "按用户偏好、人数、预算取舍；天气显示下雨的日期，室内景点可用 preferred_day 指定到那天。"
                "选一个落脚酒店（hotel_id），尽量靠近所选景点。顺序和分天由后续算法决定，不需要你排。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"需求：{requirement.model_dump_json()}\n"
                f"候选：{json.dumps(candidates, ensure_ascii=False)}"
                + (
                    "\n上一版方案未通过检查，请调整：\n" + "\n".join(f"- {item.message}" for item in feedback)
                    if feedback
                    else ""
                )
            ),
        },
    ]
    response = await llm.with_structured_output(PlaceSelection).ainvoke(messages)
    return PlaceSelection.model_validate(response)


def _keyword_score(option: CandidateOption, requirement: TravelRequirement) -> int:
    text = f"{option.name}{option.description}"
    keywords = [*requirement.styles, *requirement.food_preferences]
    return sum(1 for keyword in keywords if keyword and keyword in text)


def select_places_by_rules(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    feedback: list[Violation],
    *,
    budget_cuts: int = 0,
) -> PlaceSelection:
    """没有 LLM 时的选点：偏好命中多的优先，其余保持子 Agent 给出的顺序。

    每次因超预算重选，就多去掉一个景点，并改选最便宜的酒店。
    """
    over_budget = budget_cuts > 0
    attractions = sorted(
        _options(results, "attractions"),
        key=lambda option: (
            -_keyword_score(option, requirement),
            (option.estimated_cost or 0) if over_budget else 0,
        ),
    )
    dinners = sorted(_options(results, "food"), key=lambda option: -_keyword_score(option, requirement))
    hotels = _options(results, "hotel")
    hotels = sorted(hotels, key=lambda option: option.estimated_cost or 0) if over_budget else hotels
    limit = max(requirement.days * ATTRACTIONS_PER_DAY - budget_cuts, 0)
    return PlaceSelection(
        hotel_id=hotels[0].id if hotels else None,
        attractions=[SelectedPlace(id=option.id) for option in attractions[:limit]],
        dinners=[SelectedPlace(id=option.id) for option in dinners[: requirement.days]],
    )


def _sanitize(selection: PlaceSelection, requirement: TravelRequirement, results: list[WorkerResult]) -> tuple[PlaceSelection, list[str]]:
    """丢掉不存在或重复的 id、超出容量的景点和越界的 preferred_day。"""
    warnings: list[str] = []
    valid = {
        "attractions": {option.id for option in _options(results, "attractions")},
        "dinners": {option.id for option in _options(results, "food")},
    }

    def clean(items: list[SelectedPlace], kind: str, limit: int) -> list[SelectedPlace]:
        kept: list[SelectedPlace] = []
        for item in items:
            if item.id not in valid[kind] or any(existing.id == item.id for existing in kept):
                warnings.append(f"itinerary_selection_dropped:{kind}")
                continue
            if item.preferred_day is not None and item.preferred_day > requirement.days:
                item = item.model_copy(update={"preferred_day": None})
            kept.append(item)
        return kept[:limit]

    hotel_ids = {option.id for option in _options(results, "hotel")}
    return (
        PlaceSelection(
            hotel_id=selection.hotel_id if selection.hotel_id in hotel_ids else None,
            attractions=clean(selection.attractions, "attractions", requirement.days * ATTRACTIONS_PER_DAY),
            dinners=clean(selection.dinners, "dinners", requirement.days),
            notes=selection.notes,
        ),
        warnings,
    )


# ---------------------------------------------------------------- 2. 排程


def assign_days(
    attraction_ids: list[str],
    preferred: dict[str, int],
    days: int,
    hotel_id: str | None,
    costs: TravelCosts,
) -> list[list[str]]:
    """把景点分到每天（每天最多 ATTRACTIONS_PER_DAY 个），并排好每天内部顺序。

    先放用户/LLM 指定了日期的景点；剩下的按"离酒店最近的作为当天起点，再补离它
    最近的"贪心成组，让同一天的景点彼此靠近。
    """
    plan: list[list[str]] = [[] for _ in range(days)]
    remaining = list(attraction_ids)
    for place_id in list(remaining):
        day = preferred.get(place_id)
        if day is not None and len(plan[day - 1]) < ATTRACTIONS_PER_DAY:
            plan[day - 1].append(place_id)
            remaining.remove(place_id)

    for day_places in plan:
        while remaining and len(day_places) < ATTRACTIONS_PER_DAY:
            anchor = day_places[-1] if day_places else hotel_id
            nearest = min(remaining, key=lambda place_id: (costs.cost(anchor, place_id), remaining.index(place_id)))
            day_places.append(nearest)
            remaining.remove(nearest)

    return [_best_order(day_places, hotel_id, costs) for day_places in plan]


def _best_order(place_ids: list[str], hotel_id: str | None, costs: TravelCosts) -> list[str]:
    """每天只有两三个点，直接枚举所有顺序取总路程最短的。"""
    if len(place_ids) < 2:
        return list(place_ids)

    def route_cost(order: tuple[str, ...]) -> int:
        stops = [hotel_id, *order]
        return sum(costs.cost(a, b) for a, b in zip(stops, stops[1:]))

    return list(min(permutations(place_ids), key=route_cost))


def build_itinerary(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    selection: PlaceSelection,
    costs: TravelCosts,
) -> list[ItineraryDay]:
    options = {option.id: option for worker in ("attractions", "food") for option in _options(results, worker)}
    preferred = {item.id: item.preferred_day for item in selection.attractions if item.preferred_day}
    day_plans = assign_days(
        [item.id for item in selection.attractions], preferred, requirement.days, selection.hotel_id, costs
    )
    dinner_pool = [item.id for item in selection.dinners]
    used_dinners: set[str] = set()

    itinerary: list[ItineraryDay] = []
    for index, place_ids in enumerate(day_plans):
        slots: list[TimeSlot] = []
        previous = selection.hotel_id
        for period, place_id in zip(("morning", "afternoon"), place_ids):
            slots.append(_slot(period, options[place_id], costs.minutes(previous, place_id)))
            previous = place_id
        if len(place_ids) < 1:
            slots.append(TimeSlot(period="morning", title=f"{requirement.destination}分区自由活动",
                                  description="该时段没有留下带证据的景点候选。"))
        if len(place_ids) < 2:
            slots.append(TimeSlot(period="afternoon", title="自由活动",
                                  description="没有第二个带证据的景点候选可安排。"))

        dinner_id = _pick_dinner(dinner_pool, used_dinners, previous, costs)
        if dinner_id is not None:
            used_dinners.add(dinner_id)
            slots.append(_slot("evening", options[dinner_id], costs.minutes(previous, dinner_id)))
        else:
            slots.append(TimeSlot(period="evening", title="自由安排晚餐", description="没有带证据的餐饮候选可安排。"))

        notes = [] if place_ids else [f"第{index + 1}天已无未使用的景点候选。"]
        itinerary.append(
            ItineraryDay(
                day=index + 1,
                date=requirement.departure_date.fromordinal(requirement.departure_date.toordinal() + index),
                slots=slots,
                notes=notes,
            )
        )
    return itinerary


def _pick_dinner(pool: list[str], used: set[str], previous: str | None, costs: TravelCosts) -> str | None:
    if not pool:
        return None
    unused = [dinner for dinner in pool if dinner not in used]
    candidates = unused or pool
    return min(candidates, key=lambda dinner: (costs.cost(previous, dinner), candidates.index(dinner)))


def _slot(period: str, option: CandidateOption, travel_minutes: int | None) -> TimeSlot:
    opening = option.attributes.get("opening_hours")
    location = option.attributes.get("location") or option.attributes.get("district")
    return TimeSlot(
        period=period,
        title=option.name,
        description=option.description,
        location=str(location) if location else None,
        travel_minutes=travel_minutes,
        opening_window=str(opening) if opening else None,
        estimated_cost=option.estimated_cost,
    )


# ---------------------------------------------------------------- 3. 检查


def _opening_minutes(window: str | None) -> tuple[int, int] | None:
    match = _OPENING_PATTERN.search(window or "")
    if not match:
        return None
    open_h, open_m, close_h, close_m = (int(value) for value in match.groups())
    return open_h * 60 + open_m, close_h * 60 + close_m


def check_itinerary(requirement: TravelRequirement, results: list[WorkerResult], itinerary: list[ItineraryDay]) -> list[Violation]:
    violations: list[Violation] = []
    for day in itinerary:
        for slot in day.slots:
            window = _opening_minutes(slot.opening_window)
            limits = _PERIOD_WINDOWS.get(slot.period)
            if window is None or limits is None:
                continue
            latest_open, earliest_close = limits
            if latest_open is not None and window[0] > latest_open:
                violations.append(Violation(code="opening_hours",
                                            message=f"第{day.day}天上午的{slot.title}开门太晚（{slot.opening_window}）。"))
            if earliest_close is not None and window[1] < earliest_close:
                violations.append(Violation(code="opening_hours",
                                            message=f"第{day.day}天下午的{slot.title}关门太早（{slot.opening_window}）。"))
        travel = sum(slot.travel_minutes or 0 for slot in day.slots)
        if travel > MAX_DAILY_TRAVEL_MINUTES:
            violations.append(Violation(code="daily_travel",
                                        message=f"第{day.day}天路上耗时约 {travel} 分钟，超过 {MAX_DAILY_TRAVEL_MINUTES} 分钟。"))

    if requirement.budget is not None:
        budget = calculate_budget(requirement, results, itinerary)
        if budget.total_estimate is not None and budget.total_estimate > requirement.budget:
            violations.append(Violation(
                code="over_budget",
                message=f"已知费用约 {budget.total_estimate:.0f} 元，超出预算 {requirement.budget:.0f} 元，请换便宜的选项或少排收费景点。",
            ))
    return violations


# ---------------------------------------------------------------- 编排入口


async def plan_itinerary(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    *,
    llm: Any | None = None,
) -> tuple[list[ItineraryDay], list[str]]:
    if not _options(results, "attractions") and not _options(results, "food"):
        # 什么候选都没有时沿用原来的逐日兜底骨架。
        return schedule_itinerary(requirement, results)

    costs = TravelCosts(results)
    warnings: list[str] = []
    feedback: list[Violation] = []
    itinerary: list[ItineraryDay] = []
    budget_cuts = 0
    for _round in range(MAX_ROUNDS):
        selection = None
        if llm is not None:
            try:
                selection = await select_places_with_llm(llm, requirement, results, feedback)
            except Exception as exc:
                app_logger.warning(f"LLM 选点失败，改用规则选点: {type(exc).__name__}: {exc}")
                warnings.append("itinerary_selection_fallback:llm_failed")
                llm = None
        if selection is None:
            budget_cuts += any(item.code == "over_budget" for item in feedback)
            selection = select_places_by_rules(requirement, results, feedback, budget_cuts=budget_cuts)
        selection, sanitize_warnings = _sanitize(selection, requirement, results)
        warnings.extend(sanitize_warnings)

        itinerary = build_itinerary(requirement, results, selection, costs)
        feedback = check_itinerary(requirement, results, itinerary)
        if not feedback:
            break
    warnings.extend(f"itinerary_constraint:{item.code}:{item.message}" for item in feedback)
    return itinerary, list(dict.fromkeys(warnings))


__all__ = [
    "PlaceSelection",
    "SelectedPlace",
    "TravelCosts",
    "Violation",
    "assign_days",
    "build_itinerary",
    "check_itinerary",
    "plan_itinerary",
    "select_places_by_rules",
    "select_places_with_llm",
]
