"""3. 检查与兜底：找出违规反馈给模型重选；模型几轮都改不掉的，由算法修。"""

from __future__ import annotations

from app.agents.trip_cost import estimate_trip_cost
from app.places.catalog import named_by
from app.schemas.planning import CandidateOption, ItineraryDay, TravelRequirement, WorkerResult
from app.agents.itinerary.rules import (
    _candidates,
    _DINNER_WINDOW,
    _fits,
    _LUNCH_WINDOW,
    MAX_DAILY_TRAVEL_MINUTES,
    MAX_DINNER_MINUTES,
    MAX_HOP_MINUTES,
    _open_during,
    _opening_minutes,
    _options,
    PlaceSelection,
    same_place,
    SIGHT_PERIOD_WINDOWS,
    Violation,
)
from app.agents.itinerary.travel import TravelCosts
from app.agents.itinerary.scheduling import build_itinerary, _slot


def check_itinerary(requirement: TravelRequirement, results: list[WorkerResult], itinerary: list[ItineraryDay]) -> list[Violation]:
    violations: list[Violation] = []
    costs = TravelCosts(results)
    ids_by_name = {
        option.name: option.id for worker in ("attractions", "food") for option in _options(results, worker)
    }
    for day in itinerary:
        for slot in day.slots:
            window = _opening_minutes(slot.opening_window)
            limits = SIGHT_PERIOD_WINDOWS.get(slot.period)
            if window is None or limits is None:
                continue
            latest_open, earliest_close = limits
            label = "上午" if slot.period == "morning" else "下午"
            if latest_open is not None and window[0] > latest_open:
                violations.append(Violation(code="opening_hours",
                                            message=f"第{day.day}天{label}的{slot.title}开门太晚（{slot.opening_window}）。"))
            if earliest_close is not None and window[1] < earliest_close:
                violations.append(Violation(code="opening_hours",
                                            message=f"第{day.day}天{label}的{slot.title}关门太早（{slot.opening_window}）。"))
        travel = sum(slot.travel_minutes or 0 for slot in day.slots)
        if travel > MAX_DAILY_TRAVEL_MINUTES:
            violations.append(Violation(code="daily_travel",
                                        message=f"第{day.day}天路上耗时约 {travel} 分钟，超过 {MAX_DAILY_TRAVEL_MINUTES} 分钟。"))
            continue
        # 没有真实耗时的路段按坐标估算：防止把相隔几十公里的景点排进同一天。
        # 只看景点之间；去郊外玩一天再回城里吃晚饭是正常安排，不算违规。
        visited = [slot for slot in day.slots if slot.period in ("morning", "afternoon") and slot.title in ids_by_name]
        stops = [ids_by_name[slot.title] for slot in visited]
        estimated = sum(costs.known(first, second) or 0 for first, second in zip(stops, stops[1:]))
        if estimated > MAX_DAILY_TRAVEL_MINUTES:
            names = "、".join(slot.title for slot in visited)
            violations.append(Violation(code="daily_travel",
                                        message=f"第{day.day}天的{names}相距太远，路上估计要 {estimated} 分钟以上。"))
    for day in itinerary:
        for slot in day.slots:
            if slot.period == "evening" and (slot.travel_minutes or 0) > MAX_DINNER_MINUTES:
                violations.append(Violation(
                    code="dinner_far",
                    message=f"第{day.day}天晚饭{slot.title}离上一站约 {slot.travel_minutes} 分钟，太远了。",
                ))
    titles = [slot.title for day in itinerary for slot in day.slots]
    for term in requirement.must_visit:
        if any(named_by(title, [term]) for title in titles):
            continue
        known = any(named_by(option.name, [term]) for worker in ("attractions", "food") for option in _options(results, worker))
        message = f"你点名要去的{term}没有排进行程，请按 id 选上。" if known else f"没查到{term}的资料，没能排进行程，可以换个叫法再试。"
        violations.append(Violation(code="must_visit", message=message))
    return violations


def trim_to_budget(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    itinerary: list[ItineraryDay],
    hotel: CandidateOption | None,
    costs: TravelCosts,
) -> list[ItineraryDay]:
    """模型压不进预算时的兜底：先换饭，再换门票，从最贵的开始，直到不超预算。

    饭换成离上一站 45 分钟内、饭点营业、没吃过（含同连锁）的更便宜餐厅；
    景点换成离上一站不远、该时段开放、没去过的更便宜景点。用户点名要去的不换。
    """
    hotel_cost = hotel.estimated_cost if hotel else None
    by_name = {option.name: option for worker in ("attractions", "food") for option in _options(results, worker)}
    days = [day.model_copy(deep=True) for day in itinerary]

    def within_budget() -> bool:
        return estimate_trip_cost(requirement, days, hotel_cost).total <= (requirement.budget or 0)

    def previous_of(day_index: int, slot_index: int) -> str | None:
        title = days[day_index].slots[slot_index - 1].title if slot_index else None
        return by_name[title].id if title in by_name else (hotel.id if hotel else None)

    def positions(periods: tuple[str, ...]) -> list[tuple[int, int]]:
        found = [
            (day_index, slot_index)
            for day_index, day in enumerate(days)
            for slot_index, slot in enumerate(day.slots)
            if slot.period in periods and slot.estimated_cost and not named_by(slot.title, requirement.must_visit)
        ]
        return sorted(found, key=lambda pos: -(days[pos[0]].slots[pos[1]].estimated_cost or 0))

    def swap(periods: tuple[str, ...], worker: str, fits, max_minutes: int) -> None:
        pool = [option for option in _candidates(results, worker, requirement) if option.estimated_cost is not None]
        for day_index, slot_index in positions(periods):
            if within_budget():
                return
            slot = days[day_index].slots[slot_index]
            previous = previous_of(day_index, slot_index)
            used = [other.title for day in days for other in day.slots if other.period in periods]
            cheaper = [
                option for option in pool
                if option.estimated_cost < (slot.estimated_cost or 0)
                and not any(same_place(option.name, title, requirement.destination) for title in used)
                and fits(option, slot.period)
                and (previous is None or costs.cost(previous, option.id) <= max_minutes)
            ]
            if cheaper:
                best = min(cheaper, key=lambda option: (option.estimated_cost, costs.cost(previous, option.id)))
                days[day_index].slots[slot_index] = _slot(slot.period, best, *costs.leg(previous, best.id))

    def meal_fits(option: CandidateOption, period: str) -> bool:
        return _open_during(option, _LUNCH_WINDOW if period == "lunch" else _DINNER_WINDOW)

    swap(("lunch", "evening"), "food", meal_fits, MAX_DINNER_MINUTES)
    swap(("morning", "afternoon"), "attractions", lambda option, period: _fits(option, period), MAX_HOP_MINUTES)
    return days


_DISTANCE_CODES = ("daily_travel", "dinner_far")


def drop_outlying_sights(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    selection: PlaceSelection,
    itinerary: list[ItineraryDay],
    costs: TravelCosts,
) -> list[ItineraryDay] | None:
    """模型几轮都没解决"路上太久/晚饭太远"时的兜底：去掉那天最偏的景点，让算法就近补位重排。

    典型情况是上海一天里排"黄浦江 → 临港的天文馆 → 回市区吃晚饭"，路上三个多小时。
    最偏的 = 进出它的路程之和最长的那个；用户点名要去的不动。没有可去的就返回 None。
    """
    by_name = {option.name: option for option in _options(results, "attractions")}
    dropped: list[str] = []
    for day in itinerary:
        travel = sum(slot.travel_minutes or 0 for slot in day.slots)
        dinner = next((slot.travel_minutes or 0 for slot in day.slots if slot.period == "evening"), 0)
        if travel <= MAX_DAILY_TRAVEL_MINUTES and dinner <= MAX_DINNER_MINUTES:
            continue
        outlay = {}
        for index, slot in enumerate(day.slots):
            if slot.period not in ("morning", "afternoon") or slot.title not in by_name:
                continue
            if named_by(slot.title, requirement.must_visit):
                continue
            leaving = day.slots[index + 1].travel_minutes if index + 1 < len(day.slots) else 0
            outlay[slot.title] = (slot.travel_minutes or 0) + (leaving or 0)
        if outlay:
            dropped.append(max(outlay, key=outlay.get))
    if not dropped:
        return None
    ids = {by_name[name].id for name in dropped}
    narrowed = requirement.model_copy(update={"avoid_places": [*requirement.avoid_places, *dropped]})
    trimmed = selection.model_copy(update={"attractions": [item for item in selection.attractions if item.id not in ids]})
    return build_itinerary(narrowed, results, trimmed, costs)


def _budget_violations(
    requirement: TravelRequirement, itinerary: list[ItineraryDay], hotel: CandidateOption | None
) -> list[Violation]:
    if not requirement.budget:
        return []
    total = estimate_trip_cost(requirement, itinerary, hotel.estimated_cost if hotel else None).total
    if total <= requirement.budget:
        return []
    return [Violation(code="budget", message=f"按地点库价格估算约 {total:.0f} 元，超出预算 {requirement.budget:.0f} 元。")]
