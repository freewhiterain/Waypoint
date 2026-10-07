"""Deterministic itinerary scheduling and evidence-backed budget calculation."""

from __future__ import annotations

from collections.abc import Iterable

from app.schemas.planning import (
    BudgetSummary,
    CandidateOption,
    ItineraryDay,
    TimeSlot,
    TravelRequirement,
    WorkerResult,
)


def _by_worker(results: Iterable[WorkerResult], worker: str) -> WorkerResult | None:
    return next((result for result in results if result.worker == worker), None)


def _slot_from_option(period: str, option: CandidateOption) -> TimeSlot:
    attributes = option.attributes
    travel_minutes = attributes.get("travel_minutes")
    if not isinstance(travel_minutes, int) or travel_minutes < 0:
        travel_minutes = None
    location = attributes.get("location")
    opening_window = attributes.get("opening_hours")
    return TimeSlot(
        period=period,
        title=option.name,
        description=option.description,
        location=str(location) if location else None,
        travel_minutes=travel_minutes,
        opening_window=str(opening_window) if opening_window else None,
        estimated_cost=option.estimated_cost,
        evidence_indexes=[],
    )


def schedule_itinerary(
    requirement: TravelRequirement,
    results: list[WorkerResult],
) -> tuple[list[ItineraryDay], list[str]]:
    """Assign distinct grounded candidates to stable day/period slots."""

    attraction_result = _by_worker(results, "attractions")
    food_result = _by_worker(results, "food")
    attractions = list(attraction_result.options if attraction_result else [])
    foods = list(food_result.options if food_result else [])
    used_attractions: set[str] = set()
    warnings: list[str] = []
    itinerary: list[ItineraryDay] = []

    for offset in range(requirement.days):
        selected: list[CandidateOption] = []
        for _ in range(2):
            candidate = next(
                (item for item in attractions if item.name not in used_attractions),
                None,
            )
            if candidate is None:
                break
            selected.append(candidate)
            used_attractions.add(candidate.name)

        day_notes: list[str] = []
        if not selected:
            day_notes.append(f"第{offset + 1}天已无未使用的景点候选。")
        morning = (
            _slot_from_option("morning", selected[0])
            if selected
            else TimeSlot(
                period="morning",
                title=f"{requirement.destination}分区自由活动",
                description="该时段没有留下带证据的景点候选。",
            )
        )
        afternoon = (
            _slot_from_option("afternoon", selected[1])
            if len(selected) > 1
            else TimeSlot(
                period="afternoon",
                title="自由活动",
                description="没有第二个带证据的景点候选可安排。",
            )
        )
        evening = (
            _slot_from_option("evening", foods[offset % len(foods)])
            if foods
            else TimeSlot(
                period="evening",
                title="自由安排晚餐",
                description="没有带证据的餐饮候选可安排。",
            )
        )
        itinerary.append(
            ItineraryDay(
                day=offset + 1,
                date=requirement.departure_date.fromordinal(
                    requirement.departure_date.toordinal() + offset
                ),
                slots=[morning, afternoon, evening],
                notes=day_notes,
            )
        )
    return itinerary, warnings


def _slot_cost(slots: list[TimeSlot], *, multiplier: int) -> float | None:
    priced = [slot.estimated_cost for slot in slots if slot.estimated_cost is not None]
    if not priced:
        return None
    return float(sum(priced) * multiplier)


def _cheapest(options: list[CandidateOption]) -> CandidateOption | None:
    priced = [option for option in options if option.estimated_cost is not None]
    return min(priced, key=lambda option: option.estimated_cost) if priced else None


def calculate_budget(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    itinerary: list[ItineraryDay] | None = None,
) -> BudgetSummary:
    """Estimate the cost of what was actually scheduled, from evidence-backed prices only.

    门票和餐饮只算排进行程的时段，而不是把所有候选价格相加；按人数相乘；
    住宿只选一家、按晚数和房间数算；交通候选也只取一个。缺价格的类别不计入总额。
    """

    if itinerary is None:
        itinerary, _warnings = schedule_itinerary(requirement, results)
    people = requirement.adults + requirement.children
    nights = max(requirement.days - 1, 0)
    # 两个大人一间房，孩子与大人同住。
    rooms = max(-(-requirement.adults // 2), 1)

    slots = [slot for day in itinerary for slot in day.slots]
    attraction_slots = [slot for slot in slots if slot.period in {"morning", "afternoon"}]
    food_slots = [slot for slot in slots if slot.period == "evening"]

    hotel_result = _by_worker(results, "hotel")
    hotel = _cheapest(list(hotel_result.options)) if hotel_result else None
    accommodation_cost: float | None = None
    if nights == 0:
        accommodation_cost = 0.0
    elif hotel is not None:
        if hotel.attributes.get("pricing_unit") == "per_night":
            accommodation_cost = float(hotel.estimated_cost * nights * rooms)
        else:
            # 没标计价单位时按整段住宿的单间总价处理。
            accommodation_cost = float(hotel.estimated_cost * rooms)

    transport_result = _by_worker(results, "transport")
    transport = _cheapest(list(transport_result.options)) if transport_result else None

    categories = {
        "transport": float(transport.estimated_cost * people) if transport is not None else None,
        "accommodation": accommodation_cost,
        "food": _slot_cost(food_slots, multiplier=people),
        "attractions": _slot_cost(attraction_slots, multiplier=people),
        "misc": None,
    }
    known_costs = [value for value in categories.values() if value is not None]
    total = float(sum(known_costs)) if known_costs else None

    party = f"{requirement.adults} 个大人" + (f"、{requirement.children} 个孩子" if requirement.children else "")
    notes = [
        "预算只累加有研究证据支撑的候选价格，且只算排进行程的项目。",
        f"门票、餐饮、交通按 {party}（共 {people} 人）计算。",
        f"住宿按 {nights} 晚、{rooms} 间房计算。" if nights else "当天往返，不计住宿。",
        "缺失的类别不计入总估算。",
    ]
    if requirement.budget is None:
        notes.insert(0, "用户未提供明确预算。")
    else:
        notes.insert(0, f"用户预算上限：{requirement.budget:.2f} 元。")
        if total is not None and total > requirement.budget:
            notes.insert(1, f"已知费用估算 {total:.2f} 元，超出预算 {total - requirement.budget:.2f} 元。")
    return BudgetSummary(
        total_estimate=total,
        categories=categories,
        notes=notes,
    )


__all__ = ["calculate_budget", "schedule_itinerary"]
