"""Deterministic itinerary scheduling."""

from __future__ import annotations

from collections.abc import Iterable

from app.schemas.planning import (
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


__all__ = ["schedule_itinerary"]
