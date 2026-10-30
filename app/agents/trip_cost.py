"""按行程里已有的价格算一趟大概花多少钱，不经过模型。

规则和前端 web/src/lib/cost.ts 一致：门票、餐饮按人头，住宿按晚（一间房）。
排程用它检查预算，评测用它核对，前端用同一套规则展示，三处数字对得上。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.planning import ItineraryDay, TravelRequirement, WorkerResult

_MEAL_PERIODS = ("lunch", "evening")
# 与 itinerary_planner.HOTEL_NOTE_PREFIX 相同；这里不反向依赖排程模块。
_HOTEL_PREFIX = "住宿："


@dataclass(frozen=True)
class TripCost:
    tickets: float
    meals: float
    hotel: float
    # 有多少个时段没有价格数据、没算进去
    unknown: int

    @property
    def total(self) -> float:
        return self.tickets + self.meals + self.hotel


def party_size(requirement: TravelRequirement) -> int:
    return max(1, requirement.adults + requirement.children)


def estimate_trip_cost(
    requirement: TravelRequirement, itinerary: list[ItineraryDay], hotel_price: float | None
) -> TripCost:
    people = party_size(requirement)
    tickets = meals = 0.0
    unknown = 0
    for day in itinerary:
        for slot in day.slots:
            if slot.title.endswith("自由活动") or slot.title.startswith("自由安排"):
                continue
            if slot.estimated_cost is None:
                unknown += 1
            elif slot.period in _MEAL_PERIODS:
                meals += slot.estimated_cost * people
            else:
                tickets += slot.estimated_cost * people
    nights = max(0, len(itinerary) - 1)
    return TripCost(tickets=tickets, meals=meals, hotel=(hotel_price or 0) * nights, unknown=unknown)


def hotel_price(itinerary: list[ItineraryDay], worker_results: list[WorkerResult]) -> float | None:
    """行程第一天的"住宿：酒店名。…"备注 → 住宿候选里那家的每晚价格。"""
    note = next((note for note in (itinerary[0].notes if itinerary else []) if note.startswith(_HOTEL_PREFIX)), "")
    name = note.removeprefix(_HOTEL_PREFIX).split("。")[0]
    for result in worker_results:
        if result.worker == "hotel":
            for option in result.options:
                if option.name == name:
                    return option.estimated_cost
    return None
