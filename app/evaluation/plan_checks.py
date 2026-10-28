"""对结构化行程做可程序判定的约束检查（参照 ChinaTravel 的两层通过率）。

- environment：行程里的地点、门票价、营业时间能在地点库里对上，不是编出来的。
  只对地点库覆盖的城市做；库外城市靠联网搜索，没有可对照的真值。
- logic：天数和日期、不重复、用户说不去的地点、轻松的天、每天有晚饭、
  时段落在营业时间内。

评测脚本和单测共用这里的判定，线上排程也可以拿它做自检。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from app.agents.itinerary_planner import same_place
from app.places.catalog import PlaceCatalog, get_place_catalog
from app.schemas.planning import ItineraryDay, TravelRequirement

Category = Literal["environment", "logic"]


@dataclass(frozen=True)
class PlanIssue:
    category: Category
    message: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.message}"


# 每个时段至少要覆盖的营业区间（时:分 → 分钟）。景点只要求开门早于时段中段、
# 关门晚于时段结束前；餐厅要覆盖正常用餐时间。
_PERIOD_NEEDS = {
    "morning": (10 * 60, 11 * 60 + 30),
    "lunch": (12 * 60, 12 * 60 + 30),
    "afternoon": (14 * 60, 16 * 60),
    "evening": (18 * 60, 19 * 60),
    "night": (20 * 60, 20 * 60 + 30),
}
_WINDOW = re.compile(r"(\d{1,2}):(\d{2})\s*[-–~至]\s*(\d{1,2}):(\d{2})")
_PLACEHOLDER_PREFIXES = ("自由", )
_SIGHT_PERIODS = ("morning", "afternoon")


def _is_placeholder(title: str, city: str) -> bool:
    return title.startswith(_PLACEHOLDER_PREFIXES) or title.startswith(f"{city}分区")


def _window_minutes(window: str | None) -> tuple[int, int] | None:
    if not window:
        return None
    if window.strip() == "全天":
        return 0, 24 * 60
    match = _WINDOW.search(window)
    if not match:
        return None
    open_h, open_m, close_h, close_m = map(int, match.groups())
    opens, closes = open_h * 60 + open_m, close_h * 60 + close_m
    if closes <= opens:  # 跨过午夜营业
        closes += 24 * 60
    return opens, closes


def check_plan(
    requirement: TravelRequirement,
    itinerary: list[ItineraryDay],
    catalog: PlaceCatalog | None = None,
) -> list[PlanIssue]:
    catalog = catalog or get_place_catalog()
    city = requirement.destination
    issues: list[PlanIssue] = []

    def logic(message: str) -> None:
        issues.append(PlanIssue("logic", message))

    if len(itinerary) != requirement.days:
        logic(f"天数 {len(itinerary)}，要求 {requirement.days}")
    for index, day in enumerate(itinerary):
        expected = requirement.departure_date + timedelta(days=index)
        if day.date != expected:
            logic(f"第{day.day}天日期 {day.date}，应为 {expected}")

    visited: list[tuple[int, str]] = []
    for day in itinerary:
        places = [slot for slot in day.slots if not _is_placeholder(slot.title, city)]
        for slot in places:
            for seen_day, seen in visited:
                if same_place(slot.title, seen, city):
                    logic(f"「{slot.title}」第{seen_day}天已经去过，第{day.day}天又排了")
                    break
            visited.append((day.day, slot.title))
            for avoided in requirement.avoid_places:
                if avoided in slot.title or same_place(slot.title, avoided, city):
                    logic(f"用户说不去「{avoided}」，第{day.day}天仍排了「{slot.title}」")
            window = _window_minutes(slot.opening_window)
            needs = _PERIOD_NEEDS.get(slot.period)
            if window and needs and (window[0] > needs[0] or window[1] < needs[1]):
                logic(f"第{day.day}天{slot.period}「{slot.title}」营业 {slot.opening_window}，时段对不上")

        if not any(slot.period == "evening" for slot in day.slots):
            logic(f"第{day.day}天没有晚饭")
        sights = [slot for slot in places if slot.period in _SIGHT_PERIODS]
        if day.day in requirement.relaxed_days and len(sights) > 1:
            logic(f"第{day.day}天要求轻松，却排了 {len(sights)} 个景点")

    if city in catalog.cities():
        issues.extend(_environment_issues(city, itinerary, catalog))
    return issues


def _environment_issues(city: str, itinerary: list[ItineraryDay], catalog: PlaceCatalog) -> list[PlanIssue]:
    issues = []
    for day in itinerary:
        for slot in day.slots:
            if _is_placeholder(slot.title, city):
                continue
            record = catalog.find(city, slot.title)
            if record is None:
                issues.append(PlanIssue("environment", f"第{day.day}天「{slot.title}」不在{city}地点库里"))
                continue
            if record.price is not None and slot.estimated_cost is not None and abs(record.price - slot.estimated_cost) > 1:
                issues.append(PlanIssue(
                    "environment", f"「{slot.title}」费用写 {slot.estimated_cost:g} 元，地点库是 {record.price:g} 元"
                ))
            if record.opening_hours and slot.opening_window and slot.opening_window != record.opening_hours:
                issues.append(PlanIssue(
                    "environment", f"「{slot.title}」营业时间写 {slot.opening_window}，地点库是 {record.opening_hours}"
                ))
    return issues
