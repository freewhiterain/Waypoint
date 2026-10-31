"""地点之间的交通代价（实测优先，其次按坐标估算）和行程每天的天气。"""

from __future__ import annotations

import re

from app.places.catalog import straight_line_km
from app.schemas.planning import TravelRequirement, WorkerResult
from app.agents.itinerary.rules import _options


# 没有真实耗时时用于比较远近的内部代价（分钟量级），不会展示给用户。
_PROXIMITY_COST = {"same_district": 20, "unknown": 35, "different_district": 50}


_WEATHER_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})：")


def _summaries(results: list[WorkerResult], worker: str) -> list[str]:
    return [result.summary for result in results if result.worker == worker and result.summary]


def weather_by_date(results: list[WorkerResult]) -> dict[str, str]:
    """按日期取天气预报原文：{"2026-10-12": "晴 16~25℃；降水概率 5%"}。"""
    forecast: dict[str, str] = {}
    for result in results:
        if result.worker != "weather":
            continue
        for item in result.evidence:
            # 证据治理会精简 metadata，所以日期从正文里取（"成都 2026-10-12：晴 16~25℃…"）。
            match = _WEATHER_DATE.search(item.content)
            if match is None or match.group(1) in forecast:
                continue
            day = match.group(1)
            forecast[day] = item.content.split("：", 1)[-1].strip()
    return forecast


def _trip_weather(requirement: TravelRequirement, results: list[WorkerResult]) -> dict[int, str]:
    """行程每一天（从 1 开始）对应的天气；预报没覆盖到的日期不出现。"""
    forecast = weather_by_date(results)
    start = requirement.departure_date.toordinal()
    days = {index + 1: requirement.departure_date.fromordinal(start + index).isoformat() for index in range(requirement.days)}
    return {day: forecast[iso] for day, iso in days.items() if iso in forecast}


_RAIN_CHANCE = re.compile(r"降水概率\s*(\d+)%")


_OUTDOOR_TYPES = {"公园", "自然风光", "园林", "游乐园/体育娱乐", "商业街区", "人文景观"}


# 降水概率不低于这个值算雨天；要换过去的那天得明显更干。
RAINY_CHANCE, DRY_MARGIN = 70, 30


def _rain_chance(text: str | None) -> int | None:
    match = _RAIN_CHANCE.search(text or "")
    return int(match.group(1)) if match else None


def _minutes_from_km(distance_km: float) -> int:
    """直线距离换算成比较远近用的内部代价：起步 5 分钟，市内每公里约 3.5 分钟。"""
    return round(5 + distance_km * 3.5)


class TravelCosts:
    """地点之间的交通代价：优先真实分钟数，其次按坐标直线距离估，最后按区粗估。"""

    def __init__(self, results: list[WorkerResult]):
        self._minutes: dict[frozenset[str], int] = {}
        self._estimate: dict[frozenset[str], int] = {}
        self._coordinates: dict[str, tuple[float, float]] = {}
        for leg in _options(results, "transport"):
            attributes = leg.attributes
            key = frozenset((str(attributes.get("from_id")), str(attributes.get("to_id"))))
            if isinstance(attributes.get("minutes"), int):
                self._minutes[key] = attributes["minutes"]
            self._estimate[key] = _PROXIMITY_COST.get(str(attributes.get("proximity")), _PROXIMITY_COST["unknown"])
        # 交通子 Agent 只给前几个候选算路线；其余地点只要有坐标，也能比较远近。
        for worker in ("attractions", "food", "hotel"):
            for option in _options(results, worker):
                latitude, longitude = option.attributes.get("latitude"), option.attributes.get("longitude")
                if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
                    self._coordinates[option.id] = (float(latitude), float(longitude))

    def minutes(self, first: str | None, second: str | None) -> int | None:
        if first is None or second is None or first == second:
            return None
        return self._minutes.get(frozenset((first, second)))

    def known(self, first: str | None, second: str | None) -> int | None:
        """有依据的耗时：真实分钟数，或由坐标直线距离换算的估计；两者都没有时返回 None。"""
        if first is None or second is None or first == second:
            return None
        key = frozenset((first, second))
        if key in self._minutes:
            return self._minutes[key]
        if first in self._coordinates and second in self._coordinates:
            return _minutes_from_km(straight_line_km(self._coordinates[first], self._coordinates[second]))
        return None

    def cost(self, first: str | None, second: str | None) -> int:
        if first is None or second is None or first == second:
            return 0
        key = frozenset((first, second))
        known = self.known(first, second)
        return known if known is not None else self._estimate.get(key, _PROXIMITY_COST["unknown"])
