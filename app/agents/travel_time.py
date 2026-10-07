"""市内地点两两之间的交通耗时。

交通子 Agent 拿到景点、住宿、餐饮候选后，用 TravelTimeService 算出每两个地点之间
怎么走、要多久。服务按顺序询问各个 provider，前一个答不上的地点对交给下一个：

    交通图谱缓存（route_graph）→ 高德路线（amap）→ 按所在区粗估（district_estimate）

粗估永远兜底，所以每一对地点都会有结果；但粗估只给"同区 / 不同区"，不给分钟数，
没有真实数据时不编造耗时。图谱和高德 provider 只要实现 TravelTimeProvider 即可接入，
排日程那一侧不需要改。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from itertools import combinations
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from app.rag.identifiers import stable_hash
from app.schemas.planning import CandidateOption


Proximity = Literal["same_district", "different_district", "unknown"]

_DISTRICT_PATTERN = re.compile(r"(?:位于|地处|坐落于)([^。，,；;\s]{2,10}?(?:区|县|市))")


class Place(BaseModel):
    """参与交通计算的一个地点（景点 / 酒店 / 餐厅）。"""

    id: str
    name: str
    category: str
    district: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class TravelLeg(BaseModel):
    """两个地点之间的一段交通。minutes / distance_km 为空表示没有真实数据。"""

    from_id: str
    to_id: str
    from_name: str
    to_name: str
    minutes: int | None = Field(default=None, ge=0)
    distance_km: float | None = Field(default=None, ge=0)
    mode: str | None = None
    proximity: Proximity = "unknown"
    source: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def leg_id(self) -> str:
        return "leg-" + stable_hash(self.from_id, self.to_id)[:16]


class TravelTimeProvider(Protocol):
    """回答一批地点对的交通耗时；答不上的对不返回，交给下一个 provider。"""

    name: str

    async def legs(self, city: str, pairs: Sequence[tuple[Place, Place]]) -> list[TravelLeg]: ...


class DistrictEstimateProvider:
    """没有坐标和路线数据时的兜底：只判断两地是否同区，不给分钟数。"""

    name = "district_estimate"

    async def legs(self, city: str, pairs: Sequence[tuple[Place, Place]]) -> list[TravelLeg]:
        return [
            TravelLeg(
                from_id=origin.id,
                to_id=target.id,
                from_name=origin.name,
                to_name=target.name,
                proximity=self._proximity(origin, target),
                source=self.name,
            )
            for origin, target in pairs
        ]

    @staticmethod
    def _proximity(origin: Place, target: Place) -> Proximity:
        if not origin.district or not target.district:
            return "unknown"
        return "same_district" if origin.district == target.district else "different_district"


class TravelTimeService:
    def __init__(self, providers: Iterable[TravelTimeProvider]):
        self.providers = list(providers)

    async def compute(self, city: str, places: Sequence[Place]) -> list[TravelLeg]:
        """返回所有地点两两之间的交通段（无向，每对只算一次）。"""
        remaining = list(combinations(places, 2))
        legs: list[TravelLeg] = []
        for provider in self.providers:
            if not remaining:
                break
            try:
                answered = await provider.legs(city, remaining)
            except Exception:
                # 某个数据源挂了不影响整体，剩下的地点对交给后面的 provider。
                continue
            answered_keys = {frozenset((leg.from_id, leg.to_id)) for leg in answered}
            legs.extend(answered)
            remaining = [
                pair for pair in remaining if frozenset((pair[0].id, pair[1].id)) not in answered_keys
            ]
        return legs


def default_travel_time_service() -> TravelTimeService:
    return TravelTimeService([DistrictEstimateProvider()])


def place_from_option(option: CandidateOption) -> Place:
    """从候选里取出地点信息。区优先看结构化字段，其次从描述里的"位于××区"解析。"""
    attributes = option.attributes
    district = attributes.get("district") or attributes.get("location")
    if not district:
        match = _DISTRICT_PATTERN.search(option.description)
        district = match.group(1) if match else None
    return Place(
        id=option.id,
        name=option.name,
        category=option.category,
        district=str(district) if district else None,
        latitude=_as_float(attributes.get("latitude")),
        longitude=_as_float(attributes.get("longitude")),
    )


def _as_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def describe_leg(leg: TravelLeg, origin: Place, target: Place) -> str:
    """面向用户的一句话说明，同时作为这段交通的证据正文。"""
    if leg.minutes is not None:
        parts = [f"约 {leg.minutes} 分钟"]
        if leg.distance_km is not None:
            parts.append(f"{leg.distance_km:.1f} 公里")
        if leg.mode:
            parts.append(leg.mode)
        return f"{origin.name} → {target.name}：{'，'.join(parts)}（来源：{leg.source}）。"
    if leg.proximity == "same_district":
        return f"{origin.name}与{target.name}同在{origin.district}，通常距离较近；具体耗时需实时确认。"
    if leg.proximity == "different_district":
        return (
            f"{origin.name}（{origin.district}）与{target.name}（{target.district}）不在同一区，"
            "可能需要较长交通时间；具体耗时需实时确认。"
        )
    return f"{origin.name}与{target.name}之间缺少位置信息，交通耗时需实时确认。"


__all__ = [
    "DistrictEstimateProvider",
    "Place",
    "TravelLeg",
    "TravelTimeProvider",
    "TravelTimeService",
    "default_travel_time_service",
    "describe_leg",
    "place_from_option",
]
