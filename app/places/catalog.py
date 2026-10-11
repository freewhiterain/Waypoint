"""地点库：景点、餐厅、住宿的结构化数据（data/places/*.csv）。

子 Agent 在地点库覆盖的城市直接从这里取候选：名称、坐标、开放时间、价格原样带上，
不经过"检索文字 → 模型提取"。模型只负责在排日程时按用户偏好从候选里选。

地点库没有覆盖的城市返回空，子 Agent 照旧走检索流程。
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.rag.identifiers import stable_hash
from app.schemas.planning import TravelRequirement

PlaceCategory = Literal["attractions", "food", "hotel"]

PLACES_DIR = Path(__file__).resolve().parents[2] / "data" / "places"
SOURCE_NAME = "ChinaTravel 地点库"
_FILES: dict[PlaceCategory, str] = {
    "attractions": "attractions.csv",
    "food": "restaurants.csv",
    "hotel": "accommodations.csv",
}

# 候选交给排日程的模型挑选，数量随天数增加，但要控制在提示词放得下的范围。
MIN_SHORTLIST, MAX_SHORTLIST, HOTEL_SHORTLIST = 8, 24, 8
# 不适合当正餐的类型。
_NOT_A_MEAL = {"面包甜点", "咖啡店", "酒吧/酒馆", "茶馆/茶室", ""}
_SPICY = {"川菜", "湘菜", "火锅"}
# 用户常见说法 → 景点类型里的关键词。
_STYLE_ALIASES = {
    "历史": "历史古迹", "古迹": "历史古迹", "人文": "人文景观", "文化": "人文景观", "博物馆": "博物馆",
    "自然": "自然风光", "风景": "自然风光", "爬山": "自然风光", "公园": "公园", "逛街": "商业街区",
    "购物": "商业街区", "艺术": "美术馆", "亲子": "游乐园", "游乐": "游乐园",
}


class PlaceRecord(BaseModel):
    city: str
    category: PlaceCategory
    rank: int
    name: str
    type: str = ""
    latitude: float | None = None
    longitude: float | None = None
    open_time: str = ""
    close_time: str = ""
    price: float | None = None
    max_hours: float | None = None
    recommended_food: str = ""

    @property
    def id(self) -> str:
        return "place-" + stable_hash(self.city, self.category, self.name)[:16]

    @property
    def opening_hours(self) -> str | None:
        if not self.open_time or not self.close_time:
            return None
        if (self.open_time, self.close_time) == ("00:00", "24:00"):
            return "全天"
        return f"{self.open_time}-{self.close_time}"

    def describe(self) -> str:
        """一句话事实描述，同时作为这条候选的证据正文。"""
        if self.category == "attractions":
            parts = [self.type or "景点"]
            if self.opening_hours:
                parts.append("全天开放" if self.opening_hours == "全天" else f"开放时间 {self.opening_hours}")
            if self.price is not None:
                parts.append("免费" if self.price == 0 else f"门票 {self.price:g} 元")
        elif self.category == "food":
            parts = [self.type or "餐厅"]
            if self.price is not None:
                parts.append(f"人均 {self.price:g} 元")
            if self.opening_hours:
                parts.append("全天营业" if self.opening_hours == "全天" else f"营业时间 {self.opening_hours}")
            if self.recommended_food:
                parts.append("推荐菜：" + "、".join(self.recommended_food.split(",")[:5]))
        else:
            parts = ["住宿"]
            if self.type:
                parts.append(f"特色：{self.type}")
            if self.price is not None:
                parts.append(f"参考价 {self.price:g} 元/晚")
        return "，".join(parts) + "。"

    def attributes(self) -> dict[str, object]:
        values: dict[str, object] = {"type": self.type, "rank": self.rank}
        if self.latitude is not None and self.longitude is not None:
            values.update(latitude=self.latitude, longitude=self.longitude)
        if self.opening_hours:
            values["opening_hours"] = self.opening_hours
        if self.max_hours:
            values["duration"] = f"约 {self.max_hours:g} 小时"
        if self.recommended_food:
            values["recommended_food"] = self.recommended_food
        return values


def _number(value: str) -> float | None:
    try:
        return float(value) if value.strip() else None
    except ValueError:
        return None


def normalize_city(city: str) -> str:
    city = city.strip()
    return city[:-1] if len(city) > 2 and city.endswith("市") else city


class PlaceCatalog:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or PLACES_DIR
        self._records: dict[tuple[str, PlaceCategory], list[PlaceRecord]] | None = None

    def _load(self) -> dict[tuple[str, PlaceCategory], list[PlaceRecord]]:
        if self._records is None:
            records: dict[tuple[str, PlaceCategory], list[PlaceRecord]] = {}
            for category, filename in _FILES.items():
                path = self.directory / filename
                if not path.exists():
                    continue
                with path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        record = PlaceRecord(
                            city=row["city"],
                            category=category,
                            rank=int(row["rank"]),
                            name=row["name"],
                            type=(row.get("type") or "").strip(),
                            latitude=_number(row.get("latitude", "")),
                            longitude=_number(row.get("longitude", "")),
                            open_time=row.get("open_time", ""),
                            close_time=row.get("close_time", ""),
                            price=_number(row.get("price", "")),
                            max_hours=_number(row.get("max_hours", "")),
                            recommended_food=(row.get("recommended_food") or "").strip(),
                        )
                        records.setdefault((record.city, category), []).append(record)
            for items in records.values():
                items.sort(key=lambda record: record.rank)
            self._records = records
        return self._records

    def places(self, city: str, category: PlaceCategory) -> list[PlaceRecord]:
        """某城市某一类的全部地点，按热门程度排序；没有覆盖时返回空列表。"""
        return list(self._load().get((normalize_city(city), category), []))

    def find(self, city: str, name: str) -> PlaceRecord | None:
        for category in _FILES:
            for record in self.places(city, category):
                if record.name == name:
                    return record
        return None

    def shortlist(self, requirement: TravelRequirement, category: PlaceCategory) -> list[PlaceRecord]:
        """交给排日程挑选的候选：热门的在前，命中用户偏好的优先。"""
        records = self.places(requirement.destination, category)
        if not records:
            return []
        if category == "hotel":
            return records[:HOTEL_SHORTLIST]
        limit = min(max(requirement.days * 3, MIN_SHORTLIST), MAX_SHORTLIST)
        if category == "food":
            records = [record for record in records if _suits_dinner(record, requirement)]
            keywords = [*requirement.food_preferences, *requirement.styles]
        else:
            # 行程只排上午和下午两个景点时段，夜游类项目排不进去。
            records = [record for record in records if "夜游" not in record.name]
            keywords = list(requirement.styles)
        matched = [record for record in records if _matches(record, keywords)]
        # 偏好命中的最多占一半，其余仍按热门程度取，避免"喜欢历史"就排满冷门古迹。
        preferred = matched[: limit // 2]
        chosen = {record.id for record in preferred}
        rest = [record for record in records if record.id not in chosen][: limit - len(preferred)]
        return preferred + rest


def _matches(record: PlaceRecord, keywords: list[str]) -> bool:
    text = f"{record.name} {record.type} {record.recommended_food}"
    return any(keyword and (keyword in text or _STYLE_ALIASES.get(keyword, "\0") in text) for keyword in keywords)


def _suits_dinner(record: PlaceRecord, requirement: TravelRequirement) -> bool:
    if record.type in _NOT_A_MEAL:
        return False
    if any("不吃辣" in item or "不能吃辣" in item for item in requirement.food_preferences) and record.type in _SPICY:
        return False
    # 晚餐时段要开着：19 点以后才关门，或者营业到第二天凌晨。
    if record.open_time and record.close_time and record.open_time < record.close_time < "19:00":
        return False
    return True


def straight_line_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    """两个坐标（纬度, 经度）之间的直线距离，单位公里。"""
    lat1, lon1, lat2, lon2 = map(math.radians, (*first, *second))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def get_place_catalog() -> PlaceCatalog:
    return PlaceCatalog()


__all__ = [
    "PlaceCatalog",
    "PlaceCategory",
    "PlaceRecord",
    "SOURCE_NAME",
    "get_place_catalog",
    "normalize_city",
    "straight_line_km",
]
