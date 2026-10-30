"""地点库：景点、餐厅、住宿的结构化数据（data/places/*.csv）。

子 Agent 在地点库覆盖的城市直接从这里取候选：名称、坐标、开放时间、价格原样带上，
不经过"检索文字 → 模型提取"。模型只负责在排日程时按用户偏好从候选里选。

地点库没有覆盖的城市返回空，子 Agent 照旧走检索流程。
"""

from __future__ import annotations

import re

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
MIN_SHORTLIST, MAX_SHORTLIST, HOTEL_SHORTLIST = 8, 40, 8
# 不适合当正餐的类型。
_NOT_A_MEAL = {"面包甜点", "咖啡店", "酒吧/酒馆", "茶馆/茶室", ""}
_SPICY = {"川菜", "湘菜", "火锅"}
# 名字里带这些词的地点只适合晚上去。
_NIGHT_WORDS = ("夜景", "夜游", "灯光秀", "不夜城", "夜市")
# 类型不辣、但从店名就能看出主打辣味的（"泰式海鲜火锅"的类型是东南亚菜）。
_SPICY_NAME_WORDS = ("火锅", "麻辣", "香辣", "串串", "冒菜", "钵钵鸡", "辣子")
# 源数据按热门程度排序，排在前面的多是酒店里的高档餐厅（上海前 30 家人均中位数 500 元）。
# 默认按城市人均价格的 75 分位设上限，再夹在 200~350 元之间（上海 75 分位就有 500 元）；
# 用户明确要高档时才放开。
_PRICE_PERCENTILE = 0.75
_PRICE_CAP_RANGE = (200.0, 350.0)
_UPSCALE_WORDS = ("高档", "高端", "豪华", "奢华", "米其林", "黑珍珠", "不差钱", "预算充足", "人均不限")
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

    def cities(self) -> list[str]:
        """地点库覆盖的城市，按景点数量从多到少。"""
        counts: dict[str, int] = {}
        for (city, category), records in self._load().items():
            if category == "attractions":
                counts[city] = len(records)
        return sorted(counts, key=lambda city: -counts[city])

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
            return _affordable_hotels(records, requirement)[:HOTEL_SHORTLIST]
        limit = min(max(requirement.days * 3, MIN_SHORTLIST), MAX_SHORTLIST)
        night_extra: list[PlaceRecord] = []
        if category == "food":
            price_cap = None if _wants_upscale(requirement) else _price_cap(records)
            budget_cap = _budget_share(requirement, _MEAL_SHARE) / requirement.days if requirement.budget else None
            if budget_cap is not None:
                price_cap = budget_cap if price_cap is None else min(price_cap, budget_cap)
            records = [
                record
                for record in records
                if _suits_dinner(record, requirement)
                and (price_cap is None or record.price is None or record.price <= price_cap)
            ]
            keywords = [*requirement.food_preferences, *requirement.styles]
        else:
            # 夜景、夜游类地点排程时只会放进晚饭后的夜间时段：不占白天景点的名额，
            # 用户想看夜景时另外追加几个，没说时不要。
            night = [record for record in records if any(word in record.name for word in _NIGHT_WORDS)]
            records = [record for record in records if record not in night]
            if any(word in "".join(requirement.styles) for word in ("夜景", "夜游", "夜市", "夜生活", "灯光")):
                night_extra = night[: requirement.days + 1]
            keywords = list(requirement.styles)
        # 用户点名要去的放最前面，不占名额；夜景点也照样给（排程会放进夜间时段）。
        named = [record for record in self.places(requirement.destination, category)
                 if named_by(record.name, requirement.must_visit)]
        records = [record for record in records if record not in named]
        night_extra = [record for record in night_extra if record not in named]
        matched = [record for record in records if _matches(record, keywords)]
        # 偏好命中的最多占一半，其余仍按热门程度取，避免"喜欢历史"就排满冷门古迹。
        preferred = matched[: limit // 2]
        chosen = {record.id for record in preferred}
        rest = [record for record in records if record.id not in chosen][: limit - len(preferred)]
        picked = named + preferred + rest + night_extra
        if category == "food":
            picked += self._near_attractions(requirement, records, picked)
        return picked

    def _near_attractions(
        self, requirement: TravelRequirement, eligible: list[PlaceRecord], picked: list[PlaceRecord]
    ) -> list[PlaceRecord]:
        """每个候选景点附近补一家符合条件的餐厅。

        热门餐厅多在市中心；一天玩到郊区的海洋公园、迪士尼，晚饭只能跑回市中心，
        路上一个多小时。有预算时贵的被筛掉，候选更少、更集中，问题更明显。
        """
        chosen = {record.id for record in picked}
        extra: list[PlaceRecord] = []
        located = [record for record in eligible if record.latitude is not None and record.longitude is not None]
        for sight in self.shortlist(requirement, "attractions"):
            if sight.latitude is None or sight.longitude is None:
                continue
            here = (sight.latitude, sight.longitude)
            nearby = [
                (straight_line_km(here, (record.latitude, record.longitude)), record)
                for record in located
                if record.id not in chosen
            ]
            nearby = [item for item in nearby if item[0] <= _NEAR_SIGHT_KM]
            if nearby:
                _, record = min(nearby, key=lambda item: item[0])
                chosen.add(record.id)
                extra.append(record)
        return extra


_PARENTHESES = re.compile(r"[(（]([^)）]*)[)）]")


def named_by(name: str, terms: list[str]) -> bool:
    """地点名和用户说的叫法对得上："洪崖洞" ↔ "洪崖洞民俗风貌区"，"熊猫基地"也能对上全名里的片段。

    括号里是别名时也算（"秦始皇帝陵博物院(兵马俑)"），是分店名时不算（"四季民福烤鸭店(故宫店)"）。
    """
    base = name.split("(")[0].split("（")[0]
    aliases = [base, *(alias for alias in _PARENTHESES.findall(name) if alias and not alias.endswith("店"))]
    return any(
        term and len(term) >= 2 and any(term in alias or (alias and alias in term) for alias in aliases)
        for term in terms
    )


def _matches(record: PlaceRecord, keywords: list[str]) -> bool:
    text = f"{record.name} {record.type} {record.recommended_food}"
    return any(keyword and (keyword in text or _STYLE_ALIASES.get(keyword, "\0") in text) for keyword in keywords)


def _mostly_spicy(recommended_food: str) -> bool:
    """推荐菜里有两道以上、或一半以上带辣味词，就算主打辣。

    菜系写"小吃"的兔头店，推荐菜是"麻辣手撕烤兔、冷吃兔、兔腿、兔头、鸭头"，
    5 道里 2 道明确是辣的，其余也多是辣做法。
    """
    dishes = [dish for dish in recommended_food.split(",") if dish.strip()]
    spicy = [dish for dish in dishes if any(word in dish for word in (*_SPICY_NAME_WORDS, "冷吃", "辣"))]
    return len(spicy) >= 2 or (bool(dishes) and len(spicy) * 2 >= len(dishes))


# 给景点配附近餐厅时的直线距离上限（公里）。
_NEAR_SIGHT_KM = 5.0
# 有预算时给住宿、餐饮各留的份额（剩下给门票和机动）。每晚房费、每人每天餐费不超过这份额。
_HOTEL_SHARE, _MEAL_SHARE = 0.45, 0.3
_MIN_AFFORDABLE_HOTELS = 3


def _budget_share(requirement: TravelRequirement, share: float) -> float:
    """预算里这一块给每人（餐饮）的钱；住宿另算。"""
    people = max(1, requirement.adults + requirement.children)
    return (requirement.budget or 0) * share / people


def _affordable_hotels(records: list[PlaceRecord], requirement: TravelRequirement) -> list[PlaceRecord]:
    """有预算时只给住得起的酒店（每晚不超过预算的住宿份额）；一个都没有就给最便宜的几家。"""
    nights = requirement.days - 1
    if not requirement.budget or nights <= 0:
        return records
    cap = requirement.budget * _HOTEL_SHARE / nights
    affordable = [record for record in records if record.price is not None and record.price <= cap]
    if len(affordable) >= _MIN_AFFORDABLE_HOTELS:
        return affordable
    priced = sorted((record for record in records if record.price is not None), key=lambda record: record.price)
    return priced[:HOTEL_SHORTLIST]


def _price_cap(records: list[PlaceRecord]) -> float | None:
    prices = sorted(record.price for record in records if record.price)
    if len(prices) < 4:
        return None
    percentile = prices[min(len(prices) - 1, int(len(prices) * _PRICE_PERCENTILE))]
    low, high = _PRICE_CAP_RANGE
    return min(max(percentile, low), high)


def _wants_upscale(requirement: TravelRequirement) -> bool:
    text = " ".join([*requirement.styles, *requirement.food_preferences, *requirement.special_needs])
    return any(word in text for word in _UPSCALE_WORDS)


def _suits_dinner(record: PlaceRecord, requirement: TravelRequirement) -> bool:
    if record.type in _NOT_A_MEAL:
        return False
    if any("不吃辣" in item or "不能吃辣" in item for item in requirement.food_preferences) and (
        record.type in _SPICY
        or any(word in record.name for word in _SPICY_NAME_WORDS)
        or _mostly_spicy(record.recommended_food)
    ):
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
