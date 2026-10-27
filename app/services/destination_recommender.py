"""规划表单里"还没想好去哪"：只在地点库覆盖的城市里推荐目的地。

原先这里走通用知识库问答，而知识库只有成都、西安两份模拟资料，用户要的是
"推荐几个城市"，得到的却是"推荐成都的大熊猫基地、宽窄巷子"。现在候选城市
限定为地点库覆盖的城市（推荐出去的城市都能排出有真实地点的行程），每个城市
附上地点库里最热门的几个景点作为依据，模型只能在这个范围里挑、照这些景点讲理由。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.places.catalog import get_place_catalog
from app.utils.logger import app_logger

_HIGHLIGHTS_PER_CITY = 6
_MAX_PICKS = 3


class CityPick(BaseModel):
    city: str = Field(description="必须是候选城市之一")
    reason: str = Field(description="一两句推荐理由，结合用户条件")
    highlights: list[str] = Field(default_factory=list, description="从该城市的热门景点里挑 2~3 个")


class _Picks(BaseModel):
    cities: list[CityPick] = Field(default_factory=list)


class DestinationRecommendation(BaseModel):
    answer: str
    cities: list[str] = Field(default_factory=list)


def city_profiles() -> dict[str, list[str]]:
    """地点库覆盖的城市 → 最热门的几个景点名。"""
    catalog = get_place_catalog()
    return {
        city: [record.name for record in catalog.places(city, "attractions")[:_HIGHLIGHTS_PER_CITY]]
        for city in catalog.cities()
    }


def _conditions(partial_values: dict[str, Any], context: dict[str, Any]) -> str:
    labels = {"departure_date": "出发", "days": "天数", "adults": "成人", "children": "儿童",
              "styles": "喜欢", "food_preferences": "饮食", "origin": "出发地"}
    parts = []
    for key, label in labels.items():
        value = partial_values.get(key, context.get(key))
        if value in (None, "", []):
            continue
        parts.append(f"{label} {'、'.join(value) if isinstance(value, list) else value}")
    return "，".join(parts)


def _render(picks: list[CityPick], conditions: str) -> str:
    head = f"根据你的条件（{conditions}），推荐这几个城市：" if conditions else "推荐这几个城市："
    lines = [head, ""]
    for pick in picks:
        lines.append(f"**{pick.city}**：{pick.reason}")
        if pick.highlights:
            lines.append(f"- 可以去：{'、'.join(pick.highlights)}")
        lines.append("")
    lines.append("选一个城市就可以开始规划。")
    return "\n".join(lines)


def _sanitize(picks: list[CityPick], profiles: dict[str, list[str]]) -> list[CityPick]:
    """只保留候选城市；亮点只保留该城市热门景点里真有的。"""
    kept: list[CityPick] = []
    for pick in picks:
        if pick.city not in profiles or any(existing.city == pick.city for existing in kept):
            continue
        highlights = [name for name in pick.highlights if name in profiles[pick.city]]
        kept.append(pick.model_copy(update={"highlights": highlights or profiles[pick.city][:3]}))
    return kept[:_MAX_PICKS]


def _fallback(profiles: dict[str, list[str]]) -> list[CityPick]:
    return [
        CityPick(city=city, reason="热门旅行城市，地点资料完整。", highlights=highlights[:3])
        for city, highlights in list(profiles.items())[:_MAX_PICKS]
    ]


async def recommend_destinations(
    partial_values: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> DestinationRecommendation:
    context = context or {}
    profiles = city_profiles()
    conditions = _conditions(partial_values, context)
    picks: list[CityPick] = []
    if profiles and settings.llm_api_key:
        try:
            from app.core.llm import get_llm

            structured = get_llm(temperature=0.4).with_structured_output(_Picks)
            response = await structured.ainvoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是旅行目的地推荐助手，以 JSON 格式输出。只能从给定的候选城市里推荐 2~3 个，"
                            "按用户条件（天数、季节、同行人、兴趣）取舍；理由只能引用给出的热门景点和常识性的城市特点，"
                            "不要编造价格、开放时间或天气数据。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"用户条件：{conditions or '未说明'}\n"
                            f"候选城市及热门景点：{json.dumps(profiles, ensure_ascii=False)}"
                        ),
                    },
                ]
            )
            picks = _sanitize(_Picks.model_validate(response).cities, profiles)
        except Exception as exc:
            app_logger.warning(f"目的地推荐失败，改用默认推荐: {type(exc).__name__}: {exc}")
    picks = picks or _fallback(profiles)
    if not picks:
        return DestinationRecommendation(answer="暂时没有可推荐的城市，可以直接告诉我想去哪里。")
    return DestinationRecommendation(answer=_render(picks, conditions), cities=[pick.city for pick in picks])
