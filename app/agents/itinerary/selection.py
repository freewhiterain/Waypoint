"""1. 选点：模型只能按 id 从候选里挑；挑完由代码清理（去重、补上点名要去的、换近一点的酒店）。"""

from __future__ import annotations

from typing import Any
import json

from app.places.catalog import named_by
from app.schemas.planning import CandidateOption, TravelRequirement, WorkerResult
from app.utils.logger import app_logger
from app.agents.itinerary.rules import (
    ATTRACTIONS_PER_DAY,
    _candidates,
    day_capacities,
    HOTEL_DETOUR_MINUTES,
    _keyword_score,
    _options,
    PlaceSelection,
    _relaxed_hint,
    same_place,
    SelectedPlace,
    SELECTION_ATTEMPTS,
    Violation,
)
from app.agents.itinerary.travel import _summaries, TravelCosts, _trip_weather


def _option_digest(option: CandidateOption) -> dict[str, Any]:
    digest = {"id": option.id, "name": option.name, "description": option.description[:160]}
    if option.estimated_cost is not None:
        digest["estimated_cost"] = option.estimated_cost
    for key in ("type", "location", "district", "opening_hours"):
        if option.attributes.get(key):
            digest[key] = option.attributes[key]
    latitude, longitude = option.attributes.get("latitude"), option.attributes.get("longitude")
    if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
        digest["position"] = [round(latitude, 3), round(longitude, 3)]
    return digest


async def select_places_with_llm(
    llm: Any,
    requirement: TravelRequirement,
    results: list[WorkerResult],
    feedback: list[Violation],
) -> PlaceSelection:
    candidates = {
        "attractions": [_option_digest(option) for option in _candidates(results, "attractions", requirement)],
        "dinners": [_option_digest(option) for option in _candidates(results, "food", requirement)],
        "hotels": [_option_digest(option) for option in _options(results, "hotel")],
        "weather": [f"第{day}天：{text}" for day, text in _trip_weather(requirement, results).items()]
        or _summaries(results, "weather"),
        "transport": [option.description for option in _options(results, "transport")][:30],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是行程选点助手，以 JSON 格式输出。只能从给定候选里按 id 选择，不得新增地点。"
                f"每天最多 {ATTRACTIONS_PER_DAY} 个景点、1 顿晚餐；景点总数不超过 {sum(day_capacities(requirement))} 个。"
                + _relaxed_hint(requirement)
                + "按用户偏好和人数取舍；天气显示下雨的日期，室内景点可用 preferred_day 指定到那天。"
                "选一个落脚酒店（hotel_id），尽量靠近所选景点。顺序和分天由后续算法决定，不需要你排。"
                "候选里的 position 是 [纬度, 经度]，可以用来判断远近：晚餐选在所选景点附近的，"
                "离其他景点都很远的景点（比如在几十公里外的郊县）一次行程最多选一个；"
                "同一处地方的不同叫法（如灵隐寺和灵隐飞来峰景区）只选一个。"
                "迪士尼、环球影城、欢乐谷、长隆这类主题乐园和大型动物园一玩就是一整天，"
                "离市区几十公里的郊县景点（都江堰、八达岭长城这类）同样占一整天；"
                "选了它当天就不会再排别的景点，景点总数要相应少选。"
                "需求里 must_visit 是用户点名要去的地点，候选里有就必须选上。"
                "需求里 budget 是整趟所有人的总预算（元）：门票和餐饮的 estimated_cost 按人头算、"
                "酒店的 estimated_cost 是每晚房费，有预算时优先选便宜的和免费景点，总价不要超。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"需求：{requirement.model_dump_json()}\n"
                f"候选：{json.dumps(candidates, ensure_ascii=False)}"
                + (
                    "\n上一版方案未通过检查，请调整：\n" + "\n".join(f"- {item.message}" for item in feedback)
                    if feedback
                    else ""
                )
            ),
        },
    ]
    structured = llm.with_structured_output(PlaceSelection)
    last_error: Exception | None = None
    for attempt in range(SELECTION_ATTEMPTS):
        try:
            return PlaceSelection.model_validate(await structured.ainvoke(messages))
        except Exception as exc:
            last_error = exc
            app_logger.warning(f"LLM 选点第 {attempt + 1} 次未返回有效结果: {type(exc).__name__}")
    raise last_error  # type: ignore[misc]


def select_places_by_rules(
    requirement: TravelRequirement,
    results: list[WorkerResult],
) -> PlaceSelection:
    """没有 LLM 时的选点：偏好命中多的优先，其余保持子 Agent 给出的顺序。"""
    attractions = sorted(
        _candidates(results, "attractions", requirement), key=lambda option: -_keyword_score(option, requirement)
    )
    dinners = sorted(_candidates(results, "food", requirement), key=lambda option: -_keyword_score(option, requirement))
    hotels = _options(results, "hotel")
    limit = sum(day_capacities(requirement))
    return PlaceSelection(
        hotel_id=hotels[0].id if hotels else None,
        attractions=[SelectedPlace(id=option.id) for option in attractions[:limit]],
        dinners=[SelectedPlace(id=option.id) for option in dinners[: requirement.days]],
    )


def closer_hotel(selection: PlaceSelection, results: list[WorkerResult], costs: TravelCosts) -> str | None:
    """模型选的酒店离当天要去的地方明显更远时，换成平均路程最短的候选。

    有预算后便宜的酒店多在郊区（上海川沙、虹桥），模型挑了便宜的，
    每天早上先开一小时车进城。候选已按预算筛过，换哪家都住得起。
    """
    hotels = _options(results, "hotel")
    sights = [item.id for item in selection.attractions]
    if not hotels or not sights:
        return selection.hotel_id

    def mean_cost(hotel_id: str) -> float:
        return sum(costs.cost(hotel_id, sight) for sight in sights) / len(sights)

    best = min(hotels, key=lambda option: mean_cost(option.id)).id
    if selection.hotel_id is None or mean_cost(selection.hotel_id) - mean_cost(best) > HOTEL_DETOUR_MINUTES:
        return best
    return selection.hotel_id


def _must_visit_options(results: list[WorkerResult], worker: str, requirement: TravelRequirement) -> list[CandidateOption]:
    """候选里对得上用户点名要去的地点，每个叫法只取最热门的一个。"""
    found: list[CandidateOption] = []
    for term in requirement.must_visit:
        match = next((option for option in _candidates(results, worker, requirement)
                      if named_by(option.name, [term])), None)
        if match is not None and match not in found:
            found.append(match)
    return found


def _with_must_visit(
    items: list[SelectedPlace], options: list[CandidateOption], names: dict[str, str], limit: int, city: str
) -> list[SelectedPlace]:
    """模型漏选了用户点名的地点时补进去，挤掉排在最后的；已选了同一处的别名就不重复补。"""
    picked = [names.get(item.id, "") for item in items]
    missing = [
        SelectedPlace(id=option.id)
        for option in options
        if not any(same_place(option.name, name, city) for name in picked)
    ]
    if not missing:
        return items
    named = {option.id for option in options}
    return ([item for item in items if item.id in named] + missing + [item for item in items if item.id not in named])[:limit]


def _sanitize(selection: PlaceSelection, requirement: TravelRequirement, results: list[WorkerResult]) -> tuple[PlaceSelection, list[str]]:
    """丢掉不存在或重复的 id（含同一处地方的不同叫法）、超出容量的景点和越界的 preferred_day。"""
    warnings: list[str] = []
    valid = {
        "attractions": {option.id for option in _candidates(results, "attractions", requirement)},
        "dinners": {option.id for option in _candidates(results, "food", requirement)},
    }
    names = {option.id: option.name for worker in ("attractions", "food") for option in _options(results, worker)}

    def clean(items: list[SelectedPlace], kind: str, limit: int) -> list[SelectedPlace]:
        kept: list[SelectedPlace] = []
        for item in items:
            duplicate = any(
                existing.id == item.id
                or same_place(names.get(existing.id, ""), names.get(item.id, ""), requirement.destination)
                for existing in kept
            )
            if item.id not in valid[kind] or duplicate:
                warnings.append(f"itinerary_selection_dropped:{kind}")
                continue
            if item.preferred_day is not None and item.preferred_day > requirement.days:
                item = item.model_copy(update={"preferred_day": None})
            kept.append(item)
        return kept[:limit]

    hotel_ids = {option.id for option in _options(results, "hotel")}
    attraction_limit = sum(day_capacities(requirement))
    return (
        PlaceSelection(
            hotel_id=selection.hotel_id if selection.hotel_id in hotel_ids else None,
            attractions=_with_must_visit(
                clean(selection.attractions, "attractions", attraction_limit),
                _must_visit_options(results, "attractions", requirement),
                names, attraction_limit, requirement.destination,
            ),
            dinners=_with_must_visit(
                clean(selection.dinners, "dinners", requirement.days),
                _must_visit_options(results, "food", requirement),
                names, requirement.days, requirement.destination,
            ),
        ),
        warnings,
    )
