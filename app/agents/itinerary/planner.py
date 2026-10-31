"""编排入口：选点 → 排程 → 检查，最多 MAX_ROUNDS 轮，最后补上住宿、天气和交通备注。"""

from __future__ import annotations

from typing import Any

from app.core.llm import describe_llm_failure
from app.schemas.planning import CandidateOption, ItineraryDay, TravelRequirement, WorkerResult
from app.utils.logger import app_logger
from app.agents.itinerary.rules import (
    FALLBACK_REASON_PREFIX,
    _hotel_note,
    HOTEL_NOTE_PREFIX,
    MAX_ROUNDS,
    _options,
    TRANSPORT_NOTE_PREFIX,
    Violation,
    WEATHER_NOTE_PREFIX,
)
from app.agents.itinerary.travel import TravelCosts, _trip_weather
from app.agents.itinerary.selection import closer_hotel, _sanitize, select_places_by_rules, select_places_with_llm
from app.agents.itinerary.scheduling import build_itinerary
from app.agents.itinerary.repair import (
    _budget_violations,
    check_itinerary,
    _DISTANCE_CODES,
    drop_outlying_sights,
    trim_to_budget,
)


def schedule_by_rules(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    costs: TravelCosts | None = None,
) -> tuple[list[ItineraryDay], list[str]]:
    """不用 LLM 的完整排程：规则选点 + 算法排程。没有任何候选时每个时段都是兜底文案。"""
    selection, warnings = _sanitize(select_places_by_rules(requirement, results), requirement, results)
    return build_itinerary(requirement, results, selection, costs or TravelCosts(results)), warnings


async def plan_itinerary(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    *,
    llm: Any | None = None,
) -> tuple[list[ItineraryDay], list[str]]:
    costs = TravelCosts(results)
    warnings: list[str] = []
    feedback: list[Violation] = []
    itinerary: list[ItineraryDay] = []
    hotel: CandidateOption | None = None
    for _round in range(MAX_ROUNDS):
        selection = None
        if llm is not None:
            try:
                selection = await select_places_with_llm(llm, requirement, results, feedback)
            except Exception as exc:
                app_logger.warning(f"LLM 选点失败，改用规则选点: {type(exc).__name__}: {exc}")
                warnings.append("itinerary_selection_fallback:llm_failed")
                warnings.append(f"{FALLBACK_REASON_PREFIX}{describe_llm_failure(exc)}")
                llm = None
        by_rules = selection is None
        if by_rules:
            itinerary, round_warnings = schedule_by_rules(requirement, results, costs)
            # 规则选点固定用第一个住宿候选（见 select_places_by_rules）。
            hotel = next(iter(_options(results, "hotel")), None)
        else:
            selection, round_warnings = _sanitize(selection, requirement, results)
            selection = selection.model_copy(update={"hotel_id": closer_hotel(selection, results, costs)})
            itinerary = build_itinerary(requirement, results, selection, costs)
            hotel = next((option for option in _options(results, "hotel") if option.id == selection.hotel_id), None)
        warnings.extend(round_warnings)
        feedback = check_itinerary(requirement, results, itinerary) + _budget_violations(requirement, itinerary, hotel)
        # 规则选点不会根据违规原因调整，重选也是同一个结果，只有 LLM 选点才值得再来一轮。
        if not feedback or by_rules:
            break
    if selection is not None and any(item.code in _DISTANCE_CODES for item in feedback):
        repaired = drop_outlying_sights(requirement, results, selection, itinerary, costs)
        if repaired is not None:
            repaired_feedback = check_itinerary(requirement, results, repaired) + _budget_violations(
                requirement, repaired, hotel
            )
            if len(repaired_feedback) < len(feedback):
                itinerary, feedback = repaired, repaired_feedback
    if any(item.code == "budget" for item in feedback):
        # 模型几轮都压不进预算时，由算法把最贵的饭和门票换成附近更便宜的。
        itinerary = trim_to_budget(requirement, results, itinerary, hotel, costs)
        feedback = [item for item in feedback if item.code != "budget"] + _budget_violations(requirement, itinerary, hotel)
    warnings.extend(f"itinerary_constraint:{item.code}:{item.message}" for item in feedback)
    if hotel is not None and itinerary:
        itinerary[0].notes.insert(0, f"{HOTEL_NOTE_PREFIX}{hotel.name}。{_hotel_note(hotel)}")
    for day, text in _trip_weather(requirement, results).items():
        if day <= len(itinerary):
            itinerary[day - 1].notes.insert(0, f"{WEATHER_NOTE_PREFIX}{text}")
    # 城际交通不在规划范围内，也没有实时班次数据：只说明从哪来、回哪去，不编班次。
    if requirement.origin and itinerary:
        itinerary[0].notes.append(
            f"{TRANSPORT_NOTE_PREFIX}从{requirement.origin}出发，城际高铁或航班请在 12306 或航旅 App 查询实时班次。"
        )
        itinerary[-1].notes.append(f"{TRANSPORT_NOTE_PREFIX}行程结束后返回{requirement.origin}，返程班次同样需要实时查询。")
    return itinerary, list(dict.fromkeys(warnings))
