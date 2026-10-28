"""行程编排：LLM 选点 → 算法排程 → 约束检查，不通过就带着违规原因让 LLM 重选。

分工参考 ItiNera（LLM 理解偏好、算法保证空间顺路）和 LLM-Modulo（外部检查器
把关、违规反馈给 LLM）：

1. 选点（LLM）：按偏好和天气从已通过证据治理的候选里挑景点、晚餐和落脚酒店。
   只能引用候选 id，引用不存在的 id 直接丢弃；没有 LLM 或调用失败时按规则挑选。
2. 排程（算法）：以酒店为中心，按交通耗时把景点两两成组分到每天，再在每天内部
   选最省路的先后顺序；晚餐选离当天最后一站最近的餐厅。
3. 检查（代码）：开放时间、每天交通总耗时。违规时把原因交回第 1 步，最多重试
   MAX_ROUNDS 次，仍不通过就采用最后一版并把违规写进 warnings。规则选点不会根据
   反馈调整，只排一轮。

交通耗时来自交通子 Agent 的 travel_leg 候选。没有真实分钟数时（只有区级粗估），
排程只用一个内部代价来比较远近，展示给用户的耗时保持为空，不编造数字。
"""

from __future__ import annotations

import json
import re
from itertools import permutations
from typing import Any

from pydantic import BaseModel, Field

from app.core.llm import describe_llm_failure
from app.places.catalog import named_by, straight_line_km
from app.schemas.planning import (
    CandidateOption,
    ItineraryDay,
    TimeSlot,
    TravelRequirement,
    WorkerResult,
)
from app.utils.logger import app_logger

MAX_ROUNDS = 3
# 强制工具调用下模型偶尔返回空结果（解析为 None），先重试，仍失败才退回规则选点。
SELECTION_ATTEMPTS = 2
ATTRACTIONS_PER_DAY = 2
MAX_DAILY_TRAVEL_MINUTES = 180
# 同一天两个景点之间最多能接受的路程（分钟）。
MAX_HOP_MINUTES = 75
# 没有真实耗时时用于比较远近的内部代价（分钟量级），不会展示给用户。
_PROXIMITY_COST = {"same_district": 20, "unknown": 35, "different_district": 50}
_OPENING_PATTERN = re.compile(r"(\d{1,2}):(\d{2})\s*[-~至到]\s*(\d{1,2}):(\d{2})")
# 时段与开放时间的最低要求：上午 11 点前要开门，下午 15 点后才关门。
# 景点时段对开放时间的要求：(最晚几点开门, 最早几点关门)。下午 15 点才开门还算下午，再晚就是傍晚了。
# 评测的 plan_checks 用同一份标准。
# 选点退回规则时附带的原因（余额不足、连不上……），渲染时并进"智能选点暂时不可用"的提示。
FALLBACK_REASON_PREFIX = "itinerary_selection_fallback_reason:"
SIGHT_PERIOD_WINDOWS = {"morning": (11 * 60, 11 * 60 + 30), "afternoon": (15 * 60, 15 * 60)}
# 用户表达了对吃的兴趣时，每天在上午景点附近加一顿午餐（默认只排晚餐）。
# food_preferences 里不能用单字"吃"匹配：常见的"不吃辣"也会命中。兴趣（styles）里没有这种否定说法，
# 而且模型常把"想吃火锅"记成"火锅"而不是"美食"，所以兴趣里带"吃"或具体吃食也算。
_FOOD_INTEREST_WORDS = ("美食", "小吃", "吃货", "好吃", "餐厅", "馆子", "探店", "夜宵", "地道菜", "特色菜")
_FOOD_STYLE_WORDS = ("吃", "火锅", "烧烤", "海鲜", "烤鸭", "早茶", "串串", "面馆")
# 午餐要在上午景点附近：超过这个代价（分钟量级）宁可不排。
MAX_LUNCH_MINUTES = 30
# 晚饭后去看夜景，路上最多能接受的代价。
MAX_NIGHT_MINUTES = 45
# 午餐时段要开着：11:30 前开门、13:30 后才关门。
_LUNCH_WINDOW = (11 * 60 + 30, 13 * 60 + 30)
# 晚饭要在这段时间营业：只做午市、下午四点半就打烊的自助餐不能排成晚饭。
_DINNER_WINDOW = (18 * 60, 19 * 60 + 30)


class SelectedPlace(BaseModel):
    id: str
    reason: str = Field(default="", description="选它的理由，不超过 15 个字")
    preferred_day: int | None = Field(default=None, ge=1)


class PlaceSelection(BaseModel):
    """LLM 选点的结构化输出：只能引用候选 id。"""

    hotel_id: str | None = None
    attractions: list[SelectedPlace] = Field(default_factory=list)
    dinners: list[SelectedPlace] = Field(default_factory=list)
    # 原先还有一个自由文本的 notes 字段：DeepSeek 常把它写成没加引号的一段中文，
    # 整个工具调用 JSON 解析失败，约三分之一的选点因此退回规则。结果里也没用到它。


class Violation(BaseModel):
    code: str
    message: str


def _options(results: list[WorkerResult], worker: str) -> list[CandidateOption]:
    return [option for result in results if result.worker == worker for option in result.options]


def _avoided(option: CandidateOption, requirement: TravelRequirement) -> bool:
    """用户说过不想去的地点（"换掉熊猫基地"里的"熊猫基地"也能对上"成都大熊猫繁育研究基地"）。"""
    return any(term and (term in option.name or option.name in term) for term in requirement.avoid_places)


_NIGHT_WORDS = ("夜景", "夜游", "灯光秀", "不夜城", "夜市")
_NIGHT_INTEREST_WORDS = ("夜景", "夜游", "夜市", "夜生活", "灯光")


def is_night_spot(option: CandidateOption) -> bool:
    """名字里带"夜景/夜游/灯光秀"的地点只适合晚上去，不能排进上午和下午。"""
    return any(word in option.name for word in _NIGHT_WORDS)


def wants_night(requirement: TravelRequirement) -> bool:
    return any(word in "".join(requirement.styles) for word in _NIGHT_INTEREST_WORDS)


def _candidates(results: list[WorkerResult], worker: str, requirement: TravelRequirement) -> list[CandidateOption]:
    """可供选点的候选：去掉用户明确不想去的；景点里去掉只适合晚上去的（另由夜间时段安排）。"""
    return [
        option
        for option in _options(results, worker)
        if not _avoided(option, requirement) and not (worker == "attractions" and is_night_spot(option))
    ]


def _relaxed_hint(requirement: TravelRequirement) -> str:
    if _wants_relaxed_trip(requirement):
        return "用户希望整趟节奏轻松，每天只排 1 个景点，选路面平缓、不太累的。"
    days = [day for day in sorted(set(requirement.relaxed_days)) if day <= requirement.days]
    if not days:
        return ""
    return "用户要求" + "、".join(f"第{day}天" for day in days) + "轻松些，这几天只排 1 个景点，选不太累的。"


# 用户说"别太累""带老人""慢节奏"但没指定哪天时，模型会把它记成风格标签而不是 relaxed_days。
_RELAXED_STYLE_WORDS = ("轻松", "悠闲", "休闲", "慢", "不累", "别太累", "老人", "腿脚")


def _wants_relaxed_trip(requirement: TravelRequirement) -> bool:
    text = "".join([*requirement.styles, *requirement.special_needs])
    return any(word in text for word in _RELAXED_STYLE_WORDS)


def day_capacities(requirement: TravelRequirement) -> list[int]:
    """每天最多几个景点：用户要求轻松的那几天只排 1 个；整趟要轻松时每天都只排 1 个。"""
    if _wants_relaxed_trip(requirement):
        return [1] * requirement.days
    relaxed = set(requirement.relaxed_days)
    return [1 if day in relaxed else ATTRACTIONS_PER_DAY for day in range(1, requirement.days + 1)]


WEATHER_NOTE_PREFIX = "天气："
HOTEL_NOTE_PREFIX = "住宿："
TRANSPORT_NOTE_PREFIX = "交通："
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


# ---------------------------------------------------------------- 1. 选点


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
                "需求里 must_visit 是用户点名要去的地点，候选里有就必须选上。"
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


def _keyword_score(option: CandidateOption, requirement: TravelRequirement) -> int:
    text = f"{option.name}{option.description}"
    keywords = [*requirement.styles, *requirement.food_preferences]
    return sum(1 for keyword in keywords if keyword and keyword in text)


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


_GENERIC_SUFFIXES = ("风景名胜区", "风景区", "旅游区", "景区", "公园", "博物院", "博物馆", "寺", "古镇")


def _base_name(name: str) -> str:
    """去掉括号里的分店/备注："楼外楼(西湖景区孤山路店)" → "楼外楼"。"""
    return name.split("(")[0].split("（")[0].strip()


def _core_name(name: str, city: str = "") -> str:
    """去掉括号、城市名前缀和"景区/寺/公园"这类通用后缀后的名字主干："灵隐寺" → "灵隐"。

    城市名要去掉，否则"南京博物院"和"南京总统府"的主干都是"南京"，会被当成同一处。
    """
    core = _base_name(name)
    if city and core.startswith(city):
        core = core[len(city):]
    for suffix in _GENERIC_SUFFIXES:
        if core.endswith(suffix) and len(core) - len(suffix) >= 2:
            return core[: -len(suffix)]
    return core


def same_place(first: str, second: str, city: str = "") -> bool:
    """同一处地方的不同叫法：灵隐寺 / 灵隐飞来峰景区，故宫 / 故宫博物院。

    括号里的分店名不参与比较，否则"四季民福烤鸭店(故宫店)"会被当成故宫。
    """
    if not first or not second:
        return False
    if first == second:
        return True
    a, b = _core_name(first, city), _core_name(second, city)
    return len(a) >= 2 and len(b) >= 2 and (a in _base_name(second) or b in _base_name(first))


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


# ---------------------------------------------------------------- 2. 排程


def assign_days(
    attraction_ids: list[str],
    preferred: dict[str, int],
    days: int,
    hotel_id: str | None,
    costs: TravelCosts,
    capacities: list[int] | None = None,
) -> list[list[str]]:
    """把景点分到每天（每天最多 capacities[i] 个，默认 ATTRACTIONS_PER_DAY），并排好每天内部顺序。

    先放用户/LLM 指定了日期的景点；剩下的按"离酒店最近的作为当天起点，再补离它
    最近的"贪心成组，让同一天的景点彼此靠近。
    """
    capacities = capacities or [ATTRACTIONS_PER_DAY] * days
    plan: list[list[str]] = [[] for _ in range(days)]
    remaining = list(attraction_ids)
    for place_id in list(remaining):
        day = preferred.get(place_id)
        if day is not None and len(plan[day - 1]) < capacities[day - 1]:
            plan[day - 1].append(place_id)
            remaining.remove(place_id)

    for day_places, capacity in zip(plan, capacities):
        while remaining and len(day_places) < capacity:
            anchor = day_places[-1] if day_places else hotel_id
            nearest = min(remaining, key=lambda place_id: (costs.cost(anchor, place_id), remaining.index(place_id)))
            # 当天已经有景点时，离它太远的不硬凑进同一天：这个时段宁可空着，远的留给后面的日子。
            if day_places and costs.cost(anchor, nearest) > MAX_HOP_MINUTES:
                break
            day_places.append(nearest)
            remaining.remove(nearest)

    # 贪心到最后一天时，被"太远"挡下的景点没有后面的日子可去，原先会被悄悄丢掉，
    # 行程里却出现"没有第二个景点候选"的空时段。这里把它们补进还有空位、
    # 增加路程最少的那一天；真的太远时由约束检查报出来，交给 LLM 重选。
    for place_id in list(remaining):
        open_days = [day_places for day_places, capacity in zip(plan, capacities) if len(day_places) < capacity]
        if not open_days:
            break
        target = min(
            open_days,
            key=lambda day_places: costs.cost(day_places[-1] if day_places else hotel_id, place_id),
        )
        target.append(place_id)
        remaining.remove(place_id)

    return [_best_order(day_places, hotel_id, costs) for day_places in plan]


def top_up_days(
    day_plans: list[list[str]],
    candidate_ids: list[str],
    hotel_id: str | None,
    costs: TravelCosts,
    capacities: list[int] | None = None,
    names: dict[str, str] | None = None,
    city: str = "",
) -> list[list[str]]:
    """选点不够时，从没选中的候选里给空时段补一个顺路的景点。

    LLM 常常只挑出 天数×2 少一两个的景点，原先空出来的半天会写成"没有第二个
    带证据的景点候选"，可候选其实还有几十个。候选顺序已经是"命中偏好的在前、
    其余按热门程度"，这里在离当天已有景点（没有时离酒店）不超过 MAX_HOP_MINUTES
    的候选里取最近的；附近确实没有时才让时段空着。
    """
    capacities = capacities or [ATTRACTIONS_PER_DAY] * len(day_plans)
    used = {place_id for day in day_plans for place_id in day}
    used_names = [names[place_id] for place_id in used if place_id in names] if names else []
    filled: list[list[str]] = []
    for day_places, capacity in zip(day_plans, capacities):
        day_places = list(day_places)
        while len(day_places) < capacity:
            anchor = day_places[-1] if day_places else hotel_id
            pool = [
                place_id
                for place_id in candidate_ids
                if place_id not in used and not any(same_place(names.get(place_id, ""), name, city) for name in used_names)
            ] if names else [place_id for place_id in candidate_ids if place_id not in used]
            if not pool:
                break
            nearest = min(pool, key=lambda place_id: (costs.cost(anchor, place_id), pool.index(place_id)))
            if costs.cost(anchor, nearest) > MAX_HOP_MINUTES:
                break
            day_places.append(nearest)
            used.add(nearest)
            if names and nearest in names:
                used_names.append(names[nearest])
        filled.append(_best_order(day_places, hotel_id, costs))
    return filled


def _best_order(place_ids: list[str], hotel_id: str | None, costs: TravelCosts) -> list[str]:
    """每天只有两三个点，直接枚举所有顺序取总路程最短的。"""
    if len(place_ids) < 2:
        return list(place_ids)

    def route_cost(order: tuple[str, ...]) -> int:
        stops = [hotel_id, *order]
        return sum(costs.cost(a, b) for a, b in zip(stops, stops[1:]))

    return list(min(permutations(place_ids), key=route_cost))


def wants_lunch(requirement: TravelRequirement) -> bool:
    styles = "".join(requirement.styles)
    if any(word in styles for word in (*_FOOD_INTEREST_WORDS, *_FOOD_STYLE_WORDS)):
        return True
    # 模型有时把"想吃火锅"记进口味而不是兴趣；"不吃辣""不要火锅"这类忌口不算想吃。
    # 单个"吃"字在口味里太宽（"不吃辣"也有），这里只认具体吃食。
    wanted = "".join(item for item in requirement.food_preferences if not item.startswith(("不", "忌", "别", "少")))
    return any(word in wanted for word in (*_FOOD_INTEREST_WORDS, *_FOOD_STYLE_WORDS[1:]))


def _open_during(option: CandidateOption, needed: tuple[int, int]) -> bool:
    """餐厅在这段饭点是否营业；没有营业时间的一律算营业。"""
    hours = str(option.attributes.get("opening_hours") or "")
    if hours == "全天":
        return True
    window = _opening_minutes(hours)
    return window is None or (window[0] <= needed[0] and window[1] >= needed[1])


def _open_for_lunch(option: CandidateOption) -> bool:
    return _open_during(option, _LUNCH_WINDOW)


def _pick_nearby(pool: list[str], used: set[str], near: str, costs: TravelCosts, max_minutes: int) -> str | None:
    """午餐、夜景这类"顺路加一站"的时段：在离 near 不超过 max_minutes 的候选里取最近的。"""
    candidates = [place_id for place_id in pool if place_id not in used]
    if not candidates:
        return None
    best = min(candidates, key=lambda place_id: (costs.cost(near, place_id), candidates.index(place_id)))
    return best if costs.cost(near, best) <= max_minutes else None


def build_itinerary(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    selection: PlaceSelection,
    costs: TravelCosts,
) -> list[ItineraryDay]:
    options = {option.id: option for worker in ("attractions", "food") for option in _options(results, worker)}
    preferred = {item.id: item.preferred_day for item in selection.attractions if item.preferred_day}
    capacities = day_capacities(requirement)
    day_plans = assign_days(
        [item.id for item in selection.attractions], preferred, requirement.days, selection.hotel_id, costs, capacities
    )
    day_plans = top_up_days(
        day_plans,
        [option.id for option in _candidates(results, "attractions", requirement)],
        selection.hotel_id,
        costs,
        capacities,
        {option.id: option.name for option in _options(results, "attractions")},
        requirement.destination,
    )
    open_for_dinner = [
        option.id for option in _candidates(results, "food", requirement) if _open_during(option, _DINNER_WINDOW)
    ]
    dinner_pool = [item.id for item in selection.dinners if item.id in open_for_dinner]
    backup_dinners = [place_id for place_id in open_for_dinner if place_id not in dinner_pool]
    used_dinners: set[str] = set()
    night_options = {
        option.id: option
        for option in _options(results, "attractions")
        if is_night_spot(option) and not _avoided(option, requirement)
    }
    night_pool = list(night_options) if wants_night(requirement) else []
    used_nights: set[str] = set()
    # 午餐从没被选作晚餐的餐饮候选里就近挑，不交给 LLM：它只关心"离上午景点近、中午开门"。
    lunch_pool = (
        [
            option.id
            for option in _candidates(results, "food", requirement)
            if option.id not in dinner_pool and _open_for_lunch(option)
        ]
        if wants_lunch(requirement)
        else []
    )

    itinerary: list[ItineraryDay] = []
    for index, place_ids in enumerate(day_plans):
        slots: list[TimeSlot] = []
        previous = selection.hotel_id
        periods = _assign_periods([options[place_id] for place_id in place_ids])
        for period in ("morning", "afternoon"):
            option = periods.get(period)
            if period == "afternoon" and previous is not None and periods.get("morning") is not None:
                lunch_id = _pick_nearby(lunch_pool, used_dinners, previous, costs, MAX_LUNCH_MINUTES)
                if lunch_id is not None:
                    used_dinners.add(lunch_id)
                    slots.append(_slot("lunch", options[lunch_id], costs.minutes(previous, lunch_id)))
                    previous = lunch_id
            if option is not None:
                slots.append(_slot(period, option, costs.minutes(previous, option.id)))
                previous = option.id
            elif period == "morning" and not periods:
                slots.append(TimeSlot(period="morning", title=f"{requirement.destination}分区自由活动",
                                      description="该时段没有留下带证据的景点候选。"))
            elif period == "morning":
                slots.append(TimeSlot(period="morning", title="自由活动",
                                      description="当天的景点开门较晚，安排在下午；上午可以自由安排。"))
            elif capacities[index] < ATTRACTIONS_PER_DAY:
                slots.append(TimeSlot(period="afternoon", title="自由活动",
                                      description="按你的要求这天安排得轻松些，下午留给休息或在周边随意逛逛。"))
            else:
                slots.append(TimeSlot(period="afternoon", title="自由活动",
                                      description="附近没有能顺路安排的景点，下午留给休息或在周边随意逛逛。"))

        # 选中的晚餐用完了就从其余餐饮候选里就近补，不再让同一家餐厅连吃几晚。
        dinner_id = _pick_dinner(dinner_pool, used_dinners, previous, costs) or _pick_dinner(
            backup_dinners, used_dinners, previous, costs
        )
        if dinner_id is not None:
            used_dinners.add(dinner_id)
            slots.append(_slot("evening", options[dinner_id], costs.minutes(previous, dinner_id)))
            previous = dinner_id
        else:
            reason = "能查到的餐厅前几晚已经安排过，这晚可以自由选择。" if used_dinners else "没有带证据的餐饮候选可安排。"
            slots.append(TimeSlot(period="evening", title="自由安排晚餐", description=reason))

        # 想看夜景的，晚饭后就近安排一个夜景点。
        night_id = _pick_nearby(night_pool, used_nights, previous, costs, MAX_NIGHT_MINUTES) if previous else None
        if night_id is not None:
            used_nights.add(night_id)
            # "重庆夜游""夜游两江"其实是同一种活动，排过一个就把同名的都排除掉。
            used_nights.update(
                other
                for other, candidate in night_options.items()
                if same_place(candidate.name, night_options[night_id].name, requirement.destination)
            )
            slots.append(_slot("night", night_options[night_id], costs.minutes(previous, night_id)))

        notes = [] if place_ids else [f"第{index + 1}天已无未使用的景点候选。"]
        itinerary.append(
            ItineraryDay(
                day=index + 1,
                date=requirement.departure_date.fromordinal(requirement.departure_date.toordinal() + index),
                slots=slots,
                notes=notes,
            )
        )
    return itinerary


def _fits(option: CandidateOption, period: str) -> bool:
    """景点的开放时间是否适合这个时段；没有开放时间的一律算适合。"""
    window = _opening_minutes(str(option.attributes.get("opening_hours") or ""))
    if window is None:
        return True
    latest_open, earliest_close = SIGHT_PERIOD_WINDOWS[period]
    return (latest_open is None or window[0] <= latest_open) and (earliest_close is None or window[1] >= earliest_close)


def _assign_periods(places: list[CandidateOption]) -> dict[str, CandidateOption]:
    """按路程排好顺序的当天景点 → 上午 / 下午。开放时间对不上时调换，或者把唯一的景点放到下午。"""
    if len(places) >= 2:
        first, second = places[0], places[1]
        if not (_fits(first, "morning") and _fits(second, "afternoon")) and _fits(second, "morning") and _fits(first, "afternoon"):
            first, second = second, first
        return {"morning": first, "afternoon": second}
    if len(places) == 1:
        only = places[0]
        late = not _fits(only, "morning") and _fits(only, "afternoon")
        return {"afternoon" if late else "morning": only}
    return {}


def _pick_dinner(pool: list[str], used: set[str], previous: str | None, costs: TravelCosts) -> str | None:
    if not pool:
        return None
    candidates = [dinner for dinner in pool if dinner not in used]
    if not candidates:
        return None
    return min(candidates, key=lambda dinner: (costs.cost(previous, dinner), candidates.index(dinner)))


def _slot(period: str, option: CandidateOption, travel_minutes: int | None) -> TimeSlot:
    opening = option.attributes.get("opening_hours")
    location = option.attributes.get("location") or option.attributes.get("district")
    return TimeSlot(
        period=period,
        title=option.name,
        description=option.description,
        location=str(location) if location else None,
        travel_minutes=travel_minutes,
        opening_window=str(opening) if opening else None,
        estimated_cost=option.estimated_cost,
    )


# ---------------------------------------------------------------- 3. 检查


def _opening_minutes(window: str | None) -> tuple[int, int] | None:
    match = _OPENING_PATTERN.search(window or "")
    if not match:
        return None
    open_h, open_m, close_h, close_m = (int(value) for value in match.groups())
    return open_h * 60 + open_m, close_h * 60 + close_m


def check_itinerary(requirement: TravelRequirement, results: list[WorkerResult], itinerary: list[ItineraryDay]) -> list[Violation]:
    violations: list[Violation] = []
    costs = TravelCosts(results)
    ids_by_name = {
        option.name: option.id for worker in ("attractions", "food") for option in _options(results, worker)
    }
    for day in itinerary:
        for slot in day.slots:
            window = _opening_minutes(slot.opening_window)
            limits = SIGHT_PERIOD_WINDOWS.get(slot.period)
            if window is None or limits is None:
                continue
            latest_open, earliest_close = limits
            label = "上午" if slot.period == "morning" else "下午"
            if latest_open is not None and window[0] > latest_open:
                violations.append(Violation(code="opening_hours",
                                            message=f"第{day.day}天{label}的{slot.title}开门太晚（{slot.opening_window}）。"))
            if earliest_close is not None and window[1] < earliest_close:
                violations.append(Violation(code="opening_hours",
                                            message=f"第{day.day}天{label}的{slot.title}关门太早（{slot.opening_window}）。"))
        travel = sum(slot.travel_minutes or 0 for slot in day.slots)
        if travel > MAX_DAILY_TRAVEL_MINUTES:
            violations.append(Violation(code="daily_travel",
                                        message=f"第{day.day}天路上耗时约 {travel} 分钟，超过 {MAX_DAILY_TRAVEL_MINUTES} 分钟。"))
            continue
        # 没有真实耗时的路段按坐标估算：防止把相隔几十公里的景点排进同一天。
        # 只看景点之间；去郊外玩一天再回城里吃晚饭是正常安排，不算违规。
        visited = [slot for slot in day.slots if slot.period in ("morning", "afternoon") and slot.title in ids_by_name]
        stops = [ids_by_name[slot.title] for slot in visited]
        estimated = sum(costs.known(first, second) or 0 for first, second in zip(stops, stops[1:]))
        if estimated > MAX_DAILY_TRAVEL_MINUTES:
            names = "、".join(slot.title for slot in visited)
            violations.append(Violation(code="daily_travel",
                                        message=f"第{day.day}天的{names}相距太远，路上估计要 {estimated} 分钟以上。"))
    titles = [slot.title for day in itinerary for slot in day.slots]
    for term in requirement.must_visit:
        if any(named_by(title, [term]) for title in titles):
            continue
        known = any(named_by(option.name, [term]) for worker in ("attractions", "food") for option in _options(results, worker))
        message = f"你点名要去的{term}没有排进行程，请按 id 选上。" if known else f"没查到{term}的资料，没能排进行程，可以换个叫法再试。"
        violations.append(Violation(code="must_visit", message=message))
    return violations


# ---------------------------------------------------------------- 编排入口


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
            itinerary = build_itinerary(requirement, results, selection, costs)
            hotel = next((option for option in _options(results, "hotel") if option.id == selection.hotel_id), None)
        warnings.extend(round_warnings)
        feedback = check_itinerary(requirement, results, itinerary)
        # 规则选点不会根据违规原因调整，重选也是同一个结果，只有 LLM 选点才值得再来一轮。
        if not feedback or by_rules:
            break
    warnings.extend(f"itinerary_constraint:{item.code}:{item.message}" for item in feedback)
    if hotel is not None and itinerary:
        itinerary[0].notes.insert(0, f"{HOTEL_NOTE_PREFIX}{hotel.name}。{hotel.description}")
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


__all__ = [
    "PlaceSelection",
    "SelectedPlace",
    "TravelCosts",
    "Violation",
    "assign_days",
    "build_itinerary",
    "check_itinerary",
    "plan_itinerary",
    "schedule_by_rules",
    "select_places_by_rules",
    "select_places_with_llm",
    "weather_by_date",
]
