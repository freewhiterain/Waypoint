"""排程用到的常量和对候选地点的判断：哪些算同一处、哪些占一整天、哪天要轻松、餐厅几点营业。"""

from __future__ import annotations

from statistics import median
import re

from pydantic import BaseModel, Field

from app.places.catalog import is_theme_park, straight_line_km
from app.schemas.planning import CandidateOption, TravelRequirement, WorkerResult


MAX_ROUNDS = 3


# 强制工具调用下模型偶尔返回空结果（解析为 None），先重试，仍失败才退回规则选点。
SELECTION_ATTEMPTS = 2


ATTRACTIONS_PER_DAY = 2


MAX_DAILY_TRAVEL_MINUTES = 180


# 同一天两个景点之间最多能接受的路程（分钟）。
MAX_HOP_MINUTES = 75


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


# 从当天最后一站到晚饭，超过这个代价就换附近的备选，换不了就反馈给模型重选。
MAX_DINNER_MINUTES = 45


# 午餐时段要开着：11:30 前开门、13:30 后才关门。
_LUNCH_WINDOW = (11 * 60 + 30, 13 * 60 + 30)


# 晚饭要在这段时间营业：只做午市、下午四点半就打烊的自助餐不能排成晚饭。
_DINNER_WINDOW = (18 * 60, 20 * 60)


class SelectedPlace(BaseModel):
    id: str
    reason: str = Field(default="", description="选它的理由，不超过 15 个字")
    preferred_day: int | None = Field(default=None, ge=1)


class PlaceSelection(BaseModel):
    """LLM 选点的结构化输出：只能引用候选 id。"""

    hotel_id: str | None = None
    attractions: list[SelectedPlace] = Field(default_factory=list)
    dinners: list[SelectedPlace] = Field(default_factory=list)


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


# 主题乐园、大型动物园一玩就是一整天：当天不再排别的景点。地点库的建议游玩时长封顶 4 小时
# （迪士尼也只写 3.5），靠不住，按名字认。
FULL_DAY_NOTE = "这里一玩就是一整天，当天不再安排别的景点。"


def is_full_day_name(name: str) -> bool:
    return is_theme_park(name)


def is_full_day(option: CandidateOption) -> bool:
    """主题乐园、大型动物园：一玩一整天，当天只排它。"""
    return is_full_day_name(option.name)


def is_excursion(option: CandidateOption, remote: set[str] | None = None) -> bool:
    """远郊景点：往返就要大半天，单独给一天；挨着的几处可以同一天。"""
    return option.id in (remote or set()) or any(word in option.name for word in _FAMOUS_OUTSKIRTS)


# 没有坐标时（联网搜索的城市）认不出远郊，按名字补几处常见的：兵马俑在临潼、长城在延庆怀柔。
_FAMOUS_OUTSKIRTS = ("兵马俑", "秦始皇帝陵", "长城", "千岛湖", "青城山", "峨眉山", "乐山大佛", "黄山")


# 住宿备注只留第一句：联网搜到的酒店描述常是整段订房评论。
_HOTEL_SUMMARY_CHARS = 60


def _hotel_note(hotel: CandidateOption) -> str:
    """地点库的酒店有结构化的价格和特色；联网搜到的多半只有一段攻略或评论残句，不如不放。"""
    if hotel.estimated_cost is None:
        return "价格和房态请在订房平台确认。"
    return _hotel_summary(hotel.description)


def _hotel_summary(description: str) -> str:
    first = description.split("。")[0].strip()
    return (first[:_HOTEL_SUMMARY_CHARS] + "…") if len(first) > _HOTEL_SUMMARY_CHARS else (first + "。" if first else "")


# 离城市中心（候选景点坐标的中位数）超过这个直线距离的，往返就要大半天，按一整天安排：
# 成都的都江堰 56km、北京的八达岭 57km、上海的朱家角 44km。
OUTSKIRTS_KM = 35.0


def _located(options: list[CandidateOption]) -> list[tuple[str, tuple[float, float]]]:
    return [
        (option.id, (float(option.attributes["latitude"]), float(option.attributes["longitude"])))
        for option in options
        if isinstance(option.attributes.get("latitude"), (int, float))
        and isinstance(option.attributes.get("longitude"), (int, float))
    ]


def outskirts(results: list[WorkerResult]) -> set[str]:
    """离城市中心超过 OUTSKIRTS_KM 的景点。

    中心取所有候选（景点、餐厅、酒店）坐标的中位数：只看景点时，候选里远郊一多，
    中位数会被拉到城郊之间，谁都不算远。餐厅和酒店大多在市区，能把中心稳住。
    """
    reference = _located([option for worker in ("attractions", "food", "hotel") for option in _options(results, worker)])
    if len(reference) < 3:
        return set()
    center = (median(point[0] for _, point in reference), median(point[1] for _, point in reference))
    return {place_id for place_id, point in _located(_options(results, "attractions"))
            if straight_line_km(center, point) > OUTSKIRTS_KM}


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
# 模型常把"爸妈年纪大了，别太赶"概括成"带父母"，所以同行长辈本身也算要轻松。
_RELAXED_STYLE_WORDS = ("轻松", "悠闲", "休闲", "慢", "不累", "别太累", "赶", "老人", "腿脚", "年纪大", "长辈", "父母", "爸妈")


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


def _keyword_score(option: CandidateOption, requirement: TravelRequirement) -> int:
    text = f"{option.name}{option.description}"
    keywords = [*requirement.styles, *requirement.food_preferences]
    return sum(1 for keyword in keywords if keyword and keyword in text)


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


# 模型选的酒店到所选景点的平均路程比最近的那家多出这么多（分钟量级），就换成最近的。
HOTEL_DETOUR_MINUTES = 15


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


def wanted_dishes(requirement: TravelRequirement) -> list[str]:
    """用户点名想吃的东西（火锅、烤鸭、早茶……），忌口的说法不算。"""
    said = [item for item in [*requirement.styles, *requirement.food_preferences]
            if not item.startswith(("不", "忌", "别", "少"))]
    return [word for word in _FOOD_STYLE_WORDS[1:] if any(word in item for item in said)]


def _serves(option: CandidateOption, dish: str) -> bool:
    text = f"{option.name}{option.attributes.get('type') or ''}{option.attributes.get('recommended_food') or ''}"
    return dish in text


def _fits(option: CandidateOption, period: str) -> bool:
    """景点的开放时间是否适合这个时段；没有开放时间的一律算适合。"""
    window = _opening_minutes(str(option.attributes.get("opening_hours") or ""))
    if window is None:
        return True
    latest_open, earliest_close = SIGHT_PERIOD_WINDOWS[period]
    return (latest_open is None or window[0] <= latest_open) and (earliest_close is None or window[1] >= earliest_close)


_MORNING_WORDS = ("熊猫", "动物园", "野生动物", "早市")


def _morning_best(option: CandidateOption) -> bool:
    """动物上午最活跃（熊猫基地下午基本在睡），这类地点尽量排上午。"""
    return any(word in option.name for word in _MORNING_WORDS)


def _opening_minutes(window: str | None) -> tuple[int, int] | None:
    match = _OPENING_PATTERN.search(window or "")
    if not match:
        return None
    open_h, open_m, close_h, close_m = (int(value) for value in match.groups())
    return open_h * 60 + open_m, close_h * 60 + close_m
