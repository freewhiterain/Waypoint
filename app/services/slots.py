"""槽位的合并、校验与存取。

模型每一轮给出"槽位变化"，这里负责把它并进会话上的槽位状态。模型填的值必须能在
用户这一轮的话里找到依据，找不到就丢弃——宁可少填让表单去问，也不让模型替用户做决定。
"""

from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Protocol
from uuid import UUID

from sqlalchemy import select

from app.models.base import async_session_maker
from app.models.conversation import Conversation
from app.schemas.slots import SlotUpdate, TripSlots
from app.utils.logger import app_logger

SLOTS_KEY = "trip_slots"
# 槽位读写走数据库，数据库慢或没起时不能拖住对话。
_DB_TIMEOUT_SECONDS = 3
_MAX_LIST_ITEMS = 10

# 节假日对应哪一年哪一天有歧义，出发日期留空让用户在表单里填。
_HOLIDAY_WORDS = ("国庆", "春节", "过年", "五一", "劳动节", "元旦", "中秋", "端午", "清明", "寒假", "暑假", "十一")
# “5日游”里的“5日”是天数不是日期，所以单独的“N日/N号”后面不能跟“游/行”。
_EXPLICIT_DATE = re.compile(
    r"\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}|\d{1,2}\s*月\s*\d{1,2}\s*[日号]?|\d{1,2}\s*[日号](?![游行])"
)
_RELATIVE_DATE = re.compile(r"今天|明天|后天|大后天|下?下?周[一二三四五六日天末]|下?个?星期[一二三四五六日天]|礼拜[一二三四五六日天]|这周[一二三四五六日天末]|本周[一二三四五六日天末]|月底|月初")
# 天数必须是“数字 + 时间单位”的说法；“规划一下”里的“一”不算。
_DAYS_CUE = re.compile(r"(\d+|[一二两三四五六七八九十半])\s*(?:天|日|晚|周|个?星期|个?礼拜|个月)|周末|\d+[\s-]*(?:days?|nights?|weeks?)", re.I)
_PARTY_CUE = re.compile(r"人|位|口|大人|孩子|小孩|宝宝|娃|爸|妈|父母|老人|朋友|对象|老婆|老公|媳妇|情侣|夫妻|同事|同学|闺蜜|家人|一起|我们|俩|独自|自己")
# "第二天轻松一点""少安排一个景点""别太累"：才允许记 relaxed_days。
# 预算要有说预算的话："预算 3000""两千块以内""人均一千"。
_BUDGET_CUE = re.compile(r"预算|块|元|以内|不超过|别超过|花费|左右|[0-9]\s*[kKwW千万]|[一二两三四五六七八九十][千万]")
_RELAXED_CUE = re.compile(r"轻松|少安排|少去|少排|别太累|不要太累|太累|慢一点|慢点|悠闲|休息|空一点|松一点")
# 用户用指代接受推荐："就第一个吧""去那里"。
_SELECTION_CUE = re.compile(r"第[一二三四五]个|前者|后者|这个|那个|这里|那里|就它|就这|就那|听你的")


def _mentions_place(text: str, place: str) -> bool:
    """用户原话里按顺序出现了地点名中至少 3 个字（短名字要全部出现）。"""
    if place in text:
        return True
    previous = [0] * (len(place) + 1)
    for char in text:
        current = [0]
        for index, target in enumerate(place, start=1):
            current.append(previous[index - 1] + 1 if char == target else max(previous[index], current[-1]))
        previous = current
    return previous[-1] >= min(3, len(place))


def _dedupe(items: list[str]) -> list[str]:
    cleaned = [item.strip() for item in items if isinstance(item, str) and 0 < len(item.strip()) <= 10]
    return list(dict.fromkeys(cleaned))[:_MAX_LIST_ITEMS]


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value.strip()) if value else None
    except ValueError:
        return None


def apply_slot_update(
    slots: TripSlots,
    update: SlotUpdate,
    *,
    user_text: str,
    today: date | None = None,
) -> tuple[TripSlots, list[str]]:
    """把模型给出的槽位变化并进当前槽位，返回 (新槽位, 被丢弃的字段)。

    新值覆盖旧值（用户改主意时以最新的为准）。每个字段都要通过各自的依据检查。
    """
    today = today or date.today()
    values = slots.model_dump()
    dropped: list[str] = []

    destination = (update.destination or "").strip()
    if destination:
        suggested = slots.suggested_destinations
        said_by_user = destination in user_text
        accepted_suggestion = destination in suggested and (
            len(suggested) == 1 or bool(_SELECTION_CUE.search(user_text))
        )
        if len(destination) <= 80 and (said_by_user or accepted_suggestion):
            if destination != slots.destination:
                # 换了城市就是另一趟行程，上一版的修改意见不再适用。
                values["must_visit"] = []
                values["budget"] = None
                values["avoid_places"] = []
                values["relaxed_days"] = []
            values["destination"] = destination
        else:
            dropped.append("destination")

    departure = _parse_date(update.departure_date)
    if update.departure_date:
        has_explicit = bool(_EXPLICIT_DATE.search(user_text))
        has_relative = bool(_RELATIVE_DATE.search(user_text))
        holiday_only = any(word in user_text for word in _HOLIDAY_WORDS) and not has_explicit
        if departure is not None and departure >= today and (has_explicit or has_relative) and not holiday_only:
            values["departure_date"] = departure
        else:
            dropped.append("departure_date")

    if update.days is not None:
        if 1 <= update.days <= 30 and _DAYS_CUE.search(user_text):
            values["days"] = update.days
        else:
            dropped.append("days")

    origin = (update.origin or "").strip()
    if origin:
        if origin in user_text and origin != values.get("destination"):
            values["origin"] = origin
        else:
            dropped.append("origin")

    party_mentioned = bool(_PARTY_CUE.search(user_text))
    if update.adults is not None:
        if 1 <= update.adults <= 30 and party_mentioned:
            values["adults"] = update.adults
        else:
            dropped.append("adults")
    if update.children is not None:
        if 0 <= update.children <= 20 and party_mentioned:
            values["children"] = update.children
        else:
            dropped.append("children")

    # 不想去的地点必须是用户提到过的，防止模型替用户排除地点。模型常把"熊猫基地"
    # 补全成上下文里的全名"成都大熊猫繁育研究基地"，所以按公共子序列判断而不是原样出现。
    avoid = [place.strip() for place in update.avoid_places if place.strip()]
    kept_avoid = [place for place in avoid if len(place) <= 30 and _mentions_place(user_text, place)]
    if len(kept_avoid) < len(avoid):
        dropped.append("avoid_places")
    values["avoid_places"] = list(dict.fromkeys([*values.get("avoid_places", []), *kept_avoid]))[:_MAX_LIST_ITEMS]

    # 点名要去的地点同样要用户亲口说过；后来又说不去的，以不去为准。
    must = [place.strip() for place in update.must_visit if place.strip()]
    kept_must = [place for place in must if len(place) <= 30 and _mentions_place(user_text, place)]
    if len(kept_must) < len(must):
        dropped.append("must_visit")
    merged = list(dict.fromkeys([*values.get("must_visit", []), *kept_must]))
    values["must_visit"] = [
        place for place in merged
        if not any(place in avoided or avoided in place for avoided in values["avoid_places"])
    ][:_MAX_LIST_ITEMS]

    if update.budget is not None:
        if 0 < update.budget <= 10_000_000 and _BUDGET_CUE.search(user_text):
            values["budget"] = float(update.budget)
        else:
            dropped.append("budget")

    if update.relaxed_days:
        if _RELAXED_CUE.search(user_text):
            days = [day for day in update.relaxed_days if isinstance(day, int) and 1 <= day <= 30]
            values["relaxed_days"] = sorted(set(values.get("relaxed_days", [])) | set(days))
        else:
            dropped.append("relaxed_days")

    values["styles"] = _dedupe([*slots.styles, *update.styles])
    values["food_preferences"] = _dedupe([*slots.food_preferences, *update.food_preferences])
    return TripSlots(**values), dropped


def remember_suggestions(slots: TripSlots, answer: str, known_destinations: set[str]) -> TripSlots:
    """记下模型这次回答里推荐到的城市，作为候选；按在回答中出现的先后排序。"""
    found = sorted(
        (answer.index(name), name) for name in known_destinations if name in answer and name != slots.destination
    )
    return slots.model_copy(update={"suggested_destinations": [name for _position, name in found][:5]})


class SlotRepository(Protocol):
    async def get(self, conversation_id: str) -> TripSlots: ...

    async def save(self, conversation_id: str, slots: TripSlots) -> None: ...


class InMemorySlotRepository:
    def __init__(self) -> None:
        self._slots: dict[str, TripSlots] = {}

    async def get(self, conversation_id: str) -> TripSlots:
        return self._slots.get(conversation_id, TripSlots()).model_copy(deep=True)

    async def save(self, conversation_id: str, slots: TripSlots) -> None:
        self._slots[conversation_id] = slots.model_copy(deep=True)


class PostgresSlotRepository:
    """槽位存在会话记录的 extra_info 里，不新建表。"""

    def __init__(self, session_factory=async_session_maker):
        self.session_factory = session_factory

    async def get(self, conversation_id: str) -> TripSlots:
        async with self.session_factory() as session:
            extra_info = await session.scalar(
                select(Conversation.extra_info).where(Conversation.id == UUID(conversation_id))
            )
        raw = (extra_info or {}).get(SLOTS_KEY) if isinstance(extra_info, dict) else None
        if not isinstance(raw, dict):
            return TripSlots()
        try:
            return TripSlots.model_validate(raw)
        except ValueError as exc:
            app_logger.warning(f"会话槽位数据无效，按空槽位处理: {exc}")
            return TripSlots()

    async def save(self, conversation_id: str, slots: TripSlots) -> None:
        async with self.session_factory() as session, session.begin():
            conversation = await session.get(Conversation, UUID(conversation_id))
            if conversation is None:
                return
            # JSON 列不追踪原地修改，必须整体换一个新 dict 才会写回。
            conversation.extra_info = {
                **(conversation.extra_info or {}),
                SLOTS_KEY: slots.model_dump(mode="json"),
            }


async def load_slots(repository: SlotRepository, conversation_id: str) -> TripSlots:
    """读槽位失败不该打断对话：当作还没有任何槽位。"""
    try:
        return await asyncio.wait_for(repository.get(conversation_id), timeout=_DB_TIMEOUT_SECONDS)
    except Exception as exc:
        app_logger.warning(f"读取会话槽位失败，按空槽位处理: {type(exc).__name__}: {exc}")
        return TripSlots()


async def store_slots(repository: SlotRepository, conversation_id: str, slots: TripSlots) -> None:
    try:
        await asyncio.wait_for(repository.save(conversation_id, slots), timeout=_DB_TIMEOUT_SECONDS)
    except Exception as exc:
        app_logger.warning(f"保存会话槽位失败，本轮槽位不会带到下一轮: {type(exc).__name__}: {exc}")


async def confirm_slots(repository: SlotRepository, conversation_id: str | None, form_values: dict) -> None:
    """用户在表单或确认弹窗里确认过的三个槽以用户的为准，写回会话槽位。"""
    if not conversation_id:
        return
    slots = await load_slots(repository, conversation_id)
    try:
        confirmed = TripSlots(
            **{
                **slots.model_dump(),
                "destination": form_values.get("destination"),
                "departure_date": form_values.get("departure_date"),
                "days": form_values.get("days"),
                "suggested_destinations": [],
            }
        )
    except ValueError as exc:
        app_logger.warning(f"确认的槽位无法写回: {exc}")
        return
    await store_slots(repository, conversation_id, confirmed)
