"""会话槽位：每一轮都填、跨线路累积、模型填的值必须有依据。"""

import json
import uuid
from datetime import date

import pytest

from app.api.v1 import chat
from app.schemas.slots import SlotUpdate, TripSlots
from app.schemas.tools import MainAgentDecision
from app.services.slots import (
    InMemorySlotRepository,
    apply_slot_update,
    confirm_slots,
    remember_suggestions,
)

from tests.test_chat_main_agent_flow import (
    RecordingSession,
    SessionAwareRecordingRepository,
    session_factory,
)

TODAY = date(2026, 10, 11)
CITIES = {"成都", "西安", "北京"}


def merge(text, slots=None, **update):
    return apply_slot_update(slots or TripSlots(), SlotUpdate(**update), user_text=text, today=TODAY)


def test_required_slots_decide_completeness_and_form_prefill():
    slots = TripSlots(destination="成都", days=3, adults=2, styles=["美食"])

    assert slots.missing() == ["departure_date"]
    assert slots.form_values() == {"destination": "成都", "days": 3}
    assert slots.requirement_context() == {"adults": 2, "styles": ["美食"]}
    assert TripSlots(destination="成都", days=3, departure_date=date(2026, 11, 14)).is_complete()


def test_values_stated_by_the_user_are_filled_and_later_ones_override():
    slots, dropped = merge(
        "明天去成都5日游，两个人", destination="成都", days=5, departure_date="2026-10-12", adults=2
    )
    assert (slots.destination, slots.days, slots.departure_date, slots.adults) == ("成都", 5, date(2026, 10, 12), 2)
    assert dropped == []

    changed, _ = merge("还是去西安吧，玩三天", slots, destination="西安", days=3)
    assert (changed.destination, changed.days) == ("西安", 3)
    assert changed.adults == 2  # 没提到的保持不变


@pytest.mark.parametrize(
    ("text", "update", "dropped_field"),
    [
        ("帮我规划一下", {"destination": "成都"}, "destination"),  # 用户没说过成都
        ("规划一下成都", {"destination": "成都", "days": 3}, "days"),  # "一下"不是天数
        ("去成都玩", {"destination": "成都", "departure_date": "2026-10-20"}, "departure_date"),  # 没提日期
        ("去年10月1日去过成都", {"departure_date": "2025-10-01"}, "departure_date"),  # 过去的日期
        ("成都5日游", {"departure_date": "2026-10-05"}, "departure_date"),  # "5日"是天数不是日期
        ("去成都", {"adults": 4}, "adults"),  # 没提人数
        ("从成都出发去成都", {"destination": "成都", "origin": "成都"}, "origin"),  # 出发地不能等于目的地
    ],
)
def test_ungrounded_values_are_dropped(text, update, dropped_field):
    slots, dropped = merge(text, **update)

    assert dropped_field in dropped
    assert getattr(slots, dropped_field) is None


@pytest.mark.parametrize("text", ["国庆去成都玩", "春节出发", "五一假期想出去"])
def test_holiday_wording_never_becomes_a_date(text):
    slots, dropped = merge(text, departure_date="2027-10-01")

    assert slots.departure_date is None and dropped == ["departure_date"]


def test_explicit_date_wins_even_when_a_holiday_is_mentioned():
    slots, _ = merge("国庆后，10月20日出发", departure_date="2026-10-20")

    assert slots.departure_date == date(2026, 10, 20)


def test_a_single_recommended_city_is_filled_once_the_user_goes_along():
    suggested = remember_suggestions(TripSlots(days=5), "推荐你去成都，美食很多。", CITIES)
    assert suggested.suggested_destinations == ["成都"]

    slots, dropped = merge("那帮我规划一下", suggested, destination="成都")

    assert slots.destination == "成都" and dropped == []


def test_several_recommended_cities_need_an_explicit_choice():
    suggested = remember_suggestions(TripSlots(), "可以考虑西安，或者成都。", CITIES)
    assert suggested.suggested_destinations == ["西安", "成都"]  # 按回答里出现的先后

    undecided, dropped = merge("那帮我规划一下", suggested, destination="西安")
    assert undecided.destination is None and dropped == ["destination"]

    by_pointer, _ = merge("就第二个吧", suggested, destination="成都")
    by_name, _ = merge("去成都吧", suggested, destination="成都")
    assert by_pointer.destination == by_name.destination == "成都"


def test_interests_accumulate_without_duplicates():
    slots, _ = merge("喜欢美食", styles=["美食"])
    slots, _ = merge("也想看熊猫，不吃辣", slots, styles=["熊猫", "美食"], food_preferences=["不吃辣"])

    assert slots.styles == ["美食", "熊猫"]
    assert slots.food_preferences == ["不吃辣"]


@pytest.mark.asyncio
async def test_confirmed_form_values_are_written_back_and_clear_suggestions():
    repository = InMemorySlotRepository()
    await repository.save("c1", TripSlots(destination="成都", adults=2, suggested_destinations=["西安"]))

    await confirm_slots(repository, "c1", {"destination": "西安", "departure_date": "2026-11-14", "days": 4})

    slots = await repository.get("c1")
    assert (slots.destination, slots.departure_date, slots.days) == ("西安", date(2026, 11, 14), 4)
    assert slots.adults == 2 and slots.suggested_destinations == []


# ---------------------------------------------------------------- 对话层：跨线路累积


class ScriptedAgent:
    """按顺序返回预设的决策，并记下每一轮收到的槽位。"""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.seen_slots = []

    async def decide(self, message, context, slots=None):
        self.seen_slots.append(slots)
        return self.decisions.pop(0)


async def run_turn(conversation_id, message, repository):
    return [
        json.loads(frame.removeprefix("data: ").strip())
        async for frame in chat.generate_sse_stream(
            conversation_id, message, str(uuid.uuid4()), slot_repository=repository
        )
    ]


@pytest.mark.asyncio
async def test_slots_filled_on_route_one_carry_into_the_planning_form(monkeypatch):
    agent = ScriptedAgent(
        [
            MainAgentDecision(action="recommend_destination", reason="r", slot_update=SlotUpdate(days=5)),
            MainAgentDecision(
                action="collect_trip_requirements", reason="p", slot_update=SlotUpdate(destination="成都")
            ),
        ]
    )
    monkeypatch.setattr(chat, "async_session_maker", session_factory(RecordingSession([])))
    monkeypatch.setattr(chat, "MainAgentService", lambda: agent)
    monkeypatch.setattr(chat, "PostgresToolInvocationRepository", lambda: SessionAwareRecordingRepository())

    async def recommend(_message):
        return "国庆假期推荐你去成都，美食和熊猫都不错。"

    monkeypatch.setattr(chat, "answer_open_question", recommend)
    repository = InMemorySlotRepository()
    conversation_id = str(uuid.uuid4())

    first = await run_turn(conversation_id, "国庆放5天假，有没有推荐好玩的地方？", repository)
    second = await run_turn(conversation_id, "那帮我规划一下", repository)

    assert [event["type"] for event in first] == ["token", "done"]
    after_first = agent.seen_slots[1]  # 第二轮开始时模型看到的槽位
    assert after_first.days == 5 and after_first.suggested_destinations == ["成都"]
    assert after_first.destination is None  # 推荐只是候选，还没填

    # 第二轮：目的地和天数已有，表单只会问缺的出发日期。
    assert second[0]["type"] == "tool_call"
    assert second[0]["arguments"] == {"initial_values": {"destination": "成都", "days": 5}}


@pytest.mark.asyncio
async def test_slot_storage_failure_does_not_break_the_turn(monkeypatch):
    class BrokenRepository:
        async def get(self, conversation_id):
            raise ConnectionError("database down")

        async def save(self, conversation_id, slots):
            raise ConnectionError("database down")

    agent = ScriptedAgent(
        [MainAgentDecision(action="collect_trip_requirements", reason="p", slot_update=SlotUpdate(destination="成都"))]
    )
    monkeypatch.setattr(chat, "async_session_maker", session_factory(RecordingSession([])))
    monkeypatch.setattr(chat, "MainAgentService", lambda: agent)
    monkeypatch.setattr(chat, "PostgresToolInvocationRepository", lambda: SessionAwareRecordingRepository())

    events = await run_turn(str(uuid.uuid4()), "帮我规划成都旅行", BrokenRepository())

    assert events[0]["type"] == "tool_call"
    assert events[0]["arguments"] == {"initial_values": {"destination": "成都"}}


def test_english_day_count_is_accepted_as_a_days_cue():
    slots, dropped = apply_slot_update(
        TripSlots(), SlotUpdate(days=3), user_text="Plan a 3-day trip to Chengdu for me"
    )

    assert slots.days == 3 and dropped == []
