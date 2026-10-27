"""主动邀请的识别必须走结构化标记，而不是问候语文案全等比较。

生产者建会话时就给开场消息打了 extra_info={"kind": "conversation_offer"}
（app/api/v1/conversations.py），但消费端 _has_recent_proactive_offer 原先只把
_normalize(content) 和硬编码的 "需要我帮你规划一下旅行吗" 做全等比较，标记形同
虚设。后果是改一个字的问候语（哪怕只是把"一下"删掉）就会让用户回"好的"不再进入
collect_trip_requirements，退化成 direct_response —— 没有任何测试或运行时日志
会提示这条链路断了。

中间还断了一环：chat._load_recent_context 只取 role/content，把 extra_info
整个丢掉，所以即使消费端想看标记也拿不到。两端都得修。
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.services import chat_flow as chat
from app.services.main_agent import MainAgentService

_REPO_ROOT = Path(__file__).resolve().parent.parent


class _Row:
    """镜像 app.models.message.Message 上被 _load_recent_context 读到的字段。"""

    def __init__(self, role, content, extra_info=None):
        self.role = role
        self.content = content
        self.extra_info = extra_info
        self.id = uuid.uuid4()
        self.created_at = datetime.now(timezone.utc)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _statement):
        return _Result(self._rows)


@pytest.mark.asyncio
async def test_reworded_offer_is_still_recognized_through_the_marker():
    """问候语改了字，但标记还在，"好的"仍应进入规划流程。"""
    context = [
        {
            "role": "assistant",
            "content": "要不要我帮你安排一趟旅程呢？",
            "kind": "conversation_offer",
        }
    ]

    decision = await MainAgentService(use_llm=False).decide("好的", context)

    assert decision.action == "collect_trip_requirements"


@pytest.mark.asyncio
async def test_legacy_row_without_marker_still_matches_by_text():
    """改动之前入库的会话没有 kind，去掉文案兜底会让老会话突然失效。"""
    context = [{"role": "assistant", "content": "需要我帮你规划一下旅行吗？"}]

    decision = await MainAgentService(use_llm=False).decide("好的", context)

    assert decision.action == "collect_trip_requirements"


@pytest.mark.asyncio
async def test_marker_must_be_on_the_latest_non_system_message():
    """标记只在更早的消息上时不算：用户的"好的"回的是后面那句。"""
    context = [
        {
            "role": "assistant",
            "content": "需要我帮你规划一下旅行吗？",
            "kind": "conversation_offer",
        },
        {"role": "assistant", "content": "也可以先问我一个具体问题。"},
    ]

    decision = await MainAgentService(use_llm=False).decide("好的", context)

    assert decision.action == "direct_response"


@pytest.mark.asyncio
async def test_offer_text_with_trailing_content_and_no_marker_does_not_match():
    """兜底比对必须保持全等，不能放宽成子串——否则带后续内容的消息会误判。"""
    context = [{"role": "assistant", "content": "需要我帮你规划一下旅行吗？现在开始吧。"}]

    decision = await MainAgentService(use_llm=False).decide("好的", context)

    assert decision.action == "direct_response"


@pytest.mark.asyncio
async def test_user_role_with_the_marker_does_not_count():
    """标记来自助手侧；用户消息带上它也不能触发（防止被输入伪造）。"""
    context = [
        {"role": "user", "content": "需要我帮你规划一下旅行吗？", "kind": "conversation_offer"}
    ]

    decision = await MainAgentService(use_llm=False).decide("好的", context)

    assert decision.action == "direct_response"


@pytest.mark.asyncio
async def test_load_recent_context_propagates_the_marker():
    """中间那一环：不把 extra_info.kind 带出来，消费端就永远看不到标记。"""
    session = _Session(
        [
            _Row("assistant", "需要我帮你规划一下旅行吗？", {"kind": "conversation_offer"}),
        ]
    )

    context = await chat._load_recent_context(session, str(uuid.uuid4()), uuid.uuid4())

    assert context[0]["kind"] == "conversation_offer"


@pytest.mark.asyncio
async def test_load_recent_context_tolerates_missing_and_null_extra_info():
    """extra_info 是 Python 侧 default=dict，老行可能是 NULL，不能因此炸掉整轮对话。"""
    session = _Session(
        [
            _Row("assistant", "无标记", None),
            _Row("user", "空标记", {}),
        ]
    )

    context = await chat._load_recent_context(session, str(uuid.uuid4()), uuid.uuid4())

    assert [item["content"] for item in context] == ["空标记", "无标记"]
    assert all("kind" not in item for item in context)


def test_producer_still_stamps_the_marker():
    """契约测试：消费端依赖这个标记，生产者不能悄悄把它删了。"""
    source = (_REPO_ROOT / "app" / "api" / "v1" / "conversations.py").read_text(encoding="utf-8")

    assert '"kind": "conversation_offer"' in source
