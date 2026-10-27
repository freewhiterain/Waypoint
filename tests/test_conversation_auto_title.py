"""会话标题：默认标题在第一条消息时换成这句话的开头。"""

from types import SimpleNamespace

import pytest

from app.api.v1 import chat
from app.schemas.message import MessageCreate


def test_title_is_trimmed_and_truncated():
    assert chat.conversation_title_from("  帮我规划\n成都三天  ") == "帮我规划 成都三天"
    long = "我想下个月带爸妈去杭州玩四天，节奏慢一点，最好住西湖边"
    assert chat.conversation_title_from(long) == long[:20] + "…"


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Session:
    def __init__(self, conversation):
        self.conversation = conversation
        self.commits = 0

    async def execute(self, _statement):
        return _Result(self.conversation)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("title", "expected"), [("新对话", "帮我规划成都三天"), ("我的成都之旅", "我的成都之旅")])
async def test_only_default_titles_are_replaced(monkeypatch, title, expected):
    monkeypatch.setattr(chat, "generate_sse_stream", lambda *args, **kwargs: iter(()))
    conversation = SimpleNamespace(id="c1", title=title)
    session = _Session(conversation)
    user = SimpleNamespace(id="u1")

    await chat.stream_chat("c1", MessageCreate(content="帮我规划成都三天"), None, user, session)

    assert conversation.title == expected
    assert session.commits == (1 if title == "新对话" else 0)
