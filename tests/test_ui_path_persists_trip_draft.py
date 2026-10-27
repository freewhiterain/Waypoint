"""UI 走的工具结果链路必须持久化 TripDraft。

前端只调 /api/v1/chat/tools/{call_id}/result（web/src/store/chat.ts 的 submitTrip），
而 save_trip_draft 此前只有 /api/v1/planning/tasks 会调用——那个端点
UI 从不访问。结果是：

- GET /api/v1/planning/drafts/{conversation_id} 在真实流程里永远 404；
- chat.py 的 load_trip_draft_context 永远拿不到工作区，后续对话
  失去"上一版行程"这个上下文，只能重新问一遍用户。

草稿在 finish_processing 之后单独落库、失败只记日志：DraftRepository 的
契约是自带 session 的仓库，塞不进上面那个事务；而两者之中"结果已返回给
用户"比"工作区已保存"更不可丢——草稿缺失可以靠下一次规划补上，算好的
行程丢了就是白算一轮。
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from tests.test_trip_form_tool_flow import (
    InMemoryInvocationRepository,
    configure_endpoint,
    endpoint_client,
    invocation,
    parse_sse,
)


@pytest.mark.asyncio
async def test_tool_result_flow_persists_the_trip_draft(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    seeded = invocation(user_id=str(user.id))
    conversation_id = seeded.conversation_id
    repository = InMemoryInvocationRepository([seeded])
    configure_endpoint(monkeypatch, repository)

    from app.api.v1 import tools
    from app.governance.drafts import InMemoryDraftRepository

    drafts = InMemoryDraftRepository()
    monkeypatch.setattr(tools, "PostgresDraftRepository", lambda: drafts, raising=False)

    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "成都", "departure_date": "2026-09-01", "days": 3},
    }

    async with endpoint_client(user) as client:
        response = await client.post("/api/v1/chat/tools/call-1/result", json=payload)

    stream = parse_sse(response)
    assert [event["type"] for event in stream][-1] == "done"
    assert "error" not in [event["type"] for event in stream]

    stored = await drafts.get(str(user.id), conversation_id)

    assert stored is not None, "UI 链路完成规划后必须写入 TripDraft"
    assert stored.requirement["destination"] == "成都"
    assert stored.content["itinerary"]
    assert stored.version == 1


@pytest.mark.asyncio
async def test_draft_persistence_failure_does_not_break_the_stream(monkeypatch):
    """草稿落库失败不能让用户拿不到已经算好的行程。"""
    user = SimpleNamespace(id=uuid4())
    repository = InMemoryInvocationRepository([invocation(user_id=str(user.id))])
    configure_endpoint(monkeypatch, repository)

    from app.api.v1 import tools

    class BrokenDraftRepository:
        async def get(self, user_id, conversation_id):
            return None

        async def save(self, record):
            raise OSError("draft table unavailable")

    monkeypatch.setattr(tools, "PostgresDraftRepository", BrokenDraftRepository, raising=False)

    payload = {
        "tool": "collect_trip_requirements",
        "status": "completed",
        "result": {"destination": "成都", "departure_date": "2026-09-01", "days": 3},
    }

    async with endpoint_client(user) as client:
        response = await client.post("/api/v1/chat/tools/call-1/result", json=payload)

    stream = parse_sse(response)
    types = [event["type"] for event in stream]

    assert "result" in types, "草稿写入失败不应吞掉规划结果"
    assert types[-1] == "done"
